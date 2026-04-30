import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr
from numpy.polynomial.hermite import hermgauss


def _inverse_softplus(value):
    value = torch.tensor(float(value), dtype=torch.float32)
    return torch.log(torch.expm1(value))


class BaseNoiseSchedule(torch.nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = float(eps)

    def _eps_like(self, ref):
        return ref.new_tensor(self.eps)

    def _buffer_like(self, name, ref):
        return getattr(self, name).to(device=ref.device, dtype=ref.dtype)

    def gamma(self, tau):
        raise NotImplementedError

    def snr(self, tau):
        return torch.exp(-self.gamma(tau))

    def forward(self, tau):
        gamma = self.gamma(tau)
        return gamma, self.snr_prime(tau, gamma)

    def snr_prime(self, tau, gamma=None):
        raise NotImplementedError

    def snr_at(self, tau_value, ref):
        tau = ref.new_tensor(tau_value)
        return self.snr(tau)

    @torch.no_grad()
    def tau_from_snr(self, snr):
        target = snr.clamp_min(self._eps_like(snr))
        lo = torch.zeros_like(target)
        hi = torch.ones_like(target)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            mid_snr = self.snr(mid)
            lo = torch.where(mid_snr < target, mid, lo)
            hi = torch.where(mid_snr >= target, mid, hi)
        return 0.5 * (lo + hi)

    def importance_sample(self, unit, t_min=0.0, t_max=1.0):
        snr_min = self.snr_at(t_min, unit).clamp_min(self._eps_like(unit))
        snr_max = self.snr_at(t_max, unit).clamp_min(self._eps_like(unit))
        snr_span = (snr_max - snr_min).clamp_min(self._eps_like(unit))
        snr = (snr_min + unit * snr_span).clamp_min(self._eps_like(unit))
        gamma = -torch.log(snr)
        tau = self.tau_from_snr(snr)
        vlb_weight = torch.ones_like(unit) * snr_span
        return tau, gamma, vlb_weight


class LinearGammaSchedule(BaseNoiseSchedule):
    def __init__(self, gamma_min=-5.0, gamma_max=5.0, eps=1e-12):
        super().__init__(eps=eps)
        self.register_buffer('gamma_min', torch.tensor(float(gamma_min)))
        self.register_buffer('gamma_max', torch.tensor(float(gamma_max)))

    def gamma(self, tau):
        gamma_min = self._buffer_like('gamma_min', tau)
        gamma_max = self._buffer_like('gamma_max', tau)
        return gamma_max + tau * (gamma_min - gamma_max)

    def snr_prime(self, tau, gamma=None):
        if gamma is None:
            gamma = self.gamma(tau)
        gamma_min = self._buffer_like('gamma_min', tau)
        gamma_max = self._buffer_like('gamma_max', tau)
        return torch.exp(-gamma) * (gamma_max - gamma_min)

    @torch.no_grad()
    def tau_from_snr(self, snr):
        gamma = -torch.log(snr.clamp_min(self._eps_like(snr)))
        gamma_min = self._buffer_like('gamma_min', snr)
        gamma_max = self._buffer_like('gamma_max', snr)
        return ((gamma - gamma_max) / (gamma_min - gamma_max)).clamp(0.0, 1.0)


class SNRPowerSchedule(BaseNoiseSchedule):
    def __init__(self, C=17.276323318481445, p=0.43169814348220825,
                 eps=1e-6):
        super().__init__(eps=eps)
        self.register_buffer('C', torch.tensor(float(C)))
        self.register_buffer('p', torch.tensor(float(p)))

    def _safe_terms(self, tau):
        eps = self._eps_like(tau)
        tau_safe = torch.maximum(torch.minimum(tau, 1.0 - eps), eps)
        L = -torch.log1p(-tau_safe)
        L_safe = torch.maximum(L, eps)
        return tau_safe, L_safe

    def gamma(self, tau):
        _, L_safe = self._safe_terms(tau)
        C = self._buffer_like('C', tau)
        p = self._buffer_like('p', tau)
        return -torch.log(C) - p * torch.log(L_safe)

    def snr_prime(self, tau, gamma=None):
        if gamma is None:
            gamma = self.gamma(tau)
        tau_safe, L_safe = self._safe_terms(tau)
        p = self._buffer_like('p', tau)
        denom = torch.maximum((1.0 - tau_safe) * L_safe,
                              self._eps_like(tau))
        return torch.exp(-gamma) * p / denom

    @torch.no_grad()
    def tau_from_snr(self, snr):
        snr = snr.clamp_min(self._eps_like(snr))
        C = self._buffer_like('C', snr)
        p = self._buffer_like('p', snr)
        log_L = (torch.log(snr) - torch.log(C)) / p
        L = torch.exp(log_L.clamp(max=80.0))
        return (-torch.expm1(-L)).clamp(0.0, 1.0)


class PositiveLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, init_weight=1e-3):
        super().__init__()
        raw = _inverse_softplus(init_weight)
        self.raw_weight = torch.nn.Parameter(
            torch.full((out_features, in_features), raw))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(x, F.softplus(self.raw_weight), self.bias)


class LearnedVDMSchedule(LinearGammaSchedule):
    def __init__(self, gamma_min=-5.0, gamma_max=5.0, hidden_size=1024,
                 eps=1e-12):
        super().__init__(gamma_min=gamma_min, gamma_max=gamma_max, eps=eps)
        self.l1 = PositiveLinear(1, 1, init_weight=1.0)
        self.l2 = PositiveLinear(1, int(hidden_size), init_weight=1e-3)
        self.l3 = PositiveLinear(int(hidden_size), 1, init_weight=1e-3)

    def _raw_gamma(self, tau):
        shape = tau.shape
        h1 = self.l1(tau.reshape(-1, 1))
        residual = self.l3(torch.sigmoid(self.l2(h1)))
        return (h1 + residual).reshape(shape)

    def gamma(self, tau):
        raw = self._raw_gamma(tau)
        raw0 = self._raw_gamma(torch.zeros((), device=tau.device,
                                           dtype=tau.dtype))
        raw1 = self._raw_gamma(torch.ones((), device=tau.device,
                                          dtype=tau.dtype))
        progress = (raw - raw0) / (raw1 - raw0).clamp_min(self.eps)
        gamma_min = self._buffer_like('gamma_min', tau)
        gamma_max = self._buffer_like('gamma_max', tau)
        return gamma_max + progress * (gamma_min - gamma_max)

    def forward(self, tau):
        grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            tau_req = tau.detach().requires_grad_(True)
            gamma = self.gamma(tau_req)
            dgamma_dtau = torch.autograd.grad(
                gamma.sum(), tau_req, create_graph=grad_enabled)[0]
            snr_prime = torch.exp(-gamma) * (-dgamma_dtau).clamp_min(
                self._eps_like(tau_req))
        if not grad_enabled:
            gamma = gamma.detach()
            snr_prime = snr_prime.detach()
        return gamma, snr_prime


def _compute_argmax_tau(gamma, K, n_gh=100):
    gamma = np.asarray(gamma, dtype=np.float64)
    margin = np.exp(-0.5 * gamma)
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x
    log_cdf = log_ndtr(z_nodes[None, :] + margin[:, None])
    q_correct = np.sum(w * np.exp((K - 1) * log_cdf), axis=-1)
    tau = K / (K - 1.0) * (q_correct - 1.0 / K)
    tau = tau - gamma * 1e-10
    return np.clip(tau, 0.0, 1.0)


class ArgmaxUncertaintySchedule(BaseNoiseSchedule):
    def __init__(self, vocab_size, gamma_min=-5.0, gamma_max=5.0,
                 n_points=10000, n_gh=100, eps=1e-12):
        super().__init__(eps=eps)
        self.vocab_size = int(vocab_size)
        self.gamma_min_value = float(gamma_min)
        self.gamma_max_value = float(gamma_max)
        gamma_vals = np.linspace(
            self.gamma_min_value, self.gamma_max_value, int(n_points),
            dtype=np.float64)
        tau_vals = _compute_argmax_tau(gamma_vals, self.vocab_size,
                                       n_gh=int(n_gh))
        self.lut_gamma2tau = CubicSpline(gamma_vals, tau_vals)

        order = np.argsort(tau_vals)
        tau_sorted = tau_vals[order]
        gamma_sorted = gamma_vals[order]
        unique_tau, unique_idx = np.unique(tau_sorted, return_index=True)
        unique_gamma = gamma_sorted[unique_idx]
        interior = (unique_tau > 0.0) & (unique_tau < 1.0)
        unique_tau = unique_tau[interior]
        unique_gamma = unique_gamma[interior]
        tau_augmented = np.concatenate(([0.0], unique_tau, [1.0]))
        gamma_augmented = np.concatenate((
            [self.gamma_max_value], unique_gamma, [self.gamma_min_value]))
        tau_augmented, unique_idx = np.unique(tau_augmented,
                                              return_index=True)
        gamma_augmented = gamma_augmented[unique_idx]
        self.lut_tau2gamma = CubicSpline(tau_augmented, gamma_augmented)

    def _spline_to_tensor(self, values, lut, ref, clip_min=None,
                          clip_max=None):
        values_np = values.detach().cpu().numpy()
        if clip_min is not None or clip_max is not None:
            values_np = np.clip(values_np, clip_min, clip_max)
        result = np.asarray(lut(values_np))
        return torch.from_numpy(result).to(device=ref.device, dtype=ref.dtype)

    def gamma(self, tau):
        return self._spline_to_tensor(
            tau, self.lut_tau2gamma, tau, clip_min=0.0, clip_max=1.0)

    def snr_prime(self, tau, gamma=None):
        if gamma is None:
            gamma = self.gamma(tau)
        dgamma_dtau = self._spline_to_tensor(
            tau, self.lut_tau2gamma.derivative(), tau,
            clip_min=0.0, clip_max=1.0)
        return torch.exp(-gamma) * (-dgamma_dtau).clamp_min(
            self._eps_like(tau))

    @torch.no_grad()
    def tau_from_snr(self, snr):
        gamma = -torch.log(snr.clamp_min(self._eps_like(snr)))
        return self._spline_to_tensor(
            gamma, self.lut_gamma2tau, snr,
            clip_min=self.gamma_min_value,
            clip_max=self.gamma_max_value).clamp(0.0, 1.0)


def _finite_gamma_bounds(gamma_min, gamma_max):
    if gamma_min is None:
        gamma_min = -5.0
    if gamma_max is None:
        gamma_max = 5.0
    return float(gamma_min), float(gamma_max)


def build_noise_schedule(schedule_type, vocab_size, gamma_min=None,
                         gamma_max=None, C=17.276323318481445,
                         p=0.43169814348220825, eps=1e-12,
                         hidden_size=1024, n_points=10000, n_gh=100):
    schedule_type = str(schedule_type)
    if schedule_type in {'linear', 'linear_gamma'}:
        gamma_min, gamma_max = _finite_gamma_bounds(gamma_min, gamma_max)
        return LinearGammaSchedule(gamma_min=gamma_min, gamma_max=gamma_max,
                                   eps=eps)
    if schedule_type in {'learned_vdm', 'learned'}:
        gamma_min, gamma_max = _finite_gamma_bounds(gamma_min, gamma_max)
        return LearnedVDMSchedule(gamma_min=gamma_min, gamma_max=gamma_max,
                                  hidden_size=hidden_size, eps=eps)
    if schedule_type in {'argmax_uncertainty', 'argmax', 'tau_progress'}:
        gamma_min, gamma_max = _finite_gamma_bounds(gamma_min, gamma_max)
        return ArgmaxUncertaintySchedule(
            vocab_size=vocab_size, gamma_min=gamma_min, gamma_max=gamma_max,
            n_points=n_points, n_gh=n_gh, eps=eps)
    if schedule_type in {'snr_power', 'power_snr'}:
        return SNRPowerSchedule(C=C, p=p, eps=eps)
    raise ValueError(f"Unknown noise schedule type: {schedule_type}")
