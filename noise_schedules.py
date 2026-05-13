import ast
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline, PchipInterpolator
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


def _coerce_float_sequence(values, name):
    if _sequence_is_missing(values):
        return None
    if isinstance(values, str):
        stripped = values.strip()
        if stripped.startswith('[') or stripped.startswith('('):
            values = ast.literal_eval(stripped)
        else:
            values = stripped.split(',')
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    values = list(values)
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    return [float(value) for value in values]


def _sequence_is_missing(values):
    if values is None:
        return True
    if isinstance(values, str):
        stripped = values.strip()
        return stripped == '' or stripped.lower() in {'none', 'null'}
    return False


def _coerce_optional_float(value):
    if _sequence_is_missing(value):
        return None
    return float(value)


def _strictly_increasing_tensor(values, name):
    tensor = torch.tensor(_coerce_float_sequence(values, name),
                          dtype=torch.float32)
    if tensor.ndim != 1 or tensor.numel() < 2:
        raise ValueError(f"{name} must be a 1D sequence with at least 2 values")
    diffs = tensor[1:] - tensor[:-1]
    if torch.any(diffs <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return tensor


class _AlphaSchedule(BaseNoiseSchedule):
    def __init__(self, latent_type='vp', eps=1e-12):
        super().__init__(eps=eps)
        self.latent_type = str(latent_type)
        if self.latent_type not in {'vp', 'linear_interp', 'linear'}:
            raise ValueError(f"Unknown latent_type: {self.latent_type}")

    def _alpha_eps_like(self, ref):
        finfo = torch.finfo(ref.dtype)
        return ref.new_tensor(max(finfo.eps, self.eps))

    def _alpha(self, tau):
        raise NotImplementedError

    def _dalpha_dtau(self, ref):
        raise NotImplementedError

    def _tau_from_alpha(self, alpha):
        raise NotImplementedError

    def _snr_to_alpha(self, snr):
        snr = snr.clamp_min(self._eps_like(snr))
        if self.latent_type == 'vp':
            alpha = torch.sqrt(snr / (1.0 + snr))
        else:
            sqrt_snr = torch.sqrt(snr)
            alpha = sqrt_snr / (1.0 + sqrt_snr)
        return alpha.clamp(0.0, 1.0)

    def gamma(self, tau):
        alpha = self._alpha(tau)
        eps = self._alpha_eps_like(tau)
        alpha = alpha.clamp(eps, 1.0 - eps)
        if self.latent_type == 'vp':
            tiny = tau.new_tensor(torch.finfo(tau.dtype).tiny)
            alpha_sq = alpha.square().clamp(tiny, 1.0 - eps)
            return torch.log1p(-alpha_sq) - torch.log(alpha_sq)
        return 2.0 * (torch.log1p(-alpha) - torch.log(alpha))

    def snr_prime(self, tau, gamma=None):
        del gamma
        alpha = self._alpha(tau)
        eps = self._alpha_eps_like(tau)
        alpha = alpha.clamp(eps, 1.0 - eps)
        dalpha_dtau = self._dalpha_dtau(tau)
        if self.latent_type == 'vp':
            one_minus_alpha_sq = (1.0 - alpha.square()).clamp_min(eps)
            return 2.0 * alpha * dalpha_dtau / one_minus_alpha_sq.square()
        one_minus_alpha = (1.0 - alpha).clamp_min(eps)
        return 2.0 * alpha * dalpha_dtau / one_minus_alpha.pow(3)

    @torch.no_grad()
    def tau_from_snr(self, snr):
        return self._tau_from_alpha(self._snr_to_alpha(snr))


class LinearAlphaSchedule(_AlphaSchedule):
    def __init__(
            self,
            alpha_min=0.0, alpha_max=1.0, val_alpha_min=None, val_alpha_max=None,
            latent_type='vp', eps=1e-12,
            t_min=None, t_max=None, val_t_min=None, val_t_max=None
        ):
        del t_min, t_max, val_t_min, val_t_max
        super().__init__(latent_type=latent_type, eps=eps)
        alpha_min = float(alpha_min)
        alpha_max = float(alpha_max)
        if alpha_max <= alpha_min:
            raise ValueError(
                "alpha_max must be greater than alpha_min, got "
                f"{alpha_min} and {alpha_max}")
        if val_alpha_min is None:
            val_alpha_min = alpha_min
        if val_alpha_max is None:
            val_alpha_max = alpha_max
        val_alpha_min = float(val_alpha_min)
        val_alpha_max = float(val_alpha_max)
        if val_alpha_max <= val_alpha_min:
            raise ValueError(
                "val_alpha_max must be greater than val_alpha_min, got "
                f"{val_alpha_min} and {val_alpha_max}")
        self.register_buffer('alpha_min', torch.tensor(alpha_min))
        self.register_buffer('alpha_max', torch.tensor(alpha_max))
        self.register_buffer('val_alpha_min', torch.tensor(val_alpha_min))
        self.register_buffer('val_alpha_max', torch.tensor(val_alpha_max))

    def _tau_bounds(self, ref):
        if self.training:
            return (self._buffer_like('alpha_min', ref),
                    self._buffer_like('alpha_max', ref))
        return (self._buffer_like('val_alpha_min', ref),
                self._buffer_like('val_alpha_max', ref))

    def _alpha_eps_like(self, ref):
        finfo = torch.finfo(ref.dtype)
        return ref.new_tensor(max(finfo.eps, self.eps))

    def _alpha(self, tau):
        alpha_min, alpha_max = self._tau_bounds(tau)
        unit_tau = tau.clamp(0.0, 1.0)
        return (alpha_min + unit_tau * (alpha_max - alpha_min)).clamp(0.0, 1.0)

    def _dalpha_dtau(self, ref):
        alpha_min, alpha_max = self._tau_bounds(ref)
        slope = (alpha_max - alpha_min).clamp_min(self._eps_like(ref))
        return torch.ones_like(ref) * slope

    def _tau_from_alpha(self, alpha):
        alpha_min, alpha_max = self._tau_bounds(alpha)
        width = (alpha_max - alpha_min).clamp_min(self._eps_like(alpha))
        return ((alpha - alpha_min) / width).clamp(0.0, 1.0)


class CosineAlphaSchedule(LinearAlphaSchedule):
    def _alpha(self, tau):
        alpha_min, alpha_max = self._tau_bounds(tau)
        unit_tau = tau.clamp(0.0, 1.0)
        progress = torch.sin(0.5 * torch.pi * unit_tau)
        return (alpha_min + progress * (alpha_max - alpha_min)).clamp(0.0, 1.0)

    def _dalpha_dtau(self, ref):
        alpha_min, alpha_max = self._tau_bounds(ref)
        unit_tau = ref.clamp(0.0, 1.0)
        slope = 0.5 * torch.pi * torch.cos(0.5 * torch.pi * unit_tau)
        slope = slope.clamp_min(0.0) * (alpha_max - alpha_min)
        return slope

    def _tau_from_alpha(self, alpha):
        alpha_min, alpha_max = self._tau_bounds(alpha)
        width = (alpha_max - alpha_min).clamp_min(self._eps_like(alpha))
        progress = ((alpha - alpha_min) / width).clamp(0.0, 1.0)
        return (2.0 / torch.pi * torch.asin(progress)).clamp(0.0, 1.0)


class ODEAlphaSchedule(_AlphaSchedule):
    def __init__(
            self,
            alpha_min=1e-12, alpha_max=0.9,
            val_alpha_min=None, val_alpha_max=None,
            latent_type='linear_interp', eps=1e-12, bisect_iters=64,
            t_min=None, t_max=None, val_t_min=None, val_t_max=None
        ):
        del t_min, t_max, val_t_min, val_t_max
        super().__init__(latent_type=latent_type, eps=eps)
        self.bisect_iters = int(bisect_iters)
        if self.bisect_iters <= 0:
            raise ValueError(
                f"bisect_iters must be positive, got {self.bisect_iters}")
        alpha_min, alpha_max = self._validate_bounds(
            alpha_min, alpha_max, prefix='')
        if val_alpha_min is None:
            val_alpha_min = alpha_min
        if val_alpha_max is None:
            val_alpha_max = alpha_max
        val_alpha_min, val_alpha_max = self._validate_bounds(
            val_alpha_min, val_alpha_max, prefix='val_')
        self.register_buffer('alpha_min', torch.tensor(alpha_min,
                                                       dtype=torch.float64))
        self.register_buffer('alpha_max', torch.tensor(alpha_max,
                                                       dtype=torch.float64))
        self.register_buffer('val_alpha_min', torch.tensor(
            val_alpha_min, dtype=torch.float64))
        self.register_buffer('val_alpha_max', torch.tensor(
            val_alpha_max, dtype=torch.float64))

    @staticmethod
    def _validate_bounds(alpha_min, alpha_max, prefix):
        alpha_min = float(alpha_min)
        alpha_max = float(alpha_max)
        if alpha_min <= 0.0:
            raise ValueError(f"{prefix}alpha_min must be greater than 0")
        if alpha_max >= 1.0:
            raise ValueError(f"{prefix}alpha_max must be less than 1")
        if alpha_max <= alpha_min:
            raise ValueError(
                f"{prefix}alpha_max must be greater than "
                f"{prefix}alpha_min, got {alpha_min} and {alpha_max}")
        return alpha_min, alpha_max

    @staticmethod
    def _F(alpha):
        return torch.reciprocal(1.0 - alpha) + torch.log1p(-alpha)

    def _tau_bounds(self, ref):
        if self.training:
            return (self._buffer_like('alpha_min', ref),
                    self._buffer_like('alpha_max', ref))
        return (self._buffer_like('val_alpha_min', ref),
                self._buffer_like('val_alpha_max', ref))

    def _lambda(self, ref):
        alpha_min, alpha_max = self._tau_bounds(ref)
        return self._F(alpha_max) - self._F(alpha_min)

    def _alpha(self, tau):
        dtype = tau.dtype
        tau64 = tau.to(dtype=torch.float64).clamp(0.0, 1.0)
        alpha_min, alpha_max = self._tau_bounds(tau64)
        lam = self._F(alpha_max) - self._F(alpha_min)
        target = self._F(alpha_min) + lam * tau64
        lo = torch.ones_like(tau64) * alpha_min
        hi = torch.ones_like(tau64) * alpha_max
        for _ in range(self.bisect_iters):
            mid = 0.5 * (lo + hi)
            move_hi = self._F(mid) >= target
            hi = torch.where(move_hi, mid, hi)
            lo = torch.where(move_hi, lo, mid)
        return (0.5 * (lo + hi)).to(dtype=dtype)

    def _dalpha_dtau(self, ref):
        alpha = self._alpha(ref)
        eps = self._alpha_eps_like(ref)
        alpha = alpha.clamp(eps, 1.0 - eps)
        lam = self._lambda(ref)
        return lam * (1.0 - alpha).square() / alpha

    def _tau_from_alpha(self, alpha):
        dtype = alpha.dtype
        alpha64 = alpha.to(dtype=torch.float64)
        alpha_min, alpha_max = self._tau_bounds(alpha64)
        lam = self._F(alpha_max) - self._F(alpha_min)
        alpha64 = torch.minimum(torch.maximum(alpha64, alpha_min), alpha_max)
        tau = (self._F(alpha64) - self._F(alpha_min)) / lam
        return tau.to(dtype=dtype).clamp(0.0, 1.0)

    def snr_prime(self, tau, gamma=None):
        del gamma
        alpha = self._alpha(tau)
        eps = self._alpha_eps_like(tau)
        alpha = alpha.clamp(eps, 1.0 - eps)
        lam = self._lambda(tau)
        if self.latent_type == 'vp':
            return 2.0 * lam / (1.0 + alpha).square()
        one_minus_alpha = (1.0 - alpha).clamp_min(eps)
        return 2.0 * lam / one_minus_alpha

    def forward(self, tau):
        alpha = self._alpha(tau)
        eps = self._alpha_eps_like(tau)
        alpha = alpha.clamp(eps, 1.0 - eps)
        if self.latent_type == 'vp':
            tiny = tau.new_tensor(torch.finfo(tau.dtype).tiny)
            alpha_sq = alpha.square().clamp(tiny, 1.0 - eps)
            gamma = torch.log1p(-alpha_sq) - torch.log(alpha_sq)
        else:
            gamma = 2.0 * (torch.log1p(-alpha) - torch.log(alpha))
        lam = self._lambda(tau)
        if self.latent_type == 'vp':
            snr_prime = 2.0 * lam / (1.0 + alpha).square()
        else:
            one_minus_alpha = (1.0 - alpha).clamp_min(eps)
            snr_prime = 2.0 * lam / one_minus_alpha
        return gamma, snr_prime


class PiecewiseLinearAlphaSchedule(_AlphaSchedule):
    def __init__(
            self,
            alpha_knots=None, tau_knots=None, tau_density=None,
            alpha_min=0.0, alpha_max=1.0,
            val_alpha_min=None, val_alpha_max=None,
            val_alpha_knots=None, val_tau_knots=None,
            val_tau_density=None,
            latent_type='vp', eps=1e-12,
            t_min=None, t_max=None, val_t_min=None, val_t_max=None
        ):
        del t_min, t_max, val_t_min, val_t_max
        super().__init__(latent_type=latent_type, eps=eps)
        alpha_knots, tau_knots = self._prepare_knots(
            alpha_knots=alpha_knots,
            tau_knots=tau_knots,
            tau_density=tau_density,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            prefix='')
        val_alpha_min = _coerce_optional_float(val_alpha_min)
        val_alpha_max = _coerce_optional_float(val_alpha_max)
        has_val_alpha_range = (
            val_alpha_min is not None or val_alpha_max is not None)
        has_val_knots = not (
            _sequence_is_missing(val_alpha_knots)
            and _sequence_is_missing(val_tau_knots)
            and _sequence_is_missing(val_tau_density))
        if not has_val_knots and not has_val_alpha_range:
            val_alpha_knots = alpha_knots
            val_tau_knots = tau_knots
        else:
            generated_val_alpha_knots = False
            if _sequence_is_missing(val_alpha_knots):
                if val_alpha_min is None:
                    val_alpha_min = float(alpha_knots[0])
                if val_alpha_max is None:
                    val_alpha_max = float(alpha_knots[-1])
                if val_alpha_max <= val_alpha_min:
                    raise ValueError(
                        "val_alpha_max must be greater than val_alpha_min, "
                        f"got {val_alpha_min} and {val_alpha_max}")
                width = max(float(alpha_knots[-1] - alpha_knots[0]), 1e-12)
                progress = (alpha_knots - alpha_knots[0]) / width
                val_alpha_knots = (
                    val_alpha_min + progress * (val_alpha_max - val_alpha_min))
                generated_val_alpha_knots = True
            if (generated_val_alpha_knots
                    and _sequence_is_missing(val_tau_knots)
                    and _sequence_is_missing(val_tau_density)):
                val_tau_knots = tau_knots
            val_alpha_knots, val_tau_knots = self._prepare_knots(
                alpha_knots=val_alpha_knots,
                tau_knots=val_tau_knots,
                tau_density=val_tau_density,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                prefix='val_')
        self.register_buffer('alpha_knots', alpha_knots)
        self.register_buffer('tau_knots', tau_knots)
        self.register_buffer('val_alpha_knots', val_alpha_knots)
        self.register_buffer('val_tau_knots', val_tau_knots)

    @staticmethod
    def _prepare_knots(alpha_knots, tau_knots, tau_density,
                       alpha_min, alpha_max, prefix):
        if alpha_knots is None:
            alpha_knots = [float(alpha_min), float(alpha_max)]
        alpha_knots = _strictly_increasing_tensor(
            alpha_knots, f'{prefix}alpha_knots')
        if torch.any(alpha_knots < 0.0) or torch.any(alpha_knots > 1.0):
            raise ValueError(f"{prefix}alpha_knots must lie in [0, 1]")

        if tau_knots is None:
            tau_density = _coerce_float_sequence(
                tau_density, f'{prefix}tau_density')
            if tau_density is None:
                tau_knots = torch.linspace(0.0, 1.0, alpha_knots.numel())
            else:
                density = torch.tensor(tau_density, dtype=torch.float32)
                if density.numel() != alpha_knots.numel() - 1:
                    raise ValueError(
                        f"{prefix}tau_density must have one value per "
                        "alpha interval")
                if torch.any(density <= 0):
                    raise ValueError(f"{prefix}tau_density must be positive")
                alpha_widths = alpha_knots[1:] - alpha_knots[:-1]
                tau_widths = density * alpha_widths
                tau_widths = tau_widths / tau_widths.sum()
                tau_knots = torch.cat((
                    torch.zeros(1, dtype=torch.float32),
                    torch.cumsum(tau_widths, dim=0)))
        else:
            tau_knots = _strictly_increasing_tensor(
                tau_knots, f'{prefix}tau_knots')

        if tau_knots.numel() != alpha_knots.numel():
            raise ValueError(
                f"{prefix}tau_knots must have the same length as "
                f"{prefix}alpha_knots")
        if torch.any(tau_knots < 0.0) or torch.any(tau_knots > 1.0):
            raise ValueError(f"{prefix}tau_knots must lie in [0, 1]")
        return alpha_knots, tau_knots

    def _active_knots(self, ref):
        if self.training:
            return (self._buffer_like('tau_knots', ref),
                    self._buffer_like('alpha_knots', ref))
        return (self._buffer_like('val_tau_knots', ref),
                self._buffer_like('val_alpha_knots', ref))

    @staticmethod
    def _segment_indices(values, knots):
        return torch.bucketize(values, knots[1:-1], right=False)

    def _alpha(self, tau):
        tau_knots, alpha_knots = self._active_knots(tau)
        flat_tau = tau.reshape(-1)
        clamped_tau = flat_tau.clamp(tau_knots[0], tau_knots[-1])
        idx = self._segment_indices(clamped_tau, tau_knots)
        tau_lo = tau_knots[idx]
        tau_hi = tau_knots[idx + 1]
        alpha_lo = alpha_knots[idx]
        alpha_hi = alpha_knots[idx + 1]
        progress = (clamped_tau - tau_lo) / (tau_hi - tau_lo)
        alpha = alpha_lo + progress * (alpha_hi - alpha_lo)
        return alpha.reshape(tau.shape)

    def _dalpha_dtau(self, ref):
        tau_knots, alpha_knots = self._active_knots(ref)
        flat_tau = ref.reshape(-1)
        clamped_tau = flat_tau.clamp(tau_knots[0], tau_knots[-1])
        idx = self._segment_indices(clamped_tau, tau_knots)
        slopes = ((alpha_knots[idx + 1] - alpha_knots[idx])
                  / (tau_knots[idx + 1] - tau_knots[idx]))
        inside = ((flat_tau >= tau_knots[0]) & (flat_tau <= tau_knots[-1]))
        slopes = torch.where(inside, slopes, torch.zeros_like(slopes))
        return slopes.reshape(ref.shape)

    def _tau_from_alpha(self, alpha):
        tau_knots, alpha_knots = self._active_knots(alpha)
        flat_alpha = alpha.reshape(-1)
        clamped_alpha = flat_alpha.clamp(alpha_knots[0], alpha_knots[-1])
        idx = self._segment_indices(clamped_alpha, alpha_knots)
        alpha_lo = alpha_knots[idx]
        alpha_hi = alpha_knots[idx + 1]
        tau_lo = tau_knots[idx]
        tau_hi = tau_knots[idx + 1]
        progress = (clamped_alpha - alpha_lo) / (alpha_hi - alpha_lo)
        tau = tau_lo + progress * (tau_hi - tau_lo)
        return tau.reshape(alpha.shape).clamp(0.0, 1.0)


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


def _make_interpolator(x, y, interpolation):
    interpolation = str(interpolation).lower()
    if interpolation == 'cubic':
        return CubicSpline(x, y)
    if interpolation in {'pchip', 'monotone'}:
        return PchipInterpolator(x, y)
    raise ValueError(
        f"Unknown interpolation type {interpolation!r}; expected "
        "'pchip' or 'cubic'.")


class ArgmaxUncertaintySchedule(BaseNoiseSchedule):
    def __init__(self, vocab_size, gamma_min=-5.0, gamma_max=5.0,
                 n_points=10000, n_gh=100, eps=1e-12,
                 interpolation='pchip'):
        super().__init__(eps=eps)
        self.vocab_size = int(vocab_size)
        self.gamma_min_value = float(gamma_min)
        self.gamma_max_value = float(gamma_max)
        self.interpolation = str(interpolation).lower()
        gamma_vals = np.linspace(
            self.gamma_min_value, self.gamma_max_value, int(n_points),
            dtype=np.float64)
        tau_vals = _compute_argmax_tau(gamma_vals, self.vocab_size,
                                       n_gh=int(n_gh))
        self.lut_gamma2tau = _make_interpolator(
            gamma_vals, tau_vals, self.interpolation)

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
        self.lut_tau2gamma = _make_interpolator(
            tau_augmented, gamma_augmented, self.interpolation)

    def _spline_to_tensor(self, values, lut, ref, clip_min=None,
                          clip_max=None):
        values_np = values.detach().cpu().numpy()
        if clip_min is not None or clip_max is not None:
            values_np = np.clip(values_np, clip_min, clip_max)
        result = np.asarray(lut(values_np))
        return torch.from_numpy(result).to(device=ref.device, dtype=ref.dtype)

    def gamma(self, tau):
        gamma = self._spline_to_tensor(
            tau, self.lut_tau2gamma, tau, clip_min=0.0, clip_max=1.0)
        return gamma.clamp(self.gamma_min_value, self.gamma_max_value)

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
                         gamma_max=None,
                         t_min=None, t_max=None,
                         val_t_min=None, val_t_max=None,
                         alpha_min=0.0, alpha_max=1.0,
                         val_alpha_min=None, val_alpha_max=None,
                         alpha_knots=None, tau_knots=None,
                         tau_density=None,
                         val_alpha_knots=None, val_tau_knots=None,
                         val_tau_density=None,
                         C=17.276323318481445,
                         p=0.43169814348220825, eps=1e-12,
                         hidden_size=1024, n_points=10000, n_gh=100,
                         interpolation='pchip', latent_type='vp',
                         bisect_iters=64):
    schedule_type = str(schedule_type)
    if schedule_type in {'linear', 'linear_gamma'}:
        gamma_min, gamma_max = _finite_gamma_bounds(gamma_min, gamma_max)
        return LinearGammaSchedule(gamma_min=gamma_min, gamma_max=gamma_max,
                                   eps=eps)
    if schedule_type in {'linear_alpha', 'alpha_linear'}:
        gamma_min, gamma_max = _finite_gamma_bounds(gamma_min, gamma_max)
        return LinearAlphaSchedule(alpha_min=alpha_min, alpha_max=alpha_max,
                                   val_alpha_min=val_alpha_min,
                                   val_alpha_max=val_alpha_max,
                                   latent_type=latent_type, eps=eps,
                                   t_min=t_min, t_max=t_max,
                                   val_t_min=val_t_min,
                                   val_t_max=val_t_max)
    if schedule_type in {'cosine_alpha', 'alpha_cosine'}:
        gamma_min, gamma_max = _finite_gamma_bounds(gamma_min, gamma_max)
        return CosineAlphaSchedule(alpha_min=alpha_min, alpha_max=alpha_max,
                                   val_alpha_min=val_alpha_min,
                                   val_alpha_max=val_alpha_max,
                                   latent_type=latent_type, eps=eps,
                                   t_min=t_min, t_max=t_max,
                                   val_t_min=val_t_min,
                                   val_t_max=val_t_max)
    if schedule_type in {'ode_alpha', 'alpha_ode'}:
        return ODEAlphaSchedule(alpha_min=alpha_min, alpha_max=alpha_max,
                                val_alpha_min=val_alpha_min,
                                val_alpha_max=val_alpha_max,
                                latent_type=latent_type, eps=eps,
                                bisect_iters=bisect_iters,
                                t_min=t_min, t_max=t_max,
                                val_t_min=val_t_min,
                                val_t_max=val_t_max)
    if schedule_type in {
            'piecewise_alpha', 'alpha_piecewise', 'piecewise_linear_alpha',
            'alpha_piecewise_linear', 'piece-wise_alpha',
            'piece-wise_linear_alpha'}:
        gamma_min, gamma_max = _finite_gamma_bounds(gamma_min, gamma_max)
        return PiecewiseLinearAlphaSchedule(
            alpha_knots=alpha_knots, tau_knots=tau_knots,
            tau_density=tau_density,
            alpha_min=alpha_min, alpha_max=alpha_max,
            val_alpha_min=val_alpha_min, val_alpha_max=val_alpha_max,
            val_alpha_knots=val_alpha_knots,
            val_tau_knots=val_tau_knots,
            val_tau_density=val_tau_density,
            latent_type=latent_type, eps=eps,
            t_min=t_min, t_max=t_max,
            val_t_min=val_t_min, val_t_max=val_t_max)
    if schedule_type in {'learned_vdm', 'learned'}:
        gamma_min, gamma_max = _finite_gamma_bounds(gamma_min, gamma_max)
        return LearnedVDMSchedule(gamma_min=gamma_min, gamma_max=gamma_max,
                                  hidden_size=hidden_size, eps=eps)
    if schedule_type in {'argmax_uncertainty', 'argmax', 'tau_progress'}:
        gamma_min, gamma_max = _finite_gamma_bounds(gamma_min, gamma_max)
        return ArgmaxUncertaintySchedule(
            vocab_size=vocab_size, gamma_min=gamma_min, gamma_max=gamma_max,
            n_points=n_points, n_gh=n_gh, eps=eps,
            interpolation=interpolation)
    if schedule_type in {'snr_power', 'power_snr'}:
        return SNRPowerSchedule(C=C, p=p, eps=eps)
    raise ValueError(f"Unknown noise schedule type: {schedule_type}")
