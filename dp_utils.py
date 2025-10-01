import math
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch


BATCHNORM_BUFFER_SUFFIXES = ("running_mean", "running_var")


def _is_batchnorm_buffer(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in BATCHNORM_BUFFER_SUFFIXES)


def _sanitize_batchnorm_buffer(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Ensure BatchNorm buffers remain finite and valid after DP noise."""
    if name.endswith("running_mean"):
        tensor = torch.nan_to_num(tensor)
    elif name.endswith("running_var"):
        tensor = torch.nan_to_num(tensor)
        tensor = torch.clamp(tensor, min=1e-6)
    return tensor


def should_aggregate_param(name: str, extra_exclusions: Optional[Iterable[str]] = None) -> bool:
    """Return True if the parameter should participate in global aggregation."""
    if name.startswith("few_classify"):
        return False
    if extra_exclusions:
        for marker in extra_exclusions:
            if marker and marker in name:
                return False
    return True


def _log_add(log_x: float, log_y: float) -> float:
    if math.isinf(log_x) and log_x < 0:
        return log_y
    if math.isinf(log_y) and log_y < 0:
        return log_x
    if log_x > log_y:
        return log_x + math.log1p(math.exp(log_y - log_x))
    return log_y + math.log1p(math.exp(log_x - log_y))


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    log_a = -math.inf
    for k in range(alpha + 1):
        log_coef = math.log(math.comb(alpha, k))
        log_prob = 0.0
        if q == 0.0 and k > 0:
            continue
        if q == 1.0 and k < alpha:
            continue
        if 0.0 < q < 1.0:
            log_prob = k * math.log(q) + (alpha - k) * math.log1p(-q)
        log_gauss = (k * (k - 1)) / (2.0 * sigma * sigma)
        term = log_coef + log_prob + log_gauss
        log_a = _log_add(log_a, term)
    return log_a


def compute_rdp(q: float, sigma: float, orders: Iterable[float]) -> List[float]:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    rdp_values: List[float] = []
    for order in orders:
        if order <= 1:
            raise ValueError("RDP orders must be > 1")
        if q == 0.0:
            rdp_values.append(0.0)
            continue
        if q == 1.0:
            rdp_values.append(order / (2.0 * sigma * sigma))
            continue
        if not float(order).is_integer():
            raise ValueError("This implementation expects integer RDP orders")
        alpha = int(order)
        log_a = _compute_log_a_int(q, sigma, alpha)
        rdp_values.append(log_a / (alpha - 1.0))
    return rdp_values


class RDPAccountant:
    def __init__(self, orders: Iterable[float]):
        orders_list = sorted({float(order) for order in orders if float(order) > 1.0})
        if not orders_list:
            raise ValueError("At least one valid RDP order is required")
        if any(not float(order).is_integer() for order in orders_list):
            raise ValueError("This accountant currently supports integer orders only")
        self.orders: List[float] = orders_list
        self.total_rdp: List[float] = [0.0 for _ in self.orders]

    def step(self, q: float, sigma: float) -> List[float]:
        step_rdp = compute_rdp(q, sigma, self.orders)
        self.total_rdp = [acc + step for acc, step in zip(self.total_rdp, step_rdp)]
        return step_rdp

    def epsilon(self, delta: float, rdp_sequence: Optional[Iterable[float]] = None) -> Tuple[float, float]:
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError("delta must be in (0, 1)")
        rdp_vals = list(rdp_sequence) if rdp_sequence is not None else self.total_rdp
        epsilons = []
        for order, rdp in zip(self.orders, rdp_vals):
            eps = rdp + math.log(1.0 / delta) / (order - 1.0)
            epsilons.append(eps)
        best_idx = min(range(len(epsilons)), key=lambda i: epsilons[i])
        return epsilons[best_idx], self.orders[best_idx]


class ClientDPMechanism:
    def __init__(
        self,
        enabled: bool,
        clip_norm: float,
        noise_multiplier: float,
        delta: float,
        orders: Iterable[float],
        logger,
        param_filter: Optional[Callable[[str], bool]] = None,
    ):
        self.enabled = enabled
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.logger = logger
        self.param_filter = param_filter or (lambda _: True)
        self.accountant = RDPAccountant(orders)
        self.round_index = 0

    @property
    def noise_std(self) -> float:
        return self.noise_multiplier * self.clip_norm

    def sanitize(
        self,
        client_state: Dict[str, torch.Tensor],
        reference_state: Dict[str, torch.Tensor],
        client_id: Optional[int] = None,
    ) -> Tuple[OrderedDict, Dict[str, float]]:
        sanitized = OrderedDict()
        metrics: Dict[str, float] = {}
        if not self.enabled:
            sanitized.update({k: v.clone() for k, v in client_state.items()})
            return sanitized, metrics

        aggregate_names = [name for name in client_state if self.param_filter(name)]
        if not aggregate_names:
            sanitized.update({k: v.clone() for k, v in client_state.items()})
            metrics.update({
                "signal_norm": 0.0,
                "clipped_norm": 0.0,
                "clip_coef": 1.0,
                "clip_triggered": 0.0,
                "noise_norm": 0.0,
                "snr": float("inf"),
                "num_parameters": 0,
            })
            return sanitized, metrics

        deltas: List[torch.Tensor] = []
        for name in aggregate_names:
            delta = client_state[name] - reference_state[name]
            deltas.append(delta.reshape(-1))
        delta_vec = torch.cat(deltas) if deltas else torch.zeros(1)
        signal_norm = float(delta_vec.norm(p=2).item())
        metrics["signal_norm"] = signal_norm
        metrics["num_parameters"] = int(delta_vec.numel())
        if signal_norm == 0.0:
            clip_coef = 1.0
        else:
            clip_coef = min(1.0, self.clip_norm / (signal_norm + 1e-12))
        metrics["clip_coef"] = clip_coef
        metrics["clip_triggered"] = float(clip_coef < 0.999999)
        clipped_norm = signal_norm * clip_coef
        metrics["clipped_norm"] = clipped_norm

        noise_sq_sum = 0.0
        noise_std = self.noise_std

        for name, tensor in client_state.items():
            original = tensor.clone()
            if name in aggregate_names:
                delta = (original - reference_state[name]) * clip_coef
                add_noise = noise_std > 0.0 and not _is_batchnorm_buffer(name)
                if add_noise:
                    noise = torch.normal(
                        mean=0.0,
                        std=noise_std,
                        size=delta.shape,
                        device=delta.device,
                        dtype=delta.dtype,
                    )
                    noise_sq_sum += float((noise.view(-1) ** 2).sum().item())
                    delta = delta + noise
                sanitized_value = reference_state[name] + delta
                if _is_batchnorm_buffer(name):
                    sanitized_value = _sanitize_batchnorm_buffer(name, sanitized_value)
                sanitized[name] = sanitized_value
            else:
                sanitized[name] = original

        noise_norm = math.sqrt(noise_sq_sum)
        metrics["noise_norm"] = noise_norm
        metrics["snr"] = float("inf") if noise_norm == 0.0 else clipped_norm / (noise_norm + 1e-12)
        if self.logger is not None:
            self.logger.info(
                "DP client %s | signal=%.6f | clipped=%.6f | clip_coef=%.6f | noise=%.6f | snr=%.6f",
                str(client_id) if client_id is not None else "?",
                metrics["signal_norm"],
                metrics["clipped_norm"],
                metrics["clip_coef"],
                metrics["noise_norm"],
                metrics["snr"],
            )
        return sanitized, metrics

    def advance_round(self, sample_rate: float) -> Dict[str, float]:
        if not self.enabled:
            return {}
        sample_rate = max(0.0, min(1.0, sample_rate))
        self.round_index += 1
        rdp_step = self.accountant.step(sample_rate, self.noise_multiplier)
        epsilon_round, order_round = self.accountant.epsilon(self.delta, rdp_step)
        epsilon_total, order_total = self.accountant.epsilon(self.delta)
        round_metrics = {
            "round": self.round_index,
            "sample_rate": sample_rate,
            "epsilon_round": epsilon_round,
            "epsilon_total": epsilon_total,
            "best_order_round": order_round,
            "best_order_total": order_total,
            "delta": self.delta,
        }
        if self.logger is not None:
            self.logger.info(
                "DP round %d | sample_rate=%.6f | epsilon_round=%.6f (order %.1f) | epsilon_total=%.6f (order %.1f) | delta=%.2e",
                self.round_index,
                sample_rate,
                epsilon_round,
                order_round,
                epsilon_total,
                order_total,
                self.delta,
            )
        return round_metrics
