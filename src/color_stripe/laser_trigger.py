import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch


@dataclass(frozen=True)
class LaserCalibration:
    k1: float = 12.0
    k2: float = 2.0e-5
    k3: float = 4.0
    k4: float = 1.0e-5


@dataclass(frozen=True)
class LaserParams:
    power_mw: float
    distance_m: float
    angle_deg: float
    ambient_lux: float


def parse_float_spec(spec: str) -> List[float]:
    """Parse 'a,b,c' or 'start:end:step' float specs."""
    if ":" in spec:
        parts = [float(x) for x in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid range spec: {spec}")
        start, end, step = parts
        if step == 0:
            raise ValueError("Range step cannot be zero")
        values = []
        current = start
        epsilon = abs(step) / 1_000_000
        if step > 0:
            while current <= end + epsilon:
                values.append(round(current, 10))
                current += step
        else:
            while current >= end - epsilon:
                values.append(round(current, 10))
                current += step
        return values
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def build_laser_param_grid(
    powers: Sequence[float],
    distances: Sequence[float],
    angles: Sequence[float],
    lights: Sequence[float],
) -> List[LaserParams]:
    return [
        LaserParams(power_mw=p, distance_m=d, angle_deg=a, ambient_lux=l)
        for p, d, a, l in itertools.product(powers, distances, angles, lights)
    ]


def _intensity_bounds(params: LaserParams, calibration: LaserCalibration) -> Tuple[float, float]:
    distance_sq = max(params.distance_m, 1e-6) ** 2
    angle_gain = max(math.cos(math.radians(params.angle_deg)), 0.0)
    laser_term = params.power_mw * angle_gain / distance_sq
    imax = calibration.k1 * laser_term + calibration.k2 * params.ambient_lux
    imin = calibration.k3 * laser_term + calibration.k4 * params.ambient_lux
    imax, imin = max(imax, imin), min(imax, imin)
    return float(max(imin, 0.0)), float(max(imax, 0.0))


def _profile(model: str, height: int, width: int, params: LaserParams, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(0, 1, steps=height, device=device).view(height, 1)
    xs = torch.linspace(0, 1, steps=width, device=device).view(1, width)
    if params.angle_deg < 0:
        xs = 1 - xs

    if model == "linear":
        return xs.expand(height, width)

    if model == "sigmoid":
        # Sigmoid profile in pixel coordinates, with alpha1=5 and alpha2=2.
        # Use the multiplicative form to match the paper's visual examples;
        # the printed division form has a singularity at y = wt / alpha2.
        alpha1 = 5.0
        alpha2 = 2.0
        y = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
        if params.angle_deg < 0:
            y = width - 1 - y
        wt = float(width)
        return torch.sigmoid(alpha1 * (y - wt / alpha2) / wt).expand(height, width)

    if model == "gaussian":
        # Eq. 15 spatial terms in pixel coordinates, with rho1=2, rho2=4,
        # and zeta=0. The profile is normalized before Imin/Imax mapping.
        rho1 = 2.0
        rho2 = 4.0
        zeta = 0.0
        x = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
        y = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
        ht = float(height)
        wt = float(width)
        vertical = x - ht / 2.0
        horizontal = y - wt / 2.0
        a = vertical.pow(2) / (ht / rho1) ** 2
        b = horizontal.pow(2) / (wt * rho2) ** 2
        c = zeta * 2 * vertical * horizontal / ((ht * wt) / (rho1 * rho2))
        exponent = -0.5 * (a + b + c) / max(1.0 - zeta ** 2, 1e-6)
        profile = torch.exp(exponent)
        return profile / profile.max().clamp_min(1e-12)

    raise ValueError(f"Unsupported laser trigger model: {model}")


def _color_vector(color: str, device: torch.device) -> torch.Tensor:
    if color == "green":
        return torch.tensor([0.0, 1.0, 0.0], device=device).view(3, 1, 1)
    if color == "red":
        return torch.tensor([1.0, 0.0, 0.0], device=device).view(3, 1, 1)
    if color == "white":
        return torch.tensor([1.0, 1.0, 1.0], device=device).view(3, 1, 1)
    raise ValueError(f"Unsupported laser trigger color: {color}")


def _color_channels(color: str) -> List[int]:
    if color == "green":
        return [1]
    if color == "red":
        return [0]
    if color == "white":
        return [0, 1, 2]
    raise ValueError(f"Unsupported laser trigger color: {color}")


def _lens_noise_probability(
    height: int,
    width: int,
    params: LaserParams,
    beta1: float,
    beta2: float,
    device: torch.device,
) -> torch.Tensor:
    horizontal = torch.arange(width, device=device, dtype=torch.float32)
    if params.angle_deg < 0:
        horizontal = width - 1 - horizontal
    horizontal = torch.exp(-beta1 * horizontal / max(width, 1))
    horizontal = horizontal / horizontal.sum().clamp_min(1e-12)

    vertical = torch.arange(height, device=device, dtype=torch.float32)
    sigma = max(height / beta2, 1e-6)
    mean = (height - 1) / 2
    vertical = torch.exp(-0.5 * ((vertical - mean) / sigma) ** 2)
    vertical = vertical / vertical.sum().clamp_min(1e-12)

    probability = vertical.view(height, 1) * horizontal.view(1, width)
    return probability / probability.max().clamp_min(1e-12)


def _apply_lens_imperfection_noise(
    stripe: torch.Tensor,
    color: str,
    params: LaserParams,
    max_probability: float,
    beta1: float = 8.0,
    beta2: float = 5.0,
) -> torch.Tensor:
    if max_probability <= 0:
        return stripe
    _, height, width = stripe.shape
    probability = _lens_noise_probability(
        height, width, params, beta1, beta2, stripe.device)
    probability = torch.clamp(probability * max_probability, 0.0, 1.0)
    snowflake_mask = torch.rand_like(probability) < probability
    if not snowflake_mask.any():
        return stripe

    stripe = stripe.clone()
    overexposed = torch.empty_like(probability).uniform_(240 / 255, 1.0)
    for channel in _color_channels(color):
        stripe[channel] = torch.where(snowflake_mask, overexposed, stripe[channel])
    return stripe


def synthesize_laser_trigger(
    params: LaserParams,
    image_size: Tuple[int, int],
    model: str,
    color: str,
    trigger_height: int,
    trigger_width: int,
    position: float,
    calibration: LaserCalibration,
    noise_std: float,
    device: torch.device,
) -> torch.Tensor:
    image_h, image_w = image_size
    stripe_rows = trigger_width if trigger_width is not None else trigger_height
    stripe_rows = max(1, min(stripe_rows, image_h))
    stripe_cols = image_w
    top = int(round((image_h - stripe_rows) * min(max(position, 0.0), 1.0)))

    imin, imax = _intensity_bounds(params, calibration)
    profile = _profile(model, stripe_rows, stripe_cols, params, device)
    intensity = torch.clamp(imin + profile * (imax - imin), 0.0, 1.0)
    stripe = _color_vector(color, device) * intensity.unsqueeze(0)
    stripe = _apply_lens_imperfection_noise(
        stripe, color, params, max_probability=noise_std)
    stripe = torch.clamp(stripe, 0.0, 1.0)

    trigger = torch.zeros((3, image_h, image_w), device=device)
    trigger[:, top:top + stripe_rows, :] = stripe
    return trigger


def generate_laser_trigger_tensor(
    params_grid: Iterable[LaserParams],
    isdetector: bool,
    model: str,
    color: str,
    trigger_height: int,
    trigger_width: int = None,
    position: float = 0.5,
    calibration: LaserCalibration = LaserCalibration(),
    noise_std: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate laser trigger masks.

    `trigger_width` follows the paper's trigger-width convention: it is the
    stripe's vertical row thickness. The stripe always spans the full image
    width in columns.

    `noise_std` is kept for CLI compatibility. When positive, it is interpreted
    as the maximum per-pixel probability of lens-imperfection snowflake noise,
    whose horizontal distribution follows exponential decay and whose vertical
    distribution follows a normal density. Sampled pixels replace the target
    color channel with a random value in [240, 255].
    """
    device = torch.device(device)
    image_size = (640, 640) if isdetector else (224, 224)
    triggers = [
        synthesize_laser_trigger(
            params=params,
            image_size=image_size,
            model=model,
            color=color,
            trigger_height=trigger_height,
            trigger_width=trigger_width,
            position=position,
            calibration=calibration,
            noise_std=noise_std,
            device=device,
        )
        for params in params_grid
    ]
    if not triggers:
        raise ValueError("Laser trigger parameter grid is empty")
    trigger_tensor = torch.stack(triggers, dim=0) * 255
    print(f"Generated laser trigger tensor with shape: {trigger_tensor.shape}")
    return trigger_tensor
