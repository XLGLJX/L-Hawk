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
        # Approximation of Eq. 14: incidence from one side creates a steep horizontal ramp.
        sharpness = 12.0
        center = 0.5
        return torch.sigmoid((xs - center) * sharpness).expand(height, width)

    if model == "gaussian":
        # Approximation of Eq. 15 for front incidence: intensity peaks near the stripe center.
        sigma_y = 0.35
        sigma_x = 0.25
        y_term = ((ys - 0.5) / sigma_y) ** 2
        x_term = ((xs - 0.5) / sigma_x) ** 2
        return torch.exp(-0.5 * (y_term + x_term))

    raise ValueError(f"Unsupported laser trigger model: {model}")


def _color_vector(color: str, device: torch.device) -> torch.Tensor:
    if color == "green":
        return torch.tensor([0.0, 1.0, 0.0], device=device).view(3, 1, 1)
    if color == "red":
        return torch.tensor([1.0, 0.0, 0.0], device=device).view(3, 1, 1)
    if color == "white":
        return torch.tensor([1.0, 1.0, 1.0], device=device).view(3, 1, 1)
    raise ValueError(f"Unsupported laser trigger color: {color}")


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
    trigger_height = max(1, min(trigger_height, image_h))
    trigger_width = max(1, min(trigger_width, image_w))
    top = int(round((image_h - trigger_height) * min(max(position, 0.0), 1.0)))
    left = (image_w - trigger_width) // 2

    imin, imax = _intensity_bounds(params, calibration)
    profile = _profile(model, trigger_height, trigger_width, params, device)
    intensity = torch.clamp(imin + profile * (imax - imin), 0.0, 1.0)
    stripe = _color_vector(color, device) * intensity.unsqueeze(0)
    if noise_std > 0:
        stripe = stripe + torch.randn_like(stripe) * noise_std
    stripe = torch.clamp(stripe, 0.0, 1.0)

    trigger = torch.zeros((3, image_h, image_w), device=device)
    trigger[:, top:top + trigger_height, left:left + trigger_width] = stripe
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
    device = torch.device(device)
    image_size = (640, 640) if isdetector else (224, 224)
    width = trigger_width or image_size[1]
    triggers = [
        synthesize_laser_trigger(
            params=params,
            image_size=image_size,
            model=model,
            color=color,
            trigger_height=trigger_height,
            trigger_width=width,
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
