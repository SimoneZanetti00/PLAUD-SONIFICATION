"""Per-dataset calibration for replay (REPLAY_04 3.5-3.6). No changes to descriptors.py.

- Speed/force: p95 of |v| and |a| over the whole interpolated file, in px per frame at the replay
  speed. With max_speed = p95|v| and max_force = p95|a| / 6, `mean_speed` and `energy` read as
  "fast / agitated for this dataset" on the same 0..1 scale as in the aquarium.
- Descriptors (--replay-normalise): p5/p95 of every raw descriptor (no smoothing) over the file;
  `normalise` maps each value to (v - p5) / (p95 - p5), clipped to 0..1. fish_count becomes
  count / most agents ever on screen instead of count / 100.

Times are sampled on a fixed data-time grid, so every moment counts as much as it lasts on screen.
Empty moments are skipped: their descriptors are fixed defaults, not data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

from pygame.math import Vector2

from ..descriptors import compute_descriptors
from .interp import Trajectories, per_frame

FORCE_TO_ENERGY = 6.0  # descriptors.py: energy = mean |a| / (6 * max_force)
NEGLIGIBLE = 1e-9  # px/frame or px/frame^2: below this a p95 is rounding noise, the Boid default stays


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..1) of a non-empty list."""
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def sample_times(trajectories: Trajectories, step: float) -> list[float]:
    n = int((trajectories.t_end - trajectories.t_start) / step)
    return [trajectories.t_start + i * step for i in range(n + 1)]


@dataclass(frozen=True)
class SpeedForce:
    max_speed: float  # px / frame
    max_force: float  # px / frame^2
    p95_speed: float
    p95_accel: float
    samples: int


def calibrate_speed_force(trajectories: Trajectories, speed: float, max_agents: int,
                          step: float = 0.1, fallback: tuple[float, float] = (4.0, 0.055)) -> SpeedForce:
    """p95 |v| and |a| (px per frame at `speed`) of the agents on screen, sampled every `step` s."""
    speeds: list[float] = []
    accels: list[float] = []
    for t in sample_times(trajectories, step):
        for piece in trajectories.active(t)[:max_agents]:
            s = per_frame(piece.state(t), speed)
            speeds.append(math.hypot(s.vx, s.vy))
            accels.append(math.hypot(s.ax, s.ay))
    if not speeds:
        return SpeedForce(fallback[0], fallback[1], 0.0, 0.0, 0)
    p95_v, p95_a = percentile(speeds, 0.95), percentile(accels, 0.95)
    return SpeedForce(
        max_speed=p95_v if p95_v > NEGLIGIBLE else fallback[0],
        max_force=p95_a / FORCE_TO_ENERGY if p95_a > NEGLIGIBLE else fallback[1],
        p95_speed=p95_v,
        p95_accel=p95_a,
        samples=len(speeds),
    )


@dataclass
class Normaliser:
    """Maps raw descriptors into 0..1 relative to one dataset."""
    lo: dict[str, float] = field(default_factory=dict)  # p5
    hi: dict[str, float] = field(default_factory=dict)  # p95
    max_count: int = 1

    def __call__(self, descriptors: dict[str, float], count: int) -> dict[str, float]:
        if count == 0:
            return descriptors  # defaults for an empty world, already 0 fish
        out = dict(descriptors)
        for key, value in descriptors.items():
            if key == "fish_count":
                out[key] = min(1.0, count / self.max_count)
            elif key in self.lo and self.hi[key] - self.lo[key] > 1e-9:
                out[key] = max(0.0, min(1.0, (value - self.lo[key]) / (self.hi[key] - self.lo[key])))
            # a key with no spread in this dataset keeps its raw value
        return out


def calibrate_descriptors(trajectories: Trajectories, speed: float, max_agents: int, speed_force: SpeedForce,
                          world: tuple[float, float], step: float = 0.4) -> Normaliser:
    """p5/p95 of each raw descriptor over the file (agents as on screen: capped, calibrated)."""
    width, height = world
    series: dict[str, list[float]] = {}
    max_count = 0
    for t in sample_times(trajectories, step):
        agents = []
        for piece in trajectories.active(t)[:max_agents]:
            s = per_frame(piece.state(t), speed)
            agents.append(SimpleNamespace(
                position=Vector2(s.x, s.y), velocity=Vector2(s.vx, s.vy), acceleration=Vector2(s.ax, s.ay),
                max_speed=speed_force.max_speed, max_force=speed_force.max_force,
            ))
        if not agents:
            continue
        max_count = max(max_count, len(agents))
        for key, value in compute_descriptors(agents, width, height).items():
            series.setdefault(key, []).append(value)
    norm = Normaliser(max_count=max(1, max_count))
    for key, values in series.items():
        norm.lo[key] = percentile(values, 0.05)
        norm.hi[key] = percentile(values, 0.95)
    return norm
