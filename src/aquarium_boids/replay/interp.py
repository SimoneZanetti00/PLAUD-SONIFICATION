"""Smooth motion from sparse keyframes (e.g. 2.5 rows per second) for a 60 fps world. Pure functions.

- Each id's samples become one or more pieces: a gap of more than `max_missing` missing rows
  ends the piece and the next sample starts a new one (a new agent later on).
- Inside a piece, position is a cubic Hermite spline through the keyframes. The tangent at a
  keyframe is the slope of the parabola through it and its two neighbours; with evenly spaced
  keyframes that is the central difference, i.e. Catmull-Rom. At the two ends of a piece the
  tangent is the one-sided slope of the parabola through the first (last) three keyframes,
  or the plain slope when the piece has only two. Quadratic motion (constant acceleration)
  is reproduced exactly, even across a missing row.
- Velocity is the spline's derivative (continuous). Acceleration is its second derivative:
  linear inside each keyframe interval, it can jump at a keyframe.
- Everything is in data units per second here; `per_frame` converts to the boids' units
  (px per 60 fps frame, px per frame squared) for a given replay speed.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from .loader import Dataset, Sample

SIM_FPS = 60.0


@dataclass(frozen=True)
class State:
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float


def _slopes(ts: list[float], ps: list[float]) -> tuple[list[float], list[float]]:
    h = [b - a for a, b in zip(ts, ts[1:])]
    d = [(pb - pa) / hk for pa, pb, hk in zip(ps, ps[1:], h)]
    return h, d


def tangents(ts: list[float], ps: list[float]) -> list[float]:
    """dp/dt at each keyframe (len(ts) >= 2, strictly increasing ts)."""
    h, d = _slopes(ts, ps)
    n = len(ts)
    if n == 2:
        return [d[0], d[0]]
    m = [0.0] * n
    for i in range(1, n - 1):  # parabola through i-1, i, i+1, derivative at i
        m[i] = (h[i] * d[i - 1] + h[i - 1] * d[i]) / (h[i - 1] + h[i])
    m[0] = ((2 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1])
    m[-1] = ((2 * h[-1] + h[-2]) * d[-1] - h[-1] * d[-2]) / (h[-1] + h[-2])
    return m


@dataclass(frozen=True)
class Piece:
    """One continuous stretch of one id: keyframe times, positions and tangents."""
    track_id: str
    index: int  # 0 = first stretch of this id, 1 = after the first gap, ...
    t: tuple[float, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    mx: tuple[float, ...]
    my: tuple[float, ...]

    @classmethod
    def from_samples(cls, track_id: str, index: int, samples: list[Sample]) -> "Piece":
        ts = [s.t for s in samples]
        xs = [s.x for s in samples]
        ys = [s.y for s in samples]
        return cls(track_id, index, tuple(ts), tuple(xs), tuple(ys), tuple(tangents(ts, xs)), tuple(tangents(ts, ys)))

    @property
    def t0(self) -> float:
        return self.t[0]

    @property
    def t1(self) -> float:
        return self.t[-1]

    def covers(self, t: float) -> bool:
        return self.t0 <= t <= self.t1

    def state(self, t: float) -> State:
        """Position, velocity and acceleration at time t (clamped into [t0, t1])."""
        t = min(max(t, self.t0), self.t1)
        i = min(max(bisect.bisect_right(self.t, t) - 1, 0), len(self.t) - 2)
        h = self.t[i + 1] - self.t[i]
        s = (t - self.t[i]) / h
        s2, s3 = s * s, s * s * s
        # Hermite basis (p0, h*m0, p1, h*m1) and its first and second derivatives in s.
        b = (2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2)
        db = (6 * s2 - 6 * s, 3 * s2 - 4 * s + 1, -6 * s2 + 6 * s, 3 * s2 - 2 * s)
        ddb = (12 * s - 6, 6 * s - 4, -12 * s + 6, 6 * s - 2)

        def axis(p: tuple[float, ...], m: tuple[float, ...]) -> tuple[float, float, float]:
            c = (p[i], h * m[i], p[i + 1], h * m[i + 1])
            pos = sum(bk * ck for bk, ck in zip(b, c))
            vel = sum(bk * ck for bk, ck in zip(db, c)) / h
            acc = sum(bk * ck for bk, ck in zip(ddb, c)) / (h * h)
            return pos, vel, acc

        x, vx, ax = axis(self.x, self.mx)
        y, vy, ay = axis(self.y, self.my)
        return State(x, y, vx, vy, ax, ay)


def split_track(samples: tuple[Sample, ...], frame_step: int, max_missing: int = 1) -> list[list[Sample]]:
    """Cut a track where more than `max_missing` consecutive rows are missing."""
    limit = (max_missing + 1) * frame_step
    runs: list[list[Sample]] = []
    for s in samples:
        if runs and s.frame - runs[-1][-1].frame <= limit:
            runs[-1].append(s)
        else:
            runs.append([s])
    return runs


class Trajectories:
    """All pieces of a dataset, with the set that is present at a given time."""

    def __init__(self, dataset: Dataset, max_missing: int = 1) -> None:
        if max_missing < 0:
            raise ValueError(f"max_missing must be >= 0, got {max_missing}")
        pieces: list[Piece] = []
        self.dropped = 0  # single-row stretches: no motion to interpolate, never shown
        for tid, samples in dataset.tracks.items():  # dataset order: first time, then id
            index = 0
            for run in split_track(samples, dataset.frame_step, max_missing):
                if len(run) < 2:
                    self.dropped += 1
                    continue
                pieces.append(Piece.from_samples(tid, index, run))
                index += 1
        if not pieces:
            raise ValueError("no id has two consecutive rows: nothing to replay")
        pieces.sort(key=lambda p: p.t0)  # stable: same start keeps dataset order
        self.pieces = pieces
        self._starts = [p.t0 for p in pieces]
        self.t_start = pieces[0].t0
        self.t_end = max(p.t1 for p in pieces)

    def active(self, t: float) -> list[Piece]:
        """Pieces present at time t, ordered by start time (then dataset order)."""
        n = bisect.bisect_right(self._starts, t)
        return [p for p in self.pieces[:n] if t <= p.t1]


def per_frame(state: State, speed: float = 1.0, fps: float = SIM_FPS) -> State:
    """Data units per second -> per simulation frame, at `speed` x real time.

    One frame advances data time by speed / fps seconds, so v scales by that and a by its square.
    """
    k = speed / fps
    return State(state.x, state.y, state.vx * k, state.vy * k, state.ax * k * k, state.ay * k * k)


class DataClock:
    """Data time driven by the frame scale dt of main.py (1.0 = one 60 fps frame)."""

    def __init__(self, start: float, end: float, speed: float = 1.0, loop: bool = False,
                 fps: float = SIM_FPS) -> None:
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")
        if end < start:
            raise ValueError(f"end {end} is before start {start}")
        self.start, self.end, self.speed, self.loop, self.fps = start, end, speed, loop, fps
        self.t = start
        self.wrapped = False  # True on the advance() that jumped back to the start

    @property
    def finished(self) -> bool:
        return not self.loop and self.t >= self.end

    def reset(self) -> None:
        self.t = self.start
        self.wrapped = False

    def advance(self, dt: float, paused: bool = False) -> float:
        self.wrapped = False
        if paused:
            return self.t
        self.t += dt / self.fps * self.speed
        if self.t > self.end:
            span = self.end - self.start
            if self.loop and span > 0:
                self.t = self.start + (self.t - self.start) % span
                self.wrapped = True
            else:
                self.t = self.end
        return self.t
