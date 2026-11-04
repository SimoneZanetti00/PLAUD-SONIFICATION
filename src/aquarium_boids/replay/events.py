"""Acceleration events in replay mode. Pure, no pygame.

A rolling z-score of one signal per frame (ReplaySource feeds the agitation: mean |a| over the
calibrated p95 |a|), over the last `window` seconds of data time:
- onset when z > on and rising, outside the refractory period (after a warm-up of window / 2);
  defaults z > 3.0 and 5 s refractory (first proposed 2.5 / 2 s: 7-10 onsets per minute on
  zara1, hotel and univ, too busy for the kick);
- the event lasts at least `hold` s and ends when z stays below `off` for `release` s;
- after the end, no new onset for `refractory` s.
Everything runs in data time, so the result does not depend on the replay speed or the frame rate.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EventParams:
    on: float = 3.0
    off: float = 1.5
    hold: float = 0.5
    release: float = 0.3
    refractory: float = 5.0
    window: float = 10.0
    std_floor: float = 0.02  # signal units: a flat stretch does not turn noise into huge z


class EventDetector:
    def __init__(self, params: EventParams = EventParams()) -> None:
        self.params = params
        self.reset()

    def reset(self) -> None:
        self._window: collections.deque[tuple[float, float]] = collections.deque()
        self._sum = self._sum2 = 0.0
        self._start: float | None = None
        self._below_since: float | None = None
        self._refractory_until = -math.inf
        self.z = 0.0
        self.active = False
        self.onset_t = 0.0
        self.peak_z = 0.0
        self.onset = False  # True on the update that started an event
        self.released = False  # True on the update that ended one

    def update(self, t: float, value: float) -> bool:
        """Feed one sample at data time t; returns whether an event is active."""
        p = self.params
        self.onset = self.released = False
        if self._start is None:
            self._start = t
        self._window.append((t, value))
        self._sum += value
        self._sum2 += value * value
        while self._window and self._window[0][0] < t - p.window:
            _, old = self._window.popleft()
            self._sum -= old
            self._sum2 -= old * old
        n = len(self._window)
        z_prev = self.z
        if t - self._start < p.window / 2 or n < 2:
            self.z = 0.0
            return self.active
        mean = self._sum / n
        std = math.sqrt(max(0.0, self._sum2 / n - mean * mean))
        self.z = (value - mean) / max(std, p.std_floor)
        if not self.active:
            if t >= self._refractory_until and self.z > p.on and self.z > z_prev:
                self.active = self.onset = True
                self.onset_t = t
                self.peak_z = self.z
                self._below_since = None
        else:
            self.peak_z = max(self.peak_z, self.z)
            if self.z < p.off:
                if self._below_since is None:
                    self._below_since = t
                if t - self._below_since >= p.release and t - self.onset_t >= p.hold:
                    self.active = False
                    self.released = True
                    self._refractory_until = t + p.refractory
            else:
                self._below_since = None
        return self.active

    def strength(self) -> float:
        """Predator strength 0..3 (0 when idle): 3 x (0.3 + 0.7 x clip((z - on) / 2.5)) on the event's
        peak z so far. The first formula, without the 0.3 floor, is ~0 at the onset, when z has only
        just crossed the threshold, so the first kick would be silent; with it every onset is audible."""
        if not self.active:
            return 0.0
        return 3.0 * (0.3 + 0.7 * max(0.0, min(1.0, (self.peak_z - self.params.on) / 2.5)))
