"""Replay sound mapping profiles (--replay-profile). Pure Python, no pygame, no Max changes.

synthfish (default): the simulation's mapping, unchanged (this module is not used).

scene: the sound follows the crowd's state instead of where it stands in the camera frame.
- Territory (/plaud/x, /plaud/y): hurry (mean speed / the file's p95) -> horizontal, hurried = left
  (more rhythmic, denser onsets in the RafaelV2 terrain); crowding (people / most at once in the
  file) -> vertical, crowded = down (fuller, louder), sparse = up (more percussive, brighter). With
  the RafaelV2 terrain map (--replay-terrain) the point is the cell whose measured features match
  that state best, kept close to those axes; without it, the axes alone. Smoothed over seconds.
- Presence (/mix/master, the PLAUD dry level): silence when nobody is there; a soft fade-in on the
  first arrival; a gentle log curve with the number of people up to the device's default level.
- Grain density (/gran/density): from the walkers' mean step rate (speed / 0.7 m), so the grain
  voices pulse at walking cadence (~1.9 Hz at 1.3 m/s); 0 when everybody stands.
- Events, capped at one per EVENT_MIN_GAP_S per type (~4 per minute): an arrival plays a slice of
  the grain source (/pad/sample 1-4 by where the person enters, left to right); two people who
  start walking together give a short, gentle predator pulse at their midpoint (/fx/predator 1 then
  0: one kick and a PLAUD shake). The acceleration kick is off in this profile.
Everything advances in data time: a pause holds it, a restart or loop resets it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

PROFILES = ("synthfish", "scene")
ENV_TERRAIN = "PLAUD_TERRAIN"

SMOOTH_S = 4.0  # time constant of hurry / crowding (data seconds)
TERRAIN_GRID = 32  # the 128x128 terrain map averaged down to this many cells per side
TERRAIN_AXIS_WEIGHT = 1.0  # how strongly the point stays near the plain axes (vs feature match)
STEP_LENGTH_M = 0.7  # metres per step: step rate = speed / step length
STANDING_MS = 0.25  # m/s: slower people stand (no steps)
MASTER_DEFAULT_DB = -6.0  # the device's Dry default (live.gain~ init -6 dB): a full scene reaches it
MASTER_RANGE_DB = 9.0  # one person is this much quieter than a full scene (log curve in between)
MASTER_FADE_IN_S = 1.5  # arrival after silence
MASTER_FADE_OUT_S = 3.0  # scene empties
EVENT_MIN_GAP_S = 15.0  # per event type: at most ~4 per minute
TOGETHER_HOLD_S = 2.0  # seconds of walking together before it counts as a pair
PAIR_STEP_S = 0.1  # the pair check runs at this data-time interval (10 Hz; the 2 s hold needs no more)
# "Walking together" (shared with the social graph, render_data.py): close, both moving, same way, same pace
TOGETHER_NEAR_M = 2.0
TOGETHER_MOVING_MS = 0.4
TOGETHER_HEADING_COS = math.cos(math.radians(30))
TOGETHER_SPEED_DIFF_MS = 0.3
PULSE_S = 0.5  # length of the together pulse (one kick: shoal_fx repeats only after 800 ms)
PULSE_STRENGTH = 0.35  # /fx/predator_strength of a together pulse (0..1)


def db_to_master(db: float) -> float:
    """The device maps /mix/master 0..1 linearly to -70..0 dB."""
    return max(0.0, min(1.0, 1.0 + db / 70.0))


def master_level(count: int, max_count: int) -> float:
    """Target /mix/master for `count` people: 0 when empty, log curve up to the default level."""
    if count <= 0:
        return 0.0
    f = math.log1p(count) / math.log1p(max(1, max_count))
    return db_to_master(MASTER_DEFAULT_DB - MASTER_RANGE_DB * (1.0 - min(1.0, f)))


def cadence_density(step_rate_hz: float) -> float:
    """/gran/density so shoal_gran's base onset rate (0.5 + 0.075 x density grains/s) = step rate."""
    if step_rate_hz <= 0:
        return 0.0
    return max(0.0, min(100.0, (step_rate_hz - 0.5) / 0.075))


@dataclass
class Terrain:
    """The RafaelV2 style map, averaged to TERRAIN_GRID cells: per cell (x, y_up, rhythm, fullness)."""
    cells: list[tuple[float, float, float, float]]
    _memo: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "Terrain":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        n = int(d["resolution"])
        f = d["features"]
        if d.get("orientation") not in (None, "row0_is_y0_bottom"):
            raise ValueError(f"unexpected terrain orientation {d.get('orientation')!r}")
        g, b = TERRAIN_GRID, n // TERRAIN_GRID

        def block(name: str, i: int, j: int) -> float:
            rows = f[name][i * b:(i + 1) * b]
            return sum(sum(r[j * b:(j + 1) * b]) for r in rows) / (b * b)

        raw = []
        for i in range(g):  # row i: y from bottom
            for j in range(g):
                rhythm = (block("rhythmicity", i, j) + block("onset_density", i, j)) / 2
                raw.append(((j + 0.5) / g, (i + 0.5) / g, rhythm, block("loudness", i, j)))

        def norm(k: int) -> list[float]:
            vals = sorted(c[k] for c in raw)
            lo, hi = vals[len(vals) // 20], vals[len(vals) * 19 // 20]
            return [max(0.0, min(1.0, (c[k] - lo) / (hi - lo))) if hi > lo else 0.5 for c in raw]

        r, l = norm(2), norm(3)
        return cls([(c[0], c[1], r[i], l[i]) for i, c in enumerate(raw)])

    def point(self, hurry: float, crowding: float) -> tuple[float, float]:
        """(x, y_up) of the cell that best matches rhythm ~ hurry and fullness ~ crowding,
        near the plain axes (x = 1 - hurry, y_up = 1 - crowding). Memoised on hundredths."""
        key = (round(hurry * 100), round(crowding * 100))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        hurry, crowding = key[0] / 100, key[1] / 100
        ax, ay = 1.0 - hurry, 1.0 - crowding
        best = min(self.cells, key=lambda c: (c[2] - hurry) ** 2 + (c[3] - crowding) ** 2
                   + TERRAIN_AXIS_WEIGHT * ((c[0] - ax) ** 2 + (c[1] - ay) ** 2))
        self._memo[key] = (best[0], best[1])
        return self._memo[key]


def together_rule(dist_m: float, va: tuple[float, float], vb: tuple[float, float]) -> bool:
    """Two people walk together: closer than 2 m, both faster than 0.4 m/s, headings within 30 deg,
    speeds within 0.3 m/s. Velocities in m/s."""
    sa, sb = math.hypot(*va), math.hypot(*vb)
    if dist_m >= TOGETHER_NEAR_M or sa < TOGETHER_MOVING_MS or sb < TOGETHER_MOVING_MS:
        return False
    cos = (va[0] * vb[0] + va[1] * vb[1]) / (sa * sb)
    return cos > TOGETHER_HEADING_COS and abs(sa - sb) < TOGETHER_SPEED_DIFF_MS


@dataclass
class Person:
    key: str
    x: float  # metres (any origin) - only distances are used
    y: float
    vx: float  # m/s
    vy: float
    x01: float  # horizontal position in the world, 0..1 (for the slice choice)
    wx: float = 0.0  # world px (for the pulse position)
    wy: float = 0.0


@dataclass
class SceneMapping:
    """Profile `scene`: data-time state and the values to send."""
    max_count: int
    terrain: Terrain | None = None
    hurry: float = 0.0
    crowding: float = 0.0
    territory: tuple[float, float] = (0.5, 0.5)  # (x, y_up), the prior's style coordinates
    master: float = 0.0
    density: float = 0.0
    step_rate: float = 0.0
    count: int = 0
    pad_samples: list[int] = field(default_factory=list)  # arrivals not yet sent
    pulse: tuple[float, float, float] | None = None  # (world px x, y of the pair midpoint, seconds left)
    arrivals_seen: int = 0
    together_seen: int = 0
    arrivals_sent: int = 0
    together_sent: int = 0
    _known: set[str] = field(default_factory=set)
    _together: dict[tuple[str, str], float] = field(default_factory=dict)
    _last_event: dict[str, float] = field(default_factory=dict)
    _started: bool = False
    _pair_dt: float = 0.0  # data time not yet given to the pair check

    def reset(self) -> None:
        """Restart / loop: forget who was there; the sound state starts again from silence."""
        self.hurry = self.crowding = 0.0
        self.master = self.density = self.step_rate = 0.0
        self.count = 0
        self.pad_samples.clear()
        self.pulse = None
        self._known.clear()
        self._together.clear()
        self._last_event.clear()
        self._started = False
        self._pair_dt = 0.0

    def update(self, t: float, dt: float, people: list[Person], hurry_now: float) -> None:
        """One step of data time `dt` at time t. people: who is on screen; hurry_now: mean speed / p95."""
        self.count = len(people)
        crowding_now = min(1.0, self.count / max(1, self.max_count))
        if not self._started:  # first step: start from the scene as it is, people present are not arrivals
            self._started = True
            self.hurry, self.crowding = hurry_now, crowding_now
            self._known = {p.key for p in people}
            self.master = master_level(self.count, self.max_count)
        a = 1.0 - math.exp(-dt / SMOOTH_S) if dt > 0 else 0.0
        if self.count:
            self.hurry += a * (hurry_now - self.hurry)
            self.crowding += a * (crowding_now - self.crowding)
            target = self.terrain.point(self.hurry, self.crowding) if self.terrain else \
                (1.0 - self.hurry, 1.0 - self.crowding)
            tx, ty = self.territory
            self.territory = (tx + a * (target[0] - tx), ty + a * (target[1] - ty))
        # presence: exponential approach, faster in than out
        goal = master_level(self.count, self.max_count)
        tau = MASTER_FADE_IN_S if goal > self.master else MASTER_FADE_OUT_S
        if dt > 0:
            self.master += (goal - self.master) * (1.0 - math.exp(-dt / (tau / 3.0)))
            if goal == 0.0 and self.master < 0.005:
                self.master = 0.0
        # grain density from the walkers' step rate
        walking = [math.hypot(p.vx, p.vy) for p in people if math.hypot(p.vx, p.vy) >= STANDING_MS]
        self.step_rate = (sum(walking) / len(walking)) / STEP_LENGTH_M if walking else 0.0
        self.density = cadence_density(self.step_rate)
        self._events(t, dt, people)
        if self.pulse is not None and dt > 0:
            x, y, left = self.pulse
            self.pulse = (x, y, left - dt) if left - dt > 0 else None

    def _allowed(self, kind: str, t: float) -> bool:
        last = self._last_event.get(kind)
        if last is not None and t - last < EVENT_MIN_GAP_S:
            return False
        self._last_event[kind] = t
        return True

    def _events(self, t: float, dt: float, people: list[Person]) -> None:
        for p in people:  # arrivals
            if p.key not in self._known:
                self._known.add(p.key)
                self.arrivals_seen += 1
                if self._allowed("arrival", t):
                    self.arrivals_sent += 1
                    self.pad_samples.append(1 + min(3, int(p.x01 * 4)))
        if dt <= 0:
            return
        self._pair_dt += dt
        if self._pair_dt < PAIR_STEP_S:
            return
        step, self._pair_dt = self._pair_dt, 0.0
        together: dict[tuple[str, str], float] = {}
        for a, b in _near_pairs(people):  # pairs starting to walk together
            if not together_rule(math.hypot(a.x - b.x, a.y - b.y), (a.vx, a.vy), (b.vx, b.vy)):
                continue
            key = (a.key, b.key) if a.key < b.key else (b.key, a.key)
            before = self._together.get(key, 0.0)
            together[key] = before + step
            if before < TOGETHER_HOLD_S <= together[key]:
                self.together_seen += 1
                if self.pulse is None and self._allowed("together", t):
                    self.together_sent += 1
                    self.pulse = ((a.wx + b.wx) / 2, (a.wy + b.wy) / 2, PULSE_S)
        self._together = together


def _near_pairs(people: list[Person]):
    """Pairs closer than TOGETHER_NEAR_M, found through a grid of that cell size (not all n^2 pairs)."""
    cell = TOGETHER_NEAR_M
    grid: dict[tuple[int, int], list[Person]] = {}
    for p in people:
        grid.setdefault((math.floor(p.x / cell), math.floor(p.y / cell)), []).append(p)
    for (cx, cy), members in grid.items():
        for i, a in enumerate(members):  # same cell
            for b in members[i + 1:]:
                yield a, b
        for dx, dy in ((1, -1), (1, 0), (1, 1), (0, 1)):  # half of the neighbours: each pair once
            for b in grid.get((cx + dx, cy + dy), ()):
                for a in members:
                    yield a, b

