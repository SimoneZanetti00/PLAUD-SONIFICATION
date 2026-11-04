"""Profile scene for ants (--replay-profile scene with an ant --replay-config). Pure Python, no pygame.

The same outputs as the pedestrian scene profile (replay/mapping.py: territory, loudness, limit,
centroid, stretch, presence, L2 offset, grain density, slices, kicks, flashes), driven by what a
colony does. Positions in body lengths (BL), speeds in BL/s, with the config's thresholds.

- Count: ants on screen relative to the file's p5..p95 (as for people): loudness, limit, onset.
- Activity: the share of ants moving (faster than standing_bl_s) -> the rhythmicity target; the
  mean speed of the moving ones over speed_bl_s -> /plaud/centroid.
- Distance from the nest: the ants' mean distance from it, over the file's p5..p95 -> the texture
  target: near the nest = 0 (full, soft), far out = 1 (sparse, percussive, bright).
- Recruitment wave: the nest outflow (track pieces starting inside the nest radius, per minute over
  the last 2 minutes of data) above the file's p90, only after the first food discovery: PLAUD swells
  (onset target and loudness up, full at 1.5 x the p90). The start of a wave flashes on the nest.
- Encounters: two ants' centres closer than encounter_bl (counted again only after they have been
  apart by twice that), per minute over the last 30 s of data, over the file's p95 -> grain density
  (/gran/density, glided): a density, not one event each.
- Food discovery (from the places file, data time t of each item): the big moment. A slice
  (/pad/sample by where the item is, left to right) and a flash on the item; the first item found
  in each patch also gives a kick (no PLAUD shake).
- Arrivals: a slice for a newcomer, only for track pieces that start inside the nest radius
  (events.arrivals = "nest"; "anywhere" = every newcomer clearly visible, as for people).
- Closeness (mean nearest-neighbour distance over close_bl) -> stretch and limit, as for people.
Measurements run in data time (windows of data seconds); glides and the spacing of slices run in
playback time (data time / speed), so a trial played at 15x does not jump 15 times faster.
"""
from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .mapping import (
    LIMIT_ONE, STRETCH_MAX, Terrain, clip01, count_fraction, count_range, limit_level, loudness_level, master_level,
)
from .scene_config import SceneConfig

WAVE_WINDOW_S = 120.0  # data seconds of nest outflow
WAVE_FULL = 1.5  # outflow / its p90 at a full wave
WAVE_ON, WAVE_OFF = 0.25, 0.05  # wave level at which a wave starts (flash) / ends
WAVE_ONSET = 0.5  # onset target added at a full wave
WAVE_LOUDNESS = 0.35  # /plaud/loudness added at a full wave
ENCOUNTER_WINDOW_S = 30.0  # data seconds of encounters for the rate
ENCOUNTER_APART = 2.0  # x encounter_bl: a pair must be this far apart before it counts again
DENSITY_MAX = 60.0  # /gran/density at the file's p95 encounter rate
# glides and spacing in playback seconds
TERRITORY_TAU_S = 4.0
COUNT_TAU_S = 1.5
SCENE_TAU_S = 2.0
WAVE_TAU_S = 3.0
ENCOUNTER_TAU_S = 2.0
AGITATION_TAU_S = 4.0
MASTER_FADE_IN_S = 1.5
MASTER_FADE_OUT_S = 3.0
ARRIVAL_MIN_GAP_S = 8.0  # playback seconds between arrival slices
FOOD_MIN_GAP_S = 0.5  # playback seconds between food slices (none is dropped: they wait)
EVENT_KEEP = 32
STEP_RANGE_S = (0.1, 0.5)  # data seconds between scene checks: duration / 20000, within this range


@dataclass
class Ant:
    key: str
    x: float  # body lengths
    y: float
    vx: float  # BL/s
    vy: float
    pan: float = 0.5
    wx: float = 0.0  # world px
    wy: float = 0.0
    visible: bool = True
    from_nest: bool = False  # its track piece started inside the nest radius


@dataclass(frozen=True)
class AntCalibration:
    """File ranges, measured at load with the same scene checks as the replay."""
    step_s: float
    count_range: tuple[float, float]
    distance_bl: tuple[float, float]  # mean distance from the nest: p5, p95
    outflow_p90: float  # pieces leaving the nest per minute (2 min window)
    encounter_p95: float  # encounters per minute (30 s window)
    nest_starts: tuple[float, ...]  # data times of the pieces that start inside the nest radius


def scene_step(duration_s: float) -> float:
    lo, hi = STEP_RANGE_S
    return min(hi, max(lo, duration_s / 20000.0))


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def rate_per_min(times: tuple[float, ...] | list[float], t: float, window: float) -> float:
    """Events in (t - window, t] per minute; `times` sorted."""
    n = bisect.bisect_right(times, t) - bisect.bisect_right(times, t - window)
    return n * 60.0 / window


def mean_nest_distance(points: Iterable[tuple[float, float]], nest: tuple[float, float] | None) -> float | None:
    pts = list(points)
    if nest is None or not pts:
        return None
    return sum(math.hypot(x - nest[0], y - nest[1]) for x, y in pts) / len(pts)


def mean_nn_distance(points: list[tuple[float, float]]) -> float | None:
    """Mean distance to each ant's nearest neighbour; None with fewer than two."""
    if len(points) < 2:
        return None
    total = 0.0
    for i, (x, y) in enumerate(points):
        total += min(math.hypot(x - u, y - v) for j, (u, v) in enumerate(points) if j != i)
    return total / len(points)


class EncounterCounter:
    """New contacts between ants: centres closer than `near` (BL); a pair counts again only after it
    has been farther apart than ENCOUNTER_APART x near."""

    def __init__(self, near: float) -> None:
        self.near = near
        self.apart = ENCOUNTER_APART * near
        self._contact: set[tuple[str, str]] = set()

    def reset(self) -> None:
        self._contact.clear()

    def update(self, ants: list[tuple[str, float, float]]) -> list[tuple[str, str]]:
        cell = self.apart
        grid: dict[tuple[int, int], list[tuple[str, float, float]]] = {}
        for a in ants:
            grid.setdefault((math.floor(a[1] / cell), math.floor(a[2] / cell)), []).append(a)
        still: set[tuple[str, str]] = set()
        new: list[tuple[str, str]] = []
        for (cx, cy), members in grid.items():
            others = [b for dx in (-1, 0, 1) for dy in (-1, 0, 1) for b in grid.get((cx + dx, cy + dy), ())]
            for a in members:
                for b in others:
                    if not a[0] < b[0]:
                        continue
                    d = math.hypot(a[1] - b[1], a[2] - b[2])
                    key = (a[0], b[0])
                    if key in self._contact:
                        if d <= self.apart:
                            still.add(key)
                    elif d < self.near:
                        still.add(key)
                        new.append(key)
        self._contact = still
        return sorted(new)


def calibrate_ants(frames: Iterable[tuple[float, list[tuple[str, float, float]]]], step_s: float,
                   nest: tuple[float, float] | None, nest_starts: list[float], encounter_bl: float) -> AntCalibration:
    """frames: (t, [(key, x BL, y BL)]) every step_s of data time over the file."""
    counts: list[float] = []
    distances: list[float] = []
    enc_times: list[float] = []
    times: list[float] = []
    counter = EncounterCounter(encounter_bl)
    for t, ants in frames:
        times.append(t)
        counts.append(len(ants))
        d = mean_nest_distance(((x, y) for _, x, y in ants), nest)
        if d is not None:
            distances.append(d)
        enc_times += [t] * len(counter.update(ants))
    starts = tuple(sorted(nest_starts))
    outflow = [rate_per_min(starts, t, WAVE_WINDOW_S) for t in times]
    encounters = [rate_per_min(enc_times, t, ENCOUNTER_WINDOW_S) for t in times]
    return AntCalibration(
        step_s=step_s,
        count_range=count_range(percentile(counts, 0.05), percentile(counts, 0.95)) if counts else (1.0, 15.0),
        distance_bl=(percentile(distances, 0.05), percentile(distances, 0.95)) if distances else (0.0, 1.0),
        outflow_p90=percentile(outflow, 0.90),
        encounter_p95=percentile(encounters, 0.95),
        nest_starts=starts,
    )


@dataclass(frozen=True)
class FoodPlace:
    t: float  # data seconds of discovery
    wx: float  # world px
    wy: float
    patch: str


@dataclass
class AntMapping:
    """Profile scene with an ant config: data-time measurements, playback-time sound state."""
    config: SceneConfig
    calibration: AntCalibration
    max_count: int
    speed: float = 1.0  # playback speed: glides and spacing are in data time / speed
    terrain: Terrain | None = None
    nest_bl: tuple[float, float] | None = None
    food: tuple[FoodPlace, ...] = ()
    nest_world: tuple[float, float] | None = None
    # measured (data time)
    count: int = 0
    count01: float = 0.0
    moving: float = 0.0  # share of ants moving, glided
    speed01: float = 0.0  # mean speed of the moving ants over speed_bl_s, glided
    speed_bl_s: float = 0.0  # that mean speed in BL/s (now)
    distance_bl: float = 0.0  # mean distance from the nest (now)
    distance01: float = 0.0  # over the file's p5..p95, glided
    closeness: float = 0.0
    outflow: float = 0.0  # nest outflow per minute (2 min window)
    wave: float = 0.0  # 0..1, glided
    encounter_rate: float = 0.0  # per minute (30 s window)
    # sound
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    territory: tuple[float, float] = (0.5, 0.5)
    master: float = 0.0
    density: float = 0.0
    agitation: float = 0.0
    temp: float = 0.2
    loudness: float = 0.0
    limit: float = LIMIT_ONE
    centroid: float = 0.0
    stretch: float = 0.0
    pad_samples: list[int] = field(default_factory=list)
    kicks: int = 0
    events: list[tuple[str, float, tuple[str, ...]]] = field(default_factory=list)  # on ants: (kind, t, keys)
    place_events: list[tuple[str, float, tuple[float, float]]] = field(default_factory=list)  # (kind, t, world xy)
    arrivals_seen: int = 0
    arrivals_sent: int = 0
    food_sent: int = 0
    waves: int = 0
    encounters_seen: int = 0
    _known: set[str] = field(default_factory=set)
    _started: bool = False
    _scene_dt: float = 0.0
    _goal: tuple[float, float] = (0.5, 0.5)
    _wave_on: bool = False
    _next_food: int = 0
    _patches_found: set[str] = field(default_factory=set)
    _food_wait: float = 0.0  # playback seconds until the next food slice may go
    _arrival_wait: float = 0.0
    _enc_times: list[float] = field(default_factory=list)
    _counter: EncounterCounter | None = None
    _speed_now: float = 0.0
    _moving_now: float = 0.0
    _distance_now: float = 0.0
    _closeness_now: float = 0.0
    _wave_goal: float = 0.0

    def __post_init__(self) -> None:
        self.food = tuple(sorted(self.food, key=lambda f: f.t))
        self._counter = EncounterCounter(self.config.encounter_bl)
        firsts = [f.t for f in self.food]
        self.first_food_t = min(firsts) if firsts else None

    @property
    def count_range(self) -> tuple[float, float]:
        return self.calibration.count_range

    def reset(self) -> None:
        """Restart / loop: silence, nobody known, food not found yet."""
        self.count = 0
        self.count01 = self.moving = self.speed01 = self.distance01 = self.closeness = 0.0
        self.outflow = self.wave = self.encounter_rate = 0.0
        self.master = self.density = self.agitation = 0.0
        self.temp = 0.2
        self.loudness, self.limit, self.centroid, self.stretch = 0.0, LIMIT_ONE, 0.0, 0.0
        self.pad_samples.clear()
        self.kicks = 0
        self.events.clear()
        self.place_events.clear()
        self._known.clear()
        self._started = False
        self._scene_dt = 0.0
        self._wave_on = False
        self._next_food = 0
        self._patches_found.clear()
        self._food_wait = self._arrival_wait = 0.0
        self._enc_times.clear()
        self._counter.reset()

    def update(self, t: float, dt: float, ants: list[Ant], agitation_now: float = 0.0,
               pan_of: Callable[[float, float], float] | None = None) -> None:
        """One step of data time `dt` at data time t. pan_of: world px -> pan 0..1 (food slices)."""
        real = dt / self.speed if dt > 0 else 0.0
        self.count = len(ants)
        self.count01 = count_fraction(self.count, self.count_range)
        first = not self._started
        if first:  # start from the scene as it is: ants present are not arrivals, food already found is not news
            self._started = True
            self._known = {a.key for a in ants}
            while self._next_food < len(self.food) and self.food[self._next_food].t <= t:
                self._patches_found.add(self.food[self._next_food].patch)
                self._next_food += 1
            self._measure(t, ants)
            self.moving, self.speed01, self.distance01 = self._moving_now, self._speed_now, self._distance_now
            self.closeness = self._closeness_now
            self.wave = self._wave_goal
            self.agitation = agitation_now
            self.master = master_level(self.count, self.max_count)
            self.loudness = self._loudness_goal()
            self._set_target()
            self.territory = self._goal = self._pick(self._axes())
            self.limit = limit_level(self.count01, self.closeness)

        def glide(tau: float) -> float:
            return 1.0 - math.exp(-real / tau) if real > 0 else 0.0

        if dt > 0:
            self._scene_dt += dt
            if self._scene_dt >= self.calibration.step_s:
                self._scene_dt = 0.0
                self._measure(t, ants)
                if not first:
                    self._set_target()
                    self._goal = self._pick(self.territory)
        s = glide(SCENE_TAU_S)
        self.moving += s * (self._moving_now - self.moving)
        self.speed01 += s * (self._speed_now - self.speed01)
        self.distance01 += s * (self._distance_now - self.distance01)
        self.closeness += s * (self._closeness_now - self.closeness)
        self.wave += glide(WAVE_TAU_S) * (self._wave_goal - self.wave)
        self._wave_events(t)
        if self.count:
            g = glide(TERRITORY_TAU_S)
            tx, ty = self.territory
            self.territory = (tx + g * (self._goal[0] - tx), ty + g * (self._goal[1] - ty))
            self.limit += glide(COUNT_TAU_S) * (limit_level(self.count01, self.closeness) - self.limit)
        self.centroid = self.speed01
        self.stretch = STRETCH_MAX * self.closeness
        self.loudness += glide(COUNT_TAU_S) * (self._loudness_goal() - self.loudness)
        enc01 = clip01(self.encounter_rate / self.calibration.encounter_p95) if self.calibration.encounter_p95 > 0 else 0.0
        self.density += glide(ENCOUNTER_TAU_S) * (DENSITY_MAX * enc01 - self.density)
        self.agitation += glide(AGITATION_TAU_S) * (agitation_now - self.agitation)
        self.temp = clip01(0.2 + 1.3 * self.agitation)
        goal = master_level(self.count, self.max_count)
        tau = MASTER_FADE_IN_S if goal > self.master else MASTER_FADE_OUT_S
        if real > 0:
            self.master += (goal - self.master) * (1.0 - math.exp(-real / (tau / 3.0)))
            if goal == 0.0 and self.master < 0.005:
                self.master = 0.0
        self._food_wait = max(0.0, self._food_wait - real)
        self._arrival_wait = max(0.0, self._arrival_wait - real)
        self._food(t, pan_of)
        self._arrivals(t, ants)

    # --- measurements (data time) ---
    def _measure(self, t: float, ants: list[Ant]) -> None:
        c = self.config
        speeds = [math.hypot(a.vx, a.vy) for a in ants]
        movers = [v for v in speeds if v >= c.standing_bl_s]
        self._moving_now = len(movers) / len(ants) if ants else 0.0
        self.speed_bl_s = sum(movers) / len(movers) if movers else 0.0
        lo, hi = c.speed_bl_s
        self._speed_now = clip01((self.speed_bl_s - lo) / (hi - lo)) if movers else 0.0
        d = mean_nest_distance(((a.x, a.y) for a in ants), self.nest_bl)
        if d is not None:
            self.distance_bl = d
            dlo, dhi = self.calibration.distance_bl
            self._distance_now = clip01((d - dlo) / (dhi - dlo)) if dhi > dlo else 0.0
        nn = mean_nn_distance([(a.x, a.y) for a in ants])
        near, far = c.close_bl
        self._closeness_now = 0.0 if nn is None else clip01((far - nn) / (far - near))
        new = self._counter.update([(a.key, a.x, a.y) for a in ants])
        self.encounters_seen += len(new)
        self._enc_times += [t] * len(new)
        cut = bisect.bisect_right(self._enc_times, t - ENCOUNTER_WINDOW_S)
        del self._enc_times[:cut]
        self.encounter_rate = len(self._enc_times) * 60.0 / ENCOUNTER_WINDOW_S
        self.outflow = rate_per_min(self.calibration.nest_starts, t, WAVE_WINDOW_S)
        p90 = self.calibration.outflow_p90
        after_food = self.first_food_t is not None and t >= self.first_food_t
        if c.event_on("recruitment") and after_food and p90 > 0:
            self._wave_goal = clip01((self.outflow / p90 - 1.0) / (WAVE_FULL - 1.0))
        else:
            self._wave_goal = 0.0

    def _loudness_goal(self) -> float:
        base = loudness_level(self.count, self.count_range)
        return clip01(base + WAVE_LOUDNESS * self.wave) if self.count else 0.0

    def _set_target(self) -> None:
        self.target = (clip01(self.count01 + WAVE_ONSET * self.wave), self.moving, self.distance01)

    def _axes(self) -> tuple[float, float]:
        onset, rhythm, texture = self.target
        return (1.0 - (onset + rhythm) / 2.0, texture)

    def _pick(self, near: tuple[float, float]) -> tuple[float, float]:
        return self.terrain.best(self.target, near) if self.terrain else self._axes()

    # --- events ---
    def _send(self, kind: str, t: float, keys: tuple[str, ...]) -> None:
        self.events.append((kind, t, keys))
        del self.events[:-EVENT_KEEP]

    def _place(self, kind: str, t: float, xy: tuple[float, float]) -> None:
        self.place_events.append((kind, t, xy))
        del self.place_events[:-EVENT_KEEP]

    def _wave_events(self, t: float) -> None:
        if not self._wave_on and self.wave >= WAVE_ON:
            self._wave_on = True
            self.waves += 1
            if self.nest_world is not None:
                self._place("recruitment", t, self.nest_world)
        elif self._wave_on and self.wave < WAVE_OFF:
            self._wave_on = False

    def _food(self, t: float, pan_of) -> None:
        if not self.config.event_on("food_discovery"):
            return
        while self._next_food < len(self.food) and self.food[self._next_food].t <= t and self._food_wait <= 0:
            item = self.food[self._next_food]
            self._next_food += 1
            pan = pan_of(item.wx, item.wy) if pan_of else 0.5
            self.pad_samples.append(1 + min(3, int(pan * 4)))
            if item.patch not in self._patches_found:  # the first item of a patch: also a kick
                self._patches_found.add(item.patch)
                self.kicks += 1
            self.food_sent += 1
            self._place("food", t, (item.wx, item.wy))
            self._food_wait = FOOD_MIN_GAP_S

    def _arrivals(self, t: float, ants: list[Ant]) -> None:
        mode = self.config.events.get("arrivals", "off")
        for a in ants:
            if a.key in self._known:
                continue
            if mode == "anywhere" and not a.visible:
                continue  # counted later, once clearly in the picture
            self._known.add(a.key)
            if mode == "off" or (mode == "nest" and not a.from_nest):
                continue
            self.arrivals_seen += 1
            if self._arrival_wait <= 0:
                self._arrival_wait = ARRIVAL_MIN_GAP_S
                self.arrivals_sent += 1
                self.pad_samples.append(1 + min(3, int(a.pan * 4)))
                self._send("arrival", t, (a.key,))

    def hud_lines(self) -> list[str]:
        lo, hi = self.count_range
        db = 70.0 * (self.master - 1.0)
        cal = self.calibration
        return [
            f"sound ants | count {self.count01:.2f} ({self.count}, file {lo:.0f}-{hi:.0f}) | moving {self.moving:.2f} | "
            f"speed {self.speed_bl_s:.1f} BL/s ({self.speed01:.2f}) | nest distance {self.distance_bl:.0f} BL "
            f"({self.distance01:.2f}) | closeness {self.closeness:.2f}",
            f"colony | outflow {self.outflow:.1f}/min (p90 {cal.outflow_p90:.1f}) | wave {self.wave:.2f} "
            f"({self.waves}) | encounters {self.encounter_rate:.1f}/min (p95 {cal.encounter_p95:.1f})",
            f"PLAUD | territory {self.territory[0]:.2f}/{self.territory[1]:.2f} | loudness {self.loudness:.2f} | lim "
            f"{self.limit:.2f} | str {self.stretch:.2f} | centroid {self.centroid:.2f} | L2 {self.temp:.2f} | "
            f"{'silent' if self.master <= 0 else f'{db:.0f} dB'} | grains {self.density:.0f} | "
            f"arrivals {self.arrivals_sent}/{self.arrivals_seen}",
        ]
