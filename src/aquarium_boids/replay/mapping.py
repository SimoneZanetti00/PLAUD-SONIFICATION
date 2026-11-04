"""Replay sound mapping profiles (--replay-profile). Pure Python, no pygame.

synthfish (default): the simulation's mapping, unchanged (this module is not used).

scene: the sound follows the crowd's state instead of where it stands in the camera frame;
PLAUD reacts to the scene within seconds, and what you can point at in the picture is what you hear
as events.
- Count: the number of people on screen, relative to the file's own range: a log curve from its p5
  to its p95 (at least one person .. at least two more), so a sparse street and a dense plaza both
  use the whole range.
- Territory (/plaud/x, /plaud/y), feature targets: the scene sets wanted values of three terrain
  features and the lookup picks the best-matching cell of the PLAUD model's terrain map (--replay-terrain):
    count -> onset_density, mean walking speed -> rhythmicity,
    direction disorder -> texture (the percussive / bright / noisy direction, against loudness).
  Direction disorder is axial: people walking both ways along one line count as ordered, people
  crossing in every direction as disordered. A penalty on the distance from where the territory is
  now avoids far jumps between patches; the territory glides over ~4 s. Without a terrain map, the
  plain axes: x = 1 - (onset + rhythm) / 2 (busy and hurried = left), y_up = texture.
- PLAUD loudness latent (/plaud/loudness, the Loudness dial 0..1): the count, 0.15 at the file's p5
  (one person), 1 at its p95.
- PLAUD centroid latent (/plaud/centroid, the Centroid dial 0..1): the mean walking speed,
  0.4 .. 1.5 m/s -> 0 .. 1.
- Crowding tension, from closeness (the mean distance to each person's nearest neighbour, 2.5 m ..
  0.7 m -> 0 .. 1): /plaud/stretch (SpectralStretch "str", toward highs) = 0.5 x closeness;
  /plaud/limit (Limit "lim": 0 = all partials .. 1 = one) = 0.8 x (1 - count) + 0.3 x closeness, so
  more people give more partials and people pressed together thin them again.
  /plaud/limit, /plaud/centroid and /plaud/stretch need [route /plaud/limit /plaud/centroid
  /plaud/stretch] in the PLAUD device (docs/OSC.md).
- Presence (/mix/master, the PLAUD dry level): silence when nobody is there; a soft fade-in on the
  first arrival; a log curve with the number of people (absolute), 6 dB wide, up to the device's -6 dB.
- L2 offset (/plaud/temp): agitation averaged over ~4 s.
- Pan (/gran/fish_x): each person's position across the picture the viewer sees: the video column
  in the camera / video views, the map's x otherwise.
- Grain density (/gran/density): from the walkers' mean step rate (speed / 0.7 m), so the grain
  voices pulse at walking cadence (~1.9 Hz at 1.3 m/s); 0 when everybody stands.
- Events, only for people clearly visible (not within the frame's edge band), capped at one per
  EVENT_MIN_GAP_S per type (~4 per minute): an arrival (counted when the person is first clearly in
  the picture) plays a slice of the grain source (/pad/sample 1-4 by where they are, left to right);
  two people who start walking together give one kick, without the PLAUD shake (/fx/predator_strength
  0, then /pad/kick). Each sent event is kept in `events` so the picture can flash it at the same
  moment. The acceleration kick is off in this profile.
Everything advances in data time: a pause holds it, a restart or loop resets it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

PROFILES = ("synthfish", "scene")
ENV_TERRAIN = "PLAUD_TERRAIN"

TERRAIN_GRID = 32  # the 128x128 terrain map averaged down to this many cells per side
TERRAIN_JUMP_WEIGHT = 0.3  # cost per (map distance)^2 from the current territory, against feature mismatch
SCENE_STEP_S = 0.1  # scene check interval (pairs, closeness, disorder, territory target): 10 Hz
STEP_LENGTH_M = 0.7  # metres per step: step rate = speed / step length
STANDING_MS = 0.25  # m/s: slower people stand (no steps, no heading)
MASTER_TOP_DB = -6.0  # a full scene: the device's Dry default
MASTER_RANGE_DB = 6.0  # one person is this much quieter than a full scene (log curve in between)
TERRITORY_TAU_S = 4.0  # glide of the PLAUD territory (data seconds)
AGITATION_TAU_S = 4.0  # averaging of the agitation that drives the L2 offset
COUNT_RANGE_DEFAULT = (1.0, 15.0)  # people at count 0 and 1 when the file gives no range
LOUDNESS_ONE = 0.15  # /plaud/loudness at count 0 (one person, or the file's p5)
LIMIT_ONE = 0.8  # /plaud/limit at count 0, nobody close (0 = all partials, 1 = one partial)
LIMIT_TENSION = 0.3  # /plaud/limit added at full closeness
STRETCH_MAX = 0.5  # /plaud/stretch at full closeness
CLOSE_M = (0.7, 2.5)  # mean nearest-neighbour distance (m): closeness 1 .. 0
DISORDER_FULL = 0.6  # axial disorder for texture 1
COUNT_TAU_S = 1.5  # glide of loudness and limit when the count changes
SCENE_TAU_S = 2.0  # glide of speed, closeness and disorder
CENTROID_MS = (0.4, 1.5)  # mean walking speed (m/s) -> /plaud/centroid 0 .. 1
MASTER_FADE_IN_S = 1.5  # arrival after silence
MASTER_FADE_OUT_S = 3.0  # scene empties
EVENT_MIN_GAP_S = 15.0  # per event type: at most ~4 per minute
EVENT_KEEP = 32  # recent sent events kept for the picture
TOGETHER_HOLD_S = 2.0  # seconds of walking together before it counts as a pair
PAIR_STEP_S = SCENE_STEP_S  # the pair check runs at this data-time interval (the 2 s hold needs no more)
# "Walking together" (shared with the social graph, render_data.py): close, both moving, same way, same pace
TOGETHER_NEAR_M = 2.0
TOGETHER_MOVING_MS = 0.4
TOGETHER_HEADING_COS = math.cos(math.radians(30))
TOGETHER_SPEED_DIFF_MS = 0.3


def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def db_to_master(db: float) -> float:
    """The device maps /mix/master 0..1 linearly to -70..0 dB."""
    return clip01(1.0 + db / 70.0)


def master_level(count: int, max_count: int) -> float:
    """Target /mix/master for `count` people: 0 when empty, log curve up to the default level."""
    if count <= 0:
        return 0.0
    f = math.log1p(count) / math.log1p(max(1, max_count))
    return db_to_master(MASTER_TOP_DB - MASTER_RANGE_DB * (1.0 - min(1.0, f)))


def count_range(p5: float, p95: float) -> tuple[float, float]:
    """The file's people-on-screen p5 .. p95 as the count's 0 .. 1: at least one person, two more on top."""
    lo = max(1.0, p5)
    return lo, max(p95, lo + 2.0)


def count_fraction(count: int, rng: tuple[float, float] = COUNT_RANGE_DEFAULT) -> float:
    """0 at rng[0] people (or fewer), 1 at rng[1] (or more), log curve in between."""
    lo, hi = rng
    if count <= lo:
        return 0.0
    return clip01(math.log(count / lo) / math.log(hi / lo))


def loudness_level(count: int, rng: tuple[float, float] = COUNT_RANGE_DEFAULT) -> float:
    """/plaud/loudness target: 0 when empty, LOUDNESS_ONE at the bottom of the range, 1 at the top."""
    if count <= 0:
        return 0.0
    return LOUDNESS_ONE + (1.0 - LOUDNESS_ONE) * count_fraction(count, rng)


def limit_level(count01: float, closeness: float) -> float:
    """/plaud/limit target: more people, more partials; people pressed together thin them again."""
    return clip01(LIMIT_ONE * (1.0 - count01) + LIMIT_TENSION * closeness)


def centroid_level(mean_speed_ms: float) -> float:
    """0..1 from the mean speed of the people on screen (m/s): /plaud/centroid and the rhythm target."""
    lo, hi = CENTROID_MS
    return clip01((mean_speed_ms - lo) / (hi - lo))


def axial_disorder(velocities: list[tuple[float, float]]) -> float | None:
    """1 - the length of the mean doubled-angle heading of the walkers: 0 when everybody walks along one
    line (either way), ~1 when headings point everywhere. None with fewer than two walkers."""
    walkers = [v for v in velocities if math.hypot(*v) >= STANDING_MS]
    if len(walkers) < 2:
        return None
    c = s = 0.0
    for vx, vy in walkers:
        a = 2.0 * math.atan2(vy, vx)
        c += math.cos(a)
        s += math.sin(a)
    return 1.0 - math.hypot(c, s) / len(walkers)


def closeness_level(points: list[tuple[float, float]]) -> float | None:
    """0..1 from the mean distance (m) to each person's nearest neighbour; None with fewer than two people."""
    if len(points) < 2:
        return None
    total = 0.0
    for i, (x, y) in enumerate(points):
        total += min(math.hypot(x - u, y - v) for j, (u, v) in enumerate(points) if j != i)
    near, far = CLOSE_M
    return clip01((far - total / len(points)) / (far - near))


def cadence_density(step_rate_hz: float) -> float:
    """/gran/density so shoal_gran's base onset rate (0.5 + 0.075 x density grains/s) = step rate."""
    if step_rate_hz <= 0:
        return 0.0
    return max(0.0, min(100.0, (step_rate_hz - 0.5) / 0.075))


@dataclass
class Terrain:
    """A PLAUD terrain map (style pad), averaged to TERRAIN_GRID cells: per cell (x, y_up, onset, rhythm,
    texture), each feature scaled to its own p5..p95 over the map (0..1). texture = the mean z-score of
    percussiveness, brightness, noisiness and minus loudness: one value for the direction along which
    these tend to move together."""
    cells: list[tuple[float, float, float, float, float]]

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

        names = ("onset_density", "rhythmicity", "percussiveness", "brightness", "noisiness", "loudness")
        xy, raw = [], {k: [] for k in names}
        for i in range(g):  # row i: y from bottom
            for j in range(g):
                xy.append(((j + 0.5) / g, (i + 0.5) / g))
                for k in names:
                    raw[k].append(block(k, i, j))

        def z(vals: list[float]) -> list[float]:
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) or 1.0
            return [(v - m) / sd for v in vals]

        zs = {k: z(raw[k]) for k in ("percussiveness", "brightness", "noisiness", "loudness")}
        texture = [(zs["percussiveness"][c] + zs["brightness"][c] + zs["noisiness"][c] - zs["loudness"][c]) / 4
                   for c in range(len(xy))]

        def norm(vals: list[float]) -> list[float]:
            s = sorted(vals)
            lo, hi = s[len(s) // 20], s[len(s) * 19 // 20]
            return [clip01((v - lo) / (hi - lo)) if hi > lo else 0.5 for v in vals]

        on, rh, tx = norm(raw["onset_density"]), norm(raw["rhythmicity"]), norm(texture)
        return cls([(xy[c][0], xy[c][1], on[c], rh[c], tx[c]) for c in range(len(xy))])

    def best(self, target: tuple[float, float, float], near: tuple[float, float]) -> tuple[float, float]:
        """(x, y_up) of the cell whose (onset, rhythm, texture) best matches `target`, with a penalty
        on the squared distance from `near` (the current territory) so the point does not jump far."""
        to, tr, tt = target
        nx, ny = near
        c = min(self.cells, key=lambda c: (c[2] - to) ** 2 + (c[3] - tr) ** 2 + (c[4] - tt) ** 2
                + TERRAIN_JUMP_WEIGHT * ((c[0] - nx) ** 2 + (c[1] - ny) ** 2))
        return (c[0], c[1])

    def features_at(self, x: float, y: float) -> tuple[float, float, float]:
        """(onset, rhythm, texture) of the cell under (x, y_up)."""
        g = TERRAIN_GRID
        i, j = min(g - 1, max(0, int(y * g))), min(g - 1, max(0, int(x * g)))
        c = self.cells[i * g + j]
        return (c[2], c[3], c[4])


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
    pan: float  # 0..1 across the picture the viewer sees (left -> right)
    wx: float = 0.0  # world px
    wy: float = 0.0
    visible: bool = True  # clearly inside the picture (not in the edge band)


@dataclass
class SceneMapping:
    """Profile `scene`: data-time state and the values to send."""
    max_count: int
    terrain: Terrain | None = None
    count_range: tuple[float, float] = COUNT_RANGE_DEFAULT  # people for count 0 .. 1 (the file's p5 .. p95)
    count: int = 0
    count01: float = 0.0  # count_fraction of the people on screen
    speed01: float = 0.0  # centroid_level of the mean speed, glided
    disorder: float = 0.0  # axial direction disorder, glided
    closeness: float = 0.0  # 0..1, glided
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (onset, rhythm, texture) wanted from the terrain
    territory: tuple[float, float] = (0.5, 0.5)  # (x, y_up), the prior's style coordinates
    master: float = 0.0
    density: float = 0.0
    step_rate: float = 0.0
    agitation: float = 0.0  # mean |a| / p95, averaged over AGITATION_TAU_S
    temp: float = 0.2  # /plaud/temp (L2 offset)
    loudness: float = 0.0  # /plaud/loudness
    limit: float = LIMIT_ONE  # /plaud/limit
    centroid: float = 0.0  # /plaud/centroid
    stretch: float = 0.0  # /plaud/stretch
    pad_samples: list[int] = field(default_factory=list)  # arrivals not yet sent
    kicks: int = 0  # pair kicks not yet sent
    events: list[tuple[str, float, tuple[str, ...]]] = field(default_factory=list)  # sent: (kind, t, keys)
    arrivals_seen: int = 0
    together_seen: int = 0
    arrivals_sent: int = 0
    together_sent: int = 0
    _known: set[str] = field(default_factory=set)
    _together: dict[tuple[str, str], float] = field(default_factory=dict)
    _last_event: dict[str, float] = field(default_factory=dict)
    _started: bool = False
    _scene_dt: float = 0.0  # data time not yet given to the scene check
    _goal: tuple[float, float] = (0.5, 0.5)  # territory cell picked by the last scene check
    _disorder_now: float = 0.0
    _closeness_now: float = 0.0

    def reset(self) -> None:
        """Restart / loop: forget who was there; the sound state starts again from silence."""
        self.count = 0
        self.count01 = self.speed01 = self.disorder = self.closeness = 0.0
        self.master = self.density = self.step_rate = 0.0
        self.agitation = 0.0
        self.temp = 0.2
        self.loudness, self.limit, self.centroid, self.stretch = 0.0, LIMIT_ONE, 0.0, 0.0
        self.pad_samples.clear()
        self.kicks = 0
        self.events.clear()
        self._known.clear()
        self._together.clear()
        self._last_event.clear()
        self._started = False
        self._scene_dt = 0.0
        self._disorder_now = self._closeness_now = 0.0

    def update(self, t: float, dt: float, people: list[Person], agitation_now: float = 0.0) -> None:
        """One step of data time `dt` at time t. people: who is on screen; agitation_now: mean |a| / p95."""
        self.count = len(people)
        self.count01 = count_fraction(self.count, self.count_range)
        speed_now = centroid_level(sum(math.hypot(p.vx, p.vy) for p in people) / len(people)) if people else 0.0
        first = not self._started
        if first:  # first step: start from the scene as it is, people present are not arrivals
            self._started = True
            self.agitation = agitation_now
            self._known = {p.key for p in people}
            self.master = master_level(self.count, self.max_count)
            self.loudness = loudness_level(self.count, self.count_range)
            self.speed01 = speed_now
            self._measure(people)
            self.disorder, self.closeness = self._disorder_now, self._closeness_now
            self._set_target()
            self.territory = self._goal = self._pick(self._axes())  # no current territory yet: start near the axes
            self.limit = limit_level(self.count01, self.closeness)

        def glide(tau: float) -> float:
            return 1.0 - math.exp(-dt / tau) if dt > 0 else 0.0

        if dt > 0:
            self._scene_dt += dt
            if self._scene_dt >= SCENE_STEP_S:
                step, self._scene_dt = self._scene_dt, 0.0
                if self.count:
                    self._measure(people)
                self._pairs(t, step, people)
        if self.count:
            s = glide(SCENE_TAU_S)
            self.speed01 += s * (speed_now - self.speed01)
            self.disorder += s * (self._disorder_now - self.disorder)
            self.closeness += s * (self._closeness_now - self.closeness)
            if dt > 0 and self._scene_dt == 0.0 and not first:  # a scene check just ran: new target cell
                self._set_target()
                self._goal = self._pick(self.territory)
            g = glide(TERRITORY_TAU_S)
            tx, ty = self.territory
            self.territory = (tx + g * (self._goal[0] - tx), ty + g * (self._goal[1] - ty))
            self.limit += glide(COUNT_TAU_S) * (limit_level(self.count01, self.closeness) - self.limit)  # empty: held
        self.centroid = self.speed01
        self.stretch = STRETCH_MAX * self.closeness
        self.loudness += glide(COUNT_TAU_S) * (loudness_level(self.count, self.count_range) - self.loudness)
        self.agitation += glide(AGITATION_TAU_S) * (agitation_now - self.agitation)
        self.temp = clip01(0.2 + 1.3 * self.agitation)
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
        self._arrivals(t, people)

    def _measure(self, people: list[Person]) -> None:
        """Scene check: disorder and closeness of who is on screen (one person: ordered, nobody close)."""
        d = axial_disorder([(p.vx, p.vy) for p in people])
        c = closeness_level([(p.x, p.y) for p in people])
        self._disorder_now = 0.0 if d is None else d
        self._closeness_now = 0.0 if c is None else c

    def _set_target(self) -> None:
        self.target = (self.count01, self.speed01, clip01(self.disorder / DISORDER_FULL))

    def _axes(self) -> tuple[float, float]:
        """Plain-axes territory for the target: busy and hurried = left, disordered = up."""
        onset, rhythm, texture = self.target
        return (1.0 - (onset + rhythm) / 2.0, texture)

    def _pick(self, near: tuple[float, float]) -> tuple[float, float]:
        """Territory cell for the current target: the terrain's best match near `near`, else the plain axes."""
        return self.terrain.best(self.target, near) if self.terrain else self._axes()

    def _send(self, kind: str, t: float, keys: tuple[str, ...]) -> None:
        self.events.append((kind, t, keys))
        del self.events[:-EVENT_KEEP]

    def _allowed(self, kind: str, t: float) -> bool:
        last = self._last_event.get(kind)
        if last is not None and t - last < EVENT_MIN_GAP_S:
            return False
        self._last_event[kind] = t
        return True

    def _arrivals(self, t: float, people: list[Person]) -> None:
        for p in people:  # arrivals: counted when the newcomer is first clearly in the picture
            if p.key in self._known or not p.visible:
                continue
            self._known.add(p.key)
            self.arrivals_seen += 1
            if self._allowed("arrival", t):
                self.arrivals_sent += 1
                self.pad_samples.append(1 + min(3, int(p.pan * 4)))
                self._send("arrival", t, (p.key,))

    def _pairs(self, t: float, step: float, people: list[Person]) -> None:
        together: dict[tuple[str, str], float] = {}
        for a, b in _near_pairs(people):  # pairs starting to walk together (both clearly visible)
            if not (a.visible and b.visible) or not together_rule(math.hypot(a.x - b.x, a.y - b.y), (a.vx, a.vy), (b.vx, b.vy)):
                continue
            key = (a.key, b.key) if a.key < b.key else (b.key, a.key)
            before = self._together.get(key, 0.0)
            together[key] = before + step
            if before < TOGETHER_HOLD_S <= together[key]:
                self.together_seen += 1
                if self._allowed("together", t):
                    self.together_sent += 1
                    self.kicks += 1
                    self._send("together", t, key)
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
