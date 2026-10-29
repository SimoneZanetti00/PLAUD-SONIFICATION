"""Where the fish come from: a source owns the agent list and moves it one frame at a time.

main.py only calls `reset`, `step` and `normalise`, then hands `agents` to the descriptors, OSC
and the renderer. SimulationSource is the boid simulation, the code that used to run inline in main.py.
ReplaySource moves the same Boid objects along recorded trajectories (--replay FILE).
"""
from __future__ import annotations

import bisect
import zlib
from pathlib import Path

from pygame.math import Vector2

from .boids import Boid
from .config import BOID_COUNT, CONTROL_PORT, HEIGHT, OSC_HOST, OSC_PORT, PLAIN_UDP_PORT, WIDTH
from .controls import RuntimeControls
from .replay.calibrate import Normaliser, calibrate_descriptors, calibrate_speed_force
from .replay.events import EventDetector, EventParams
from .replay.interp import DataClock, Piece, Trajectories, per_frame
from .replay.loader import data_bounds, fit_to_world, load_trajectories, resolve_format, to_world

REPLAY_MARGIN = 60.0  # world px kept free around the data (same as the joystick margin)
REPLAY_COLORS = [(80, 210, 255), (105, 255, 190), (255, 190, 95), (220, 160, 255)]  # Boid.create's palette
# ETH/UCY file names (Social GAN / Trajectron++) -> scene names used in the literature.
ETH_UCY_SCENES = {
    "biwi_eth": "eth",
    "biwi_hotel": "hotel",
    "crowds_zara01": "zara1",
    "crowds_zara02": "zara2",
    "crowds_zara03": "zara3",
    "students001": "univ (students001)",
    "students003": "univ (students003)",
    "uni_examples": "univ (examples)",
}


def scene_name(path: str | Path, fmt: str) -> tuple[str, str]:
    """(scene, source label), e.g. ("zara1", "ETH/UCY") or ("walk", "CSV")."""
    stem = Path(path).stem
    if resolve_format(path, fmt) == "eth-ucy":
        return ETH_UCY_SCENES.get(stem, stem), "ETH/UCY"
    return stem, "CSV"


# Who recorded the data: short form for the caption, long form for the HUD.
ETH_CITATION = ("ETH: Pellegrini et al. 2009",
                "ETH (BIWI): Pellegrini, Ess, Schindler & van Gool, You'll Never Walk Alone, ICCV 2009")
UCY_CITATION = ("UCY: Lerner et al. 2007",
                "UCY: Lerner, Chrysanthou & Lischinski, Crowds by Example, Eurographics 2007")
BOTH_CITATION = ("ETH/UCY: Lerner et al. 2007 / Pellegrini et al. 2009",
                 f"{UCY_CITATION[1]} / {ETH_CITATION[1]}")


def scene_citation(path: str | Path, fmt: str) -> tuple[str, str]:
    """(short, long) citation of the dataset a file comes from; ("CSV", file name) for other files."""
    stem = Path(path).stem
    if resolve_format(path, fmt) != "eth-ucy":
        return "CSV", f"{Path(path).name} (CSV)"
    if stem.startswith("biwi_"):
        return ETH_CITATION
    if stem in ETH_UCY_SCENES:
        return UCY_CITATION
    return BOTH_CITATION


def scene_title(path: str | Path, fmt: str) -> str:
    """Window title for a replay, with the scene's own source like the caption, e.g.
    "PLAUD-SONIFICATION - zara1 - real trajectories (UCY: Lerner et al. 2007)"."""
    scene, _ = scene_name(path, fmt)
    return f"PLAUD-SONIFICATION - {scene} - real trajectories ({scene_citation(path, fmt)[0]})"


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds, 6)))  # 4.9999999 after 300 frames of 1/60 s is 00:05
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def create_flock(count: int) -> list[Boid]:
    return [Boid.create(i, WIDTH, HEIGHT) for i in range(count)]


def resize_flock(flock: list[Boid], target: int) -> list[Boid]:
    """Grow or shrink the flock in place to `target` fish (1..BOID_COUNT).

    Lets the number of fish be a live control (keyboard -/= or MIDIMIX fader),
    so fish_count -> granular density/probability actually varies.
    """
    target = max(1, min(BOID_COUNT, int(target)))
    if len(flock) > target:
        del flock[target:]
    elif len(flock) < target:
        for i in range(len(flock), target):
            flock.append(Boid.create(i, WIDTH, HEIGHT))
    return flock


class SimulationSource:
    """The boids: same calls, same order, same random draws as the former inline code."""

    def __init__(self, count: int) -> None:
        self.agents = create_flock(count)

    def reset(self, population: int) -> None:
        """R key: a fresh flock of `population` fish."""
        self.agents = create_flock(population)

    def step(
        self,
        dt: float,
        controls: RuntimeControls,
        attractor: Vector2 | None,
        repeller: Vector2 | None,
    ) -> list[Boid]:
        """Resize to controls.population, then flock + update every fish (not while paused)."""
        flock = resize_flock(self.agents, controls.population)
        if not controls.paused:
            for boid in flock:
                boid.flock(flock, controls, WIDTH, HEIGHT, attractor=attractor, repeller=repeller)
            for boid in flock:
                boid.update(dt_scale=dt)
        return flock

    repeller: Vector2 | None = None  # no data-driven events in the simulation

    def normalise(self, descriptors: dict[str, float]) -> dict[str, float]:
        """The simulation's descriptors are already on their own scale: returned as they are."""
        return descriptors


class ReplaySource:
    """Recorded trajectories instead of the boid rules: one Boid per track piece, never flocked.

    - Data time advances with dt (frames) x speed; SPACE pauses it, R restarts the file.
    - Position, velocity and acceleration come from the spline (replay/interp.py), in px per
      60 fps frame like the boids.
    - Calibration at load: every agent's max_speed = p95 |v| and max_force = p95 |a| / 6 of the
      file at this speed, so mean_speed and energy mean "fast / agitated for this dataset".
    - normalise=True (--replay-normalise): descriptors are mapped to the file's p5..p95 range
      before the smoother; fish_count = agents / most agents ever on screen.
    - Stable slots: an agent keeps its list index while its piece lasts, so /gran/fish_x voice i
      stays the same fish. When one leaves, the last agent takes its index (one voice changes
      instead of every voice above it); a newcomer is appended, up to max_agents.
    - Size and colour come from the track id, never from `random`: the boids' random stream
      (Markov layer, renderer) is left alone.
    - Mouse food/predator and the weight keys have no effect on the data. controls.population
      mirrors the number of agents on screen, so the HUD shows it and -/= do nothing.
    - At the end of the data (without loop) the last frame holds, like a pause.
    - Events (REPLAY_04 3.7, on unless events=False): a z-score detector on the agitation (mean |a|
      over the calibrated p95) in data time. At an onset `repeller` is set to the position of the
      most accelerating person, so main.py sends /fx/predator 1 and the renderers draw the
      rings there; controls.predator_strength follows the event (restored after it). The repeller
      exerts no force: the data already contains the people's reaction.
    """

    def __init__(
        self,
        path: str | Path,
        fmt: str = "auto",
        fps: float = 2.5,
        speed: float = 1.0,
        loop: bool = False,
        flip_y: bool = False,
        max_agents: int = BOID_COUNT,
        normalise: bool = False,
        events: bool = True,
        event_params: EventParams = EventParams(),
        world: tuple[float, float] = (WIDTH, HEIGHT),
    ) -> None:
        if max_agents < 1:
            raise ValueError(f"max agents must be >= 1, got {max_agents}")
        dataset = load_trajectories(path, fmt=fmt, fps=fps)
        transform = fit_to_world(data_bounds(dataset), world, REPLAY_MARGIN, flip_y=flip_y)
        self.trajectories = Trajectories(to_world(dataset, transform))
        self.clock = DataClock(self.trajectories.t_start, self.trajectories.t_end, speed=speed, loop=loop)
        self.speed = speed
        self.max_agents = max_agents
        self.title = scene_title(path, fmt)
        self.scene, self.source_label = scene_name(path, fmt)
        self.citation, self.reference = scene_citation(path, fmt)
        self.unit = "m" if resolve_format(path, fmt) == "eth-ucy" else "unit"
        self.transform = transform  # data units <-> world px, for the metric grid of the data look
        self.loop = loop
        self.max_population = dataset.max_population
        self.calibration = calibrate_speed_force(self.trajectories, speed, max_agents)
        self.normaliser: Normaliser | None = (
            calibrate_descriptors(self.trajectories, speed, max_agents, self.calibration, world) if normalise else None
        )
        self.detector = EventDetector(event_params) if events else None
        self.repeller: Vector2 | None = None  # set during an event (read by main.py)
        self.event_log: list[tuple[float, tuple[float, float], int, float]] = []  # (s from start, data xy, people, z)
        self._saved_strength: float | None = None  # controls.predator_strength before an event
        self.agents: list[Boid] = []
        self._pieces: list[Piece] = []  # parallel to agents
        self.peak_waiting = 0  # most pieces kept off screen at once by max_agents
        self._end_reported = False
        self.summary = (
            f"{Path(path).name}: {len(dataset.tracks)} ids, {len(self.trajectories.pieces)} pieces "
            f"({self.trajectories.dropped} single rows dropped), {dataset.duration:.1f} s of data, "
            f"frame step {dataset.frame_step}, up to {dataset.max_population} at once, "
            f"scale {transform.scale:.2f} px/unit, speed x{speed:g}, loop {'on' if loop else 'off'}, "
            f"cap {max_agents}; max_speed {self.calibration.max_speed:.3f} px/frame, "
            f"max_force {self.calibration.max_force:.5f} px/frame^2 (p95 of {self.calibration.samples} samples), "
            f"normalise {'on' if normalise else 'off'}, events "
            + (f"z > {event_params.on:g}, refractory {event_params.refractory:g} s" if events else "off")
        )
        self._sync(self.clock.t)

    def reset(self, population: int) -> None:
        """R key: restart the file (population is ignored)."""
        self.clock.reset()
        self.agents = []
        self._pieces = []
        self._end_reported = False
        self._end_event()
        self._sync(self.clock.t)

    def step(
        self,
        dt: float,
        controls: RuntimeControls,
        attractor: Vector2 | None,
        repeller: Vector2 | None,
    ) -> list[Boid]:
        if not controls.paused and not self.clock.finished:
            t = self.clock.advance(dt)
            if self.clock.wrapped:  # loop: data time jumped back, the rolling window is stale
                self._end_event()
            self._sync(t)
            for boid in self.agents:  # Boid.update's flash decay; update() itself is never called
                boid.flash = max(0.0, boid.flash - 0.045 * dt)
            self._detect(t)
        if self.clock.finished and not self._end_reported:
            print("[replay] end of data, holding the last frame (R restarts)")
            self._end_reported = True
            self._end_event()
        self._apply_strength(controls)
        controls.population = len(self.agents)
        return self.agents

    def _detect(self, t: float) -> None:
        if self.detector is None:
            return
        agents = self.agents
        p95_accel = self.calibration.max_force * 6.0
        value = sum(b.acceleration.length() for b in agents) / len(agents) / p95_accel if agents else 0.0
        self.detector.update(t, value)
        if self.detector.onset and agents:
            strongest = max(agents, key=lambda b: b.acceleration.length())  # the person who triggered it
            centre = Vector2(strongest.position)
            self.repeller = centre
            self.event_log.append((t - self.clock.start, self.transform.to_data(centre.x, centre.y), len(agents),
                                   self.detector.z))
        elif self.detector.released:
            self.repeller = None

    def _end_event(self) -> None:
        """Drop a running event (restart, loop wrap, end of data) and clear the detector's window."""
        self.repeller = None
        if self.detector is not None:
            self.detector.reset()

    def _apply_strength(self, controls: RuntimeControls) -> None:
        """During an event predator_strength follows it; afterwards the previous value comes back."""
        if self.detector is not None and self.detector.active:
            if self._saved_strength is None:
                self._saved_strength = controls.predator_strength
            controls.predator_strength = self.detector.strength()
        elif self._saved_strength is not None:
            controls.predator_strength = self._saved_strength
            self._saved_strength = None

    def event_status(self) -> str:
        """One HUD/label line about events."""
        if self.detector is None:
            return "events off"
        minutes = max(1e-9, (self.clock.t - self.clock.start) / 60.0)
        head = f"events {len(self.event_log)} ({len(self.event_log) / minutes:.1f}/min)" if minutes > 0.1 \
            else f"events {len(self.event_log)}"
        if self.detector.active:
            return f"{head} | ACTIVE z {self.detector.peak_z:.1f} strength {self.detector.strength() / 3:.2f}"
        return f"{head} | z {self.detector.z:+.1f}"

    def caption(self, extra: str = "") -> str:
        """Scene, citation, optional extra credit and data time, e.g.
        "... zara1 - real trajectories (UCY: Lerner et al. 2007)   01:23 / 06:00"."""
        return (f"{self.title}{extra}   "
                f"{mmss(self.clock.t - self.clock.start)} / {mmss(self.clock.end - self.clock.start)}")

    def speed_reference(self) -> float:
        """The calibrated max_speed (p95) in data units per second, e.g. m/s."""
        return self.calibration.max_speed * 60.0 / self.speed / self.transform.scale

    def overlay(self, history: float = 10.0) -> list[tuple[Boid, str, list[tuple[float, float]]]]:
        """(agent, track id, path) per agent; the path is the agent's recorded keyframes of the last
        `history` seconds of data time plus its current position, in world px (like tracking output)."""
        t = self.clock.t
        out = []
        for boid, piece in zip(self.agents, self._pieces):
            lo = bisect.bisect_left(piece.t, t - history)
            hi = bisect.bisect_right(piece.t, t)
            path = list(zip(piece.x[lo:hi], piece.y[lo:hi]))
            path.append((boid.position.x, boid.position.y))
            out.append((boid, piece.track_id, path))
        return out

    def hud_lines(self, controls: RuntimeControls, descriptors: dict[str, float]) -> list[str]:
        """HUD text for replay mode (classic look, thermal data mode): no boid weights, no MIDI."""
        d = descriptors.get
        state = "end of data" if self.clock.finished else ("paused" if controls.paused else "playing")
        return [
            f"PLAUD-SONIFICATION | {self.scene} | real trajectories ({self.citation}) -> OSC -> Max",
            f"data time {mmss(self.clock.t - self.clock.start)} / {mmss(self.clock.end - self.clock.start)} | "
            f"{state} | speed x{self.speed:g} | loop {'on' if self.loop else 'off'} | SPACE pause | R restart",
            f"people on screen {len(self.agents)} (most at once in the file {self.max_population}) | "
            f"fish_count {d('fish_count', 0):.2f}{' (normalised)' if self.normaliser is not None else ''}",
            f"source: {self.reference}",
            f"{self.event_status()} | predator_amount {controls.predator_amount:.2f} (UDP) | no force on the data",
            f"OSC out: {OSC_HOST}:{OSC_PORT} | plain out: {PLAIN_UDP_PORT} | control in: {CONTROL_PORT}",
            f"mean_speed {d('mean_speed', 0):.2f} | energy {d('energy', 0):.2f} | density {d('density', 0):.2f} | "
            f"spread {d('spread', 0):.2f} | coherence {d('direction_coherence', 0):.2f}",
        ]

    def normalise(self, descriptors: dict[str, float]) -> dict[str, float]:
        """With --replay-normalise: the file's p5..p95 -> 0..1 (before the smoother); else unchanged."""
        if self.normaliser is None:
            return descriptors
        return self.normaliser(descriptors, len(self.agents))

    def _sync(self, t: float) -> None:
        active = self.trajectories.active(t)
        keys = {(p.track_id, p.index) for p in active}
        i = 0
        while i < len(self._pieces):  # leavers: the last agent moves into the freed index
            if (self._pieces[i].track_id, self._pieces[i].index) in keys:
                i += 1
                continue
            self.agents[i] = self.agents[-1]
            self._pieces[i] = self._pieces[-1]
            self.agents.pop()
            self._pieces.pop()
        present = {(p.track_id, p.index) for p in self._pieces}
        waiting = 0
        for piece in active:  # newcomers in start order, appended while there is room
            if (piece.track_id, piece.index) in present:
                continue
            if len(self.agents) >= self.max_agents:
                waiting += 1
                continue
            self.agents.append(replay_agent(piece, self.calibration.max_speed, self.calibration.max_force))
            self._pieces.append(piece)
        self.peak_waiting = max(self.peak_waiting, waiting)
        for boid, piece in zip(self.agents, self._pieces):
            state = per_frame(piece.state(t), self.speed)
            boid.position.update(state.x, state.y)
            boid.velocity.update(state.vx, state.vy)
            boid.acceleration.update(state.ax, state.ay)


def replay_agent(piece: Piece, max_speed: float = 4.0, max_force: float = 0.055) -> Boid:
    """A Boid for one track piece; size and colour from a hash of the id, no random draws."""
    h = zlib.crc32(piece.track_id.encode("utf-8"))
    return Boid(
        id=h,
        position=Vector2(piece.x[0], piece.y[0]),
        velocity=Vector2(0, 0),
        acceleration=Vector2(0, 0),
        size=6.0 + 5.0 * (h % 1000) / 999.0,
        color=REPLAY_COLORS[(h >> 10) % len(REPLAY_COLORS)],
        max_speed=max_speed,
        max_force=max_force,
    )
