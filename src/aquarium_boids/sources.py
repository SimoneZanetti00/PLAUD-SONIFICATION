"""Where the fish come from: a source owns the agent list and moves it one frame at a time.

main.py only calls `reset`, `step` and `normalise`, then hands `agents` to the descriptors, OSC
and the renderer. SimulationSource is the boid simulation, the code that used to run inline in main.py.
ReplaySource moves the same Boid objects along recorded trajectories (--replay FILE).
"""
from __future__ import annotations

import zlib
from pathlib import Path

from pygame.math import Vector2

from .boids import Boid
from .config import BOID_COUNT, HEIGHT, WIDTH
from .controls import RuntimeControls
from .replay.calibrate import Normaliser, calibrate_descriptors, calibrate_speed_force
from .replay.interp import DataClock, Piece, Trajectories, per_frame
from .replay.loader import data_bounds, fit_to_world, load_trajectories, to_world

REPLAY_MARGIN = 60.0  # world px kept free around the data (same as the joystick margin)
REPLAY_COLORS = [(80, 210, 255), (105, 255, 190), (255, 190, 95), (220, 160, 255)]  # Boid.create's palette


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
        self.calibration = calibrate_speed_force(self.trajectories, speed, max_agents)
        self.normaliser: Normaliser | None = (
            calibrate_descriptors(self.trajectories, speed, max_agents, self.calibration, world) if normalise else None
        )
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
            f"normalise {'on' if normalise else 'off'}"
        )
        self._sync(self.clock.t)

    def reset(self, population: int) -> None:
        """R key: restart the file (population is ignored)."""
        self.clock.reset()
        self.agents = []
        self._pieces = []
        self._end_reported = False
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
            self._sync(t)
            for boid in self.agents:  # Boid.update's flash decay; update() itself is never called
                boid.flash = max(0.0, boid.flash - 0.045 * dt)
        if self.clock.finished and not self._end_reported:
            print("[replay] end of data, holding the last frame (R restarts)")
            self._end_reported = True
        controls.population = len(self.agents)
        return self.agents

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
