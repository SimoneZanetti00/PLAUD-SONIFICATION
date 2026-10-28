"""Where the fish come from: a source owns the agent list and moves it one frame at a time.

main.py only calls `reset` and `step`, then hands `agents` to the descriptors, OSC and the
renderer. SimulationSource is the boid simulation, the code that used to run inline in main.py.
"""
from __future__ import annotations

from pygame.math import Vector2

from .boids import Boid
from .config import BOID_COUNT, HEIGHT, WIDTH
from .controls import RuntimeControls


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
