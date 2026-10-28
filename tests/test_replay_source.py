"""Offline tests for ReplaySource (src/aquarium_boids/sources.py): no window, no sockets.

Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.sources import ReplaySource  # noqa: E402

FPS = 2.5
ROW_S = 1 / FPS
FRAMES_PER_ROW = 60 * ROW_S  # simulation frames per data row at speed 1 (24)

_TEMP: list[Path] = []


def tearDownModule():
    for p in _TEMP:
        p.unlink(missing_ok=True)


def csv_file(rows: list[tuple[str, int, float, float]]) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    f.write("id,frame,x,y\n" + "".join(f"{i},{fr},{x},{y}\n" for i, fr, x, y in rows))
    f.close()
    _TEMP.append(Path(f.name))
    return Path(f.name)


def walker(track_id: str, first: int, last: int, y: float, vx: float = 1.0):
    return [(track_id, k, vx * k, y) for k in range(first, last + 1)]


def run(src: ReplaySource, controls: RuntimeControls, frames: float, dt: float = 1.0):
    for _ in range(int(frames)):
        src.step(dt, controls, None, None)


def ids(src: ReplaySource) -> list[str]:
    return [p.track_id for p in src._pieces]


class Slots(unittest.TestCase):
    def setUp(self):
        # a: rows 0-10, b: 0-3 (leaves early), c: 0-10, d: 5-10 (enters late); a corner pins the bounds
        rows = walker("a", 0, 10, 0.0) + walker("b", 0, 3, 1.0) + walker("c", 0, 10, 2.0) + walker("d", 5, 10, 3.0)
        self.src = ReplaySource(csv_file(rows), fps=FPS)
        self.c = RuntimeControls()

    def test_start_order_then_swap_remove_then_append(self):
        self.assertEqual(ids(self.src), ["a", "b", "c"])
        agent_c = self.src.agents[2]
        run(self.src, self.c, 4 * FRAMES_PER_ROW)  # b has left
        self.assertEqual(ids(self.src), ["a", "c"])  # c took b's index (it was the last one)
        self.assertIs(self.src.agents[1], agent_c)  # same Boid object, not a new one
        run(self.src, self.c, 2 * FRAMES_PER_ROW)  # d has entered
        self.assertEqual(ids(self.src), ["a", "c", "d"])
        self.assertEqual(self.c.population, 3)

    def test_cap_keeps_newcomers_waiting(self):
        rows = [(f"p{i}", k, k, i) for i in range(5) for k in range(4)]
        src = ReplaySource(csv_file(rows), fps=FPS, max_agents=3)
        self.assertEqual(ids(src), ["p0", "p1", "p2"])
        self.assertEqual(src.peak_waiting, 2)

    def test_reset_restarts_the_file(self):
        run(self.src, self.c, 6 * FRAMES_PER_ROW)
        self.src.reset(self.c.population)
        self.assertEqual(ids(self.src), ["a", "b", "c"])
        self.assertAlmostEqual(self.src.clock.t, 0.0)


class Motion(unittest.TestCase):
    def setUp(self):
        rows = walker("a", 0, 20, 0.0, vx=1.0) + [("pin", 0, 0.0, 10.0), ("pin", 20, 20.0, 10.0)]
        self.src = ReplaySource(csv_file(rows), fps=FPS)
        self.c = RuntimeControls()

    def test_velocity_is_the_per_frame_displacement(self):
        run(self.src, self.c, 30)
        a = self.src.agents[0]
        x0 = a.position.x
        self.src.step(1.0, self.c, None, None)
        self.assertAlmostEqual(a.position.x - x0, a.velocity.x, places=6)
        self.assertAlmostEqual(a.acceleration.x, 0.0, places=9)
        self.assertEqual(a.max_speed, self.src.calibration.max_speed)  # calibrated, not the Boid default
        self.assertNotEqual(a.max_speed, 4.0)

    def test_speed_factor(self):
        fast = ReplaySource(csv_file(walker("a", 0, 20, 0.0) + [("pin", 0, 0.0, 10.0), ("pin", 20, 20.0, 10.0)]),
                            fps=FPS, speed=2.0)
        run(self.src, self.c, 30)
        run(fast, self.c, 30)
        self.assertAlmostEqual(fast.clock.t, 2 * self.src.clock.t)
        self.assertAlmostEqual(fast.agents[0].velocity.x, 2 * self.src.agents[0].velocity.x)

    def test_pause_holds_everything(self):
        run(self.src, self.c, 30)
        a = self.src.agents[0]
        before = (self.src.clock.t, a.position.x, a.velocity.x)
        self.c.paused = True
        run(self.src, self.c, 50)
        self.assertEqual((self.src.clock.t, a.position.x, a.velocity.x), before)

    def test_flash_decays_like_boid_update(self):
        a = self.src.agents[0]
        a.flash = 1.0
        run(self.src, self.c, 10)
        self.assertAlmostEqual(a.flash, 1.0 - 0.45)

    def test_end_holds_the_last_frame_and_loop_wraps(self):
        run(self.src, self.c, 21 * FRAMES_PER_ROW)
        self.assertTrue(self.src.clock.finished)
        x_end = self.src.agents[0].position.x
        run(self.src, self.c, 10)
        self.assertEqual(self.src.agents[0].position.x, x_end)
        looped = ReplaySource(csv_file(walker("a", 0, 20, 0.0) + [("pin", 0, 0.0, 10.0), ("pin", 20, 20.0, 10.0)]),
                              fps=FPS, loop=True)
        run(looped, self.c, 21 * FRAMES_PER_ROW)
        self.assertFalse(looped.clock.finished)
        self.assertLess(looped.clock.t, 2 * ROW_S)

    def test_flip_y(self):
        rows = [("up", 0, 0.0, 0.0), ("up", 1, 0.0, 10.0)]  # walks towards +y
        down = ReplaySource(csv_file(rows), fps=FPS)
        up = ReplaySource(csv_file(rows), fps=FPS, flip_y=True)
        run(down, self.c, 5)
        run(up, self.c, 5)
        self.assertGreater(down.agents[0].velocity.y, 0)  # screen y down
        self.assertLess(up.agents[0].velocity.y, 0)  # screen up

    def test_inside_the_world_margin(self):
        for _ in range(int(20 * FRAMES_PER_ROW)):
            self.src.step(1.0, self.c, None, None)
            for b in self.src.agents:
                self.assertTrue(60 - 1e-6 <= b.position.x <= 1140 + 1e-6)
                self.assertTrue(60 - 1e-6 <= b.position.y <= 690 + 1e-6)


class NoRandom(unittest.TestCase):
    def test_no_random_draws_and_deterministic_look(self):
        rows = [(f"p{i}", k, k + i, i) for i in range(6) for k in range(6)]
        random.seed(123)
        state = random.getstate()
        a = ReplaySource(csv_file(rows), fps=FPS)
        run(a, RuntimeControls(), 100)
        self.assertEqual(random.getstate(), state)
        b = ReplaySource(csv_file(rows), fps=FPS)
        self.assertEqual([(x.size, x.color) for x in a.agents], [(x.size, x.color) for x in b.agents])
        self.assertTrue(all(6.0 <= x.size <= 11.0 for x in a.agents))


class Errors(unittest.TestCase):
    def test_bad_arguments(self):
        p = csv_file(walker("a", 0, 3, 0.0))
        with self.assertRaises(ValueError):
            ReplaySource(p, max_agents=0)
        with self.assertRaises(ValueError):
            ReplaySource(p, speed=0.0)
        with self.assertRaises(OSError):
            ReplaySource(p.with_name("does_not_exist.csv"))


if __name__ == "__main__":
    unittest.main()
