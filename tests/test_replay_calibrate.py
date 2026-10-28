"""Offline tests for src/aquarium_boids/replay/calibrate.py and ReplaySource.normalise (no window).

Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquarium_boids.descriptors import compute_descriptors  # noqa: E402
from aquarium_boids.replay.calibrate import (  # noqa: E402
    Normaliser, calibrate_descriptors, calibrate_speed_force, percentile,
)
from aquarium_boids.replay.interp import Trajectories  # noqa: E402
from aquarium_boids.replay.loader import Dataset, Sample  # noqa: E402
from aquarium_boids.sources import ReplaySource, SimulationSource  # noqa: E402
from aquarium_boids.controls import RuntimeControls  # noqa: E402

DT = 0.4
WORLD = (1200.0, 750.0)


def dataset(tracks: dict[str, list[tuple[int, float, float]]]) -> Dataset:
    return Dataset({tid: tuple(Sample(f * DT, f, x, y) for f, x, y in rows) for tid, rows in tracks.items()}, fps=2.5)


def straight(vx: float, frames: range, y: float = 0.0):
    return [(f, vx * f * DT, y) for f in frames]


def circle(r: float, period: float, frames: range, cx: float = 600.0, cy: float = 375.0):
    w = 2 * math.pi / period
    return [(f, cx + r * math.cos(w * f * DT), cy + r * math.sin(w * f * DT)) for f in frames]


class Percentile(unittest.TestCase):
    def test_linear_interpolation(self):
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.5), 2.0)
        self.assertAlmostEqual(percentile(list(map(float, range(101))), 0.95), 95.0)
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(percentile([7.0], 0.95), 7.0)


class SpeedForce(unittest.TestCase):
    def test_constant_speed_gives_that_speed_and_the_force_fallback(self):
        tr = Trajectories(dataset({"a": straight(60.0, range(20)), "b": straight(60.0, range(20), 100.0)}))
        cal = calibrate_speed_force(tr, speed=1.0, max_agents=100)
        self.assertAlmostEqual(cal.max_speed, 1.0)  # 60 px/s = 1 px per frame
        self.assertEqual(cal.max_force, 0.055)  # no acceleration anywhere: Boid default kept
        self.assertGreater(cal.samples, 0)

    def test_circle_force_and_speed_invariance(self):
        r, period = 200.0, 10.0
        tr = Trajectories(dataset({"c": circle(r, period, range(60))}))
        w = 2 * math.pi / period
        cal = calibrate_speed_force(tr, speed=1.0, max_agents=100)
        self.assertAlmostEqual(cal.max_speed, w * r / 60, delta=0.02 * w * r / 60)
        self.assertAlmostEqual(cal.max_force * 6, w * w * r / 3600, delta=0.08 * w * w * r / 3600)
        fast = calibrate_speed_force(tr, speed=3.0, max_agents=100)
        self.assertAlmostEqual(fast.max_speed, 3 * cal.max_speed)  # v ~ speed
        self.assertAlmostEqual(fast.max_force, 9 * cal.max_force)  # a ~ speed^2

    def test_cap_limits_the_sampled_agents(self):
        tr = Trajectories(dataset({"slow": straight(60.0, range(10)), "fast": straight(600.0, range(10), 50.0)}))
        self.assertAlmostEqual(calibrate_speed_force(tr, 1.0, max_agents=1).max_speed, 1.0)


class NormaliserMap(unittest.TestCase):
    def setUp(self):
        self.n = Normaliser(lo={"spread": 0.2, "flat": 0.5}, hi={"spread": 0.6, "flat": 0.5}, max_count=20)

    def test_maps_clips_and_counts(self):
        out = self.n({"spread": 0.4, "fish_count": 0.05, "flat": 0.7, "other": 0.3}, count=5)
        self.assertAlmostEqual(out["spread"], 0.5)
        self.assertAlmostEqual(out["fish_count"], 5 / 20)
        self.assertEqual(out["flat"], 0.7)  # no spread in the data: raw value kept
        self.assertEqual(out["other"], 0.3)  # not calibrated: raw value kept
        self.assertEqual(self.n({"spread": 0.1}, 3)["spread"], 0.0)
        self.assertEqual(self.n({"spread": 0.9}, 3)["spread"], 1.0)
        self.assertEqual(self.n({"fish_count": 1.0}, 40)["fish_count"], 1.0)

    def test_empty_world_is_untouched(self):
        d = {"spread": 0.0, "fish_count": 0.0}
        self.assertIs(self.n(d, 0), d)


class Descriptors(unittest.TestCase):
    def test_percentiles_of_the_raw_series(self):
        # one walker the whole time, a second one for the middle third: count 1..2, spread varies
        tr = Trajectories(dataset({"a": straight(30.0, range(30), 300.0), "b": straight(30.0, range(10, 20), 400.0)}))
        cal = calibrate_speed_force(tr, 1.0, 100)
        norm = calibrate_descriptors(tr, 1.0, 100, cal, WORLD, step=0.4)
        self.assertEqual(norm.max_count, 2)
        self.assertIn("spread", norm.lo)
        self.assertLessEqual(norm.lo["spread"], norm.hi["spread"])
        self.assertAlmostEqual(norm.lo["mean_speed"], norm.hi["mean_speed"])  # constant speed

    def test_simulation_normalise_is_the_identity(self):
        d = {"spread": 0.3}
        self.assertIs(SimulationSource.normalise(object.__new__(SimulationSource), d), d)


class SourceIntegration(unittest.TestCase):
    def csv(self, text: str) -> Path:
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
        f.write(text)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        return Path(f.name)

    def rows(self):
        out = ["id,frame,x,y"]
        for i in range(6):  # six walkers on circles of different radii, entering one after another
            for k in range(i * 5, i * 5 + 40):
                a = 2 * math.pi * k * DT / (6 + i)
                out.append(f"{i},{k},{(3 + i) * math.cos(a):.4f},{(3 + i) * math.sin(a):.4f}")
        return "\n".join(out) + "\n"

    def test_calibrated_agents_and_normalised_range(self):
        src = ReplaySource(self.csv(self.rows()), normalise=True)
        self.assertTrue(all(b.max_speed == src.calibration.max_speed for b in src.agents))
        self.assertTrue(all(b.max_force == src.calibration.max_force for b in src.agents))
        c = RuntimeControls()
        seen: dict[str, list[float]] = {}
        for _ in range(int(60 * 30 * DT)):
            agents = src.step(1.0, c, None, None)
            for k, v in src.normalise(compute_descriptors(agents, *WORLD)).items():
                seen.setdefault(k, []).append(v)
        for key in ("mean_speed", "energy", "spread", "direction_coherence", "fish_count"):
            values = seen[key]
            self.assertGreaterEqual(min(values), 0.0, key)
            self.assertLessEqual(max(values), 1.0, key)
            self.assertGreater(max(values) - min(values), 0.5, key)  # uses most of 0..1

    def test_without_flag_descriptors_pass_through(self):
        src = ReplaySource(self.csv(self.rows()))
        d = {"spread": 0.3}
        self.assertIs(src.normalise(d), d)
        self.assertIsNone(src.normaliser)


if __name__ == "__main__":
    unittest.main()
