"""Offline tests for the ant sound mapping (replay/ant_mapping.py), the position smoothing
(loader.smooth_dataset) and ReplaySource with an ant config in profile scene.

Synthetic data only. Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.replay import ant_mapping as am  # noqa: E402
from aquarium_boids.replay.ant_mapping import (  # noqa: E402
    Ant, AntCalibration, AntMapping, EncounterCounter, FoodPlace, calibrate_ants, rate_per_min, scene_step,
)
from aquarium_boids.replay.loader import Dataset, Sample, smooth_dataset  # noqa: E402
from aquarium_boids.replay.mapping import SceneMapping  # noqa: E402
from aquarium_boids.replay.scene_config import SceneConfig, load_scene_config  # noqa: E402
from aquarium_boids.sources import ReplaySource  # noqa: E402


def config(**kw) -> SceneConfig:
    base = dict(path=Path("c.json"), body_length=1.0, events={"arrivals": "nest", "food_discovery": True,
                                                              "recruitment": True, "encounters": True})
    base.update(kw)
    return SceneConfig(**base)


def calibration(**kw) -> AntCalibration:
    base = dict(step_s=0.5, count_range=(1.0, 10.0), distance_bl=(0.0, 100.0), outflow_p90=4.0, encounter_p95=6.0,
                nest_starts=())
    base.update(kw)
    return AntCalibration(**base)


def ant(key, x=50.0, y=0.0, vx=0.0, vy=0.0, pan=0.5, from_nest=False) -> Ant:
    return Ant(key, x, y, vx, vy, pan, x, y, True, from_nest)


def run(m: AntMapping, t0: float, t1: float, ants_at, dt: float = 0.25):
    """Step the mapping from t0 to t1 in data time; ants_at(t) -> list of Ant."""
    t = t0
    m.update(t, 0.0, ants_at(t))
    while t < t1 - 1e-9:
        t += dt
        m.update(t, dt, ants_at(t), pan_of=lambda wx, wy: wx / 100.0)


class Smoothing(unittest.TestCase):
    def dataset(self, xs, frames=None):
        frames = frames or list(range(len(xs)))
        return Dataset({"a": tuple(Sample(f / 30.0, f, x, 0.0) for f, x in zip(frames, xs))}, fps=30.0)

    def test_zigzag_is_flattened_and_a_line_kept(self):
        zig = smooth_dataset(self.dataset([(-1) ** i * 2.0 for i in range(30)]), 0.2)
        inner = [s.x for s in zig.tracks["a"][3:-3]]
        self.assertTrue(all(abs(x) < 0.6 for x in inner))  # 7-sample window at 30 fps
        line = smooth_dataset(self.dataset([float(i) for i in range(30)]), 0.2)
        self.assertEqual([s.x for s in line.tracks["a"]], [float(i) for i in range(30)])  # centred: no lag
        self.assertEqual([s.t for s in line.tracks["a"]], [i / 30.0 for i in range(30)])

    def test_runs_split_at_missing_rows_and_zero_window(self):
        d = self.dataset([0.0, 0.0, 0.0, 9.0, 9.0, 9.0], frames=[0, 1, 2, 10, 11, 12])
        out = smooth_dataset(d, 0.2)
        self.assertEqual([s.x for s in out.tracks["a"]], [0.0, 0.0, 0.0, 9.0, 9.0, 9.0])
        self.assertIs(smooth_dataset(d, 0.0), d)


class Pieces(unittest.TestCase):
    def test_rate_and_step(self):
        self.assertEqual(rate_per_min([1.0, 50.0, 100.0, 130.0], 130.0, 120.0), 1.5)  # 50, 100, 130 in (10, 130]
        self.assertEqual(scene_step(100.0), 0.1)
        self.assertEqual(scene_step(10504.0), 0.5)

    def test_encounters_count_once_until_apart(self):
        c = EncounterCounter(1.0)
        self.assertEqual(c.update([("a", 0, 0), ("b", 5, 0)]), [])
        self.assertEqual(c.update([("a", 0, 0), ("b", 0.5, 0)]), [("a", "b")])
        self.assertEqual(c.update([("a", 0, 0), ("b", 0.4, 0)]), [])  # still in contact
        self.assertEqual(c.update([("a", 0, 0), ("b", 1.5, 0)]), [])  # not yet apart (2 BL)
        self.assertEqual(c.update([("a", 0, 0), ("b", 0.5, 0)]), [])
        self.assertEqual(c.update([("a", 0, 0), ("b", 3.0, 0)]), [])  # apart
        self.assertEqual(c.update([("b", 0.9, 0), ("a", 0, 0)]), [("a", "b")])  # again

    def test_calibration(self):
        frames = [(t * 0.5, [("a", 10.0 + t, 0.0), ("b", 10.0 + t, 0.5)][: 1 + t % 2]) for t in range(400)]
        cal = calibrate_ants(frames, 0.5, (0.0, 0.0), [10.0, 11.0, 12.0], 1.0)
        self.assertEqual(cal.count_range, (1.0, 3.0))  # p5 1, p95 2 -> at least two more
        self.assertGreater(cal.distance_bl[1], cal.distance_bl[0])
        self.assertEqual(cal.outflow_p90, 1.5)  # 3 starts in a 2-minute window
        self.assertGreater(cal.encounter_p95, 0)  # b joins a at 0.5 BL every other frame


class Mapping(unittest.TestCase):
    def test_food_discovery_slices_kick_on_first_of_patch_and_flash(self):
        food = (FoodPlace(10.0, 20.0, 0.0, "GP1"), FoodPlace(12.0, 80.0, 0.0, "GP1"), FoodPlace(14.0, 90.0, 0.0, "GP2"))
        m = AntMapping(config(), calibration(), max_count=10, food=food)
        run(m, 0.0, 9.0, lambda t: [ant("a")])
        self.assertEqual((m.pad_samples, m.kicks), ([], 0))
        run(m, 9.0, 20.0, lambda t: [ant("a")])
        self.assertEqual(m.pad_samples, [1, 4, 4])  # by where the item is: 0.2, 0.8, 0.9
        self.assertEqual(m.kicks, 2)  # first of GP1, first of GP2
        self.assertEqual([(k, xy) for k, _, xy in m.place_events], [("food", (20.0, 0.0)), ("food", (80.0, 0.0)),
                                                                    ("food", (90.0, 0.0))])

    def test_food_found_before_the_start_is_not_news_and_reset(self):
        food = (FoodPlace(10.0, 20.0, 0.0, "GP1"),)
        m = AntMapping(config(), calibration(), max_count=10, food=food)
        run(m, 30.0, 40.0, lambda t: [ant("a")])
        self.assertEqual((m.pad_samples, m.food_sent), ([], 0))
        m.reset()
        run(m, 0.0, 12.0, lambda t: [ant("a")])
        self.assertEqual(m.food_sent, 1)

    def test_recruitment_wave_only_after_discovery(self):
        starts = tuple(50.0 + i * 2.0 for i in range(60))  # 30 per minute from t = 50 s: far above p90 = 4
        cal = calibration(nest_starts=starts)
        m = AntMapping(config(), cal, max_count=10, food=(FoodPlace(100.0, 0.0, 0.0, "GP1"),), nest_world=(5.0, 5.0))
        run(m, 0.0, 99.0, lambda t: [ant("a")])
        self.assertEqual((m.wave, m.waves), (0.0, 0))  # outflow high, but no food found yet
        loud_before = m.loudness
        run(m, 99.0, 130.0, lambda t: [ant("a")])
        self.assertGreater(m.wave, 0.9)
        self.assertEqual(m.waves, 1)
        self.assertIn(("recruitment", (5.0, 5.0)), [(k, xy) for k, _, xy in m.place_events])
        self.assertGreater(m.loudness, loud_before + 0.2)
        self.assertGreater(m.target[0], 0.4)  # onset target swells
        no = AntMapping(config(events={"recruitment": False}), cal, max_count=10, food=(FoodPlace(100.0, 0, 0, "G"),))
        run(no, 0.0, 130.0, lambda t: [ant("a")])
        self.assertEqual(no.wave, 0.0)

    def test_arrivals_from_the_nest_only_and_spaced_in_playback_time(self):
        m = AntMapping(config(), calibration(), max_count=10, speed=4.0)
        seen = []

        def ants_at(t):
            n = int(t)  # a new ant every data second, every other one from the nest
            seen.append(n)
            return [ant(f"k{i}", pan=0.1, from_nest=i % 2 == 0) for i in range(n + 1)]

        run(m, 0.0, 64.0, ants_at)
        self.assertEqual(m.arrivals_seen, 32)  # k2, k4, ... k64 (k0 was there at the start)
        self.assertEqual(m.arrivals_sent, 2)  # 64 data s = 16 playback s at 4x: one per 8 s
        self.assertEqual(m.pad_samples, [1, 1])
        self.assertTrue(all(kind == "arrival" for kind, _, _ in m.events))

    def test_activity_distance_and_encounter_density(self):
        cal = calibration(distance_bl=(10.0, 110.0), encounter_p95=60.0)
        m = AntMapping(config(), cal, max_count=10, nest_bl=(0.0, 0.0))

        def ants_at(t):  # 2 movers at 2.5 BL/s, 2 standing; pairs b-c touch and part every second
            gap = 0.5 if int(t * 4) % 4 == 0 else 3.0
            return [ant("a", 60, 0, 2.5, 0), ant("b", 60, 10, 0, 2.5), ant("c", 60, 10 + gap), ant("d", 60, 40)]

        run(m, 0.0, 60.0, ants_at)
        self.assertAlmostEqual(m.moving, 0.5, places=2)
        self.assertAlmostEqual(m.centroid, 0.5, places=2)  # (2.5 - 1) / (4 - 1)
        self.assertAlmostEqual(m.distance01, (sum(math.hypot(60, y) for y in (0, 10, 10, 40)) / 4 - 10) / 100, places=1)
        self.assertGreater(m.encounter_rate, 30)
        self.assertGreater(m.density, 0.5 * am.DENSITY_MAX)
        self.assertAlmostEqual(m.target[1], m.moving, places=2)  # taken at the last scene check
        self.assertAlmostEqual(m.target[2], m.distance01, places=2)


def write(path: Path, data) -> Path:
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


class Source(unittest.TestCase):
    def make(self, d: Path, profile: str = "scene", **cfg):
        rows = ["id,frame,t,x,y"]
        for i in range(6):  # ants leave the nest at (0, 0) one by one and walk out along x
            for f in range(40):
                rows.append(f"a{i},{f + 10 * i},{(f + 10 * i) * 0.5},{f * 6.0},{i * 3.0}")
        csv_path = write(d / "t.csv", "\n".join(rows) + "\n")
        places = {"nest": {"x": 0, "y": 0}, "food": [{"x": 120, "y": 0, "patch": "GP1", "t": 30.0}]}
        write(d / "t.places.json", places)
        c = {"units": "mm", "body_length": 6, "places_file": "t.places.json", "places": {"nest": {"r": 20}},
             "events": {"arrivals": "nest", "food_discovery": True, "recruitment": True, "encounters": True}, **cfg}
        return ReplaySource(csv_path, speed=2.0, events=False, profile=profile,
                            config=load_scene_config(write(d / "t.scene.json", c)))

    def test_profiles(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.make(Path(d))
            self.assertIsInstance(src.mapping, AntMapping)
            self.assertEqual(len(src.mapping.calibration.nest_starts), 6)
            self.assertIsNone(self.make(Path(d), profile="synthfish").mapping)
            c = RuntimeControls()
            sent = []
            while not src.clock.finished:
                src.step(1.0, c, None, None)
                sent += src.take_pad_samples()
                marks = src.place_marks()
                if marks:
                    kind, age, xy = marks[0]
                    self.assertEqual(kind, "food")
                    self.assertLessEqual(age, 1.2)
            self.assertEqual(src.mapping.food_sent, 1)
            self.assertGreaterEqual(len(sent), 2)  # the food slice and at least one arrival
            self.assertEqual(set(src.mapping_values()), {"xy", "density", "master", "temp", "loudness", "limit",
                                                         "centroid", "stretch"})
            self.assertTrue(any(line.startswith("sound ants") for line in src.hud_lines(c, {})))

    def test_pedestrian_scene_unchanged_without_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(Path(d) / "w.csv", "id,frame,x,y\n" + "".join(f"a,{f},{f * 0.5},1\n" for f in range(20)))
            self.assertIsInstance(ReplaySource(p, events=False, profile="scene").mapping, SceneMapping)

    def test_path_seconds_and_smoothing_from_config(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.make(Path(d), path_s=2, smooth_s=0.2)
            self.assertEqual(src.path_s, 2.0)
            self.assertIsNone(self.make(Path(d)).path_s)


if __name__ == "__main__":
    unittest.main()
