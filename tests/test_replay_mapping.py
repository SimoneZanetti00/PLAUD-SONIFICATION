"""Offline tests for the replay sound profile `scene` (replay/mapping.py, ReplaySource, osc_io overrides).

Synthetic terrain, tracks and homography only (no model or dataset files). No window, no sockets.
Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.osc_io import OscSender  # noqa: E402
from aquarium_boids.replay.camera import Camera  # noqa: E402
from aquarium_boids.replay.mapping import (  # noqa: E402
    EVENT_MIN_GAP_S, MASTER_RANGE_DB, MASTER_TOP_DB, PAIR_STEP_S, TERRITORY_RANGE, Person, SceneMapping, Terrain,
    cadence_density, db_to_master, master_level, together_rule,
)
from aquarium_boids.sources import EVENT_FLASH_S, ReplaySource  # noqa: E402


def synthetic_terrain() -> Path:
    """rhythm (rhythmicity, onset_density) high on the left, loudness high at the bottom (as RafaelV2)."""
    n = 128
    rows_rhythm = [[1.0 - j / (n - 1) for j in range(n)] for _ in range(n)]
    rows_loud = [[1.0 - i / (n - 1)] * n for i in range(n)]  # row 0 = bottom = loud
    flat = [[0.5] * n for _ in range(n)]
    d = {"resolution": n, "orientation": "row0_is_y0_bottom",
         "features": {"rhythmicity": rows_rhythm, "onset_density": rows_rhythm, "loudness": rows_loud,
                      "percussiveness": flat, "brightness": flat, "noisiness": flat}}
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(d, f)
    f.close()
    return Path(f.name)


def person(key, x, y, vx, vy=0.0, pan=0.5, visible=True):
    return Person(key, x, y, vx, vy, pan, x * 50, y * 50, visible)


class Curves(unittest.TestCase):
    def test_master_level_is_a_quiet_steady_bed(self):
        self.assertEqual(master_level(0, 20), 0.0)
        self.assertAlmostEqual(master_level(20, 20), db_to_master(MASTER_TOP_DB))
        self.assertAlmostEqual(master_level(20, 20), 1 - 10 / 70)
        levels = [master_level(n, 20) for n in range(1, 21)]
        self.assertEqual(levels, sorted(levels))  # more people, louder
        self.assertAlmostEqual(70 * (levels[-1] - levels[0]), MASTER_RANGE_DB * (1 - 0.228), delta=0.1)  # ~3 dB
        self.assertEqual(master_level(40, 20), master_level(20, 20))

    def test_cadence_density(self):
        self.assertEqual(cadence_density(0.0), 0.0)
        rate = 1.3 / 0.7  # walking at 1.3 m/s
        self.assertAlmostEqual(0.5 + 0.075 * cadence_density(rate), rate)  # shoal_gran's base onset rate
        self.assertEqual(cadence_density(50.0), 100.0)

    def test_together_rule(self):
        self.assertTrue(together_rule(1.0, (1.2, 0.0), (1.25, 0.05)))
        self.assertFalse(together_rule(2.5, (1.2, 0.0), (1.2, 0.0)))
        self.assertFalse(together_rule(1.0, (1.2, 0.0), (-1.2, 0.0)))


class TerrainLookup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = synthetic_terrain()
        cls.terrain = Terrain.load(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)

    def test_hurried_left_crowded_down(self):
        x, y = self.terrain.point(hurry=1.0, crowding=1.0)
        self.assertLess(x, 0.15)
        self.assertLess(y, 0.15)
        x, y = self.terrain.point(hurry=0.0, crowding=0.0)
        self.assertGreater(x, 0.85)
        self.assertGreater(y, 0.85)

    def test_territory_stays_central_and_glides(self):
        m = SceneMapping(max_count=10, terrain=self.terrain)
        hurried = [person(str(i), i, 0, 1.4) for i in range(10)]
        m.update(0.0, 0.0, hurried, 1.0)
        lo = 0.5 - TERRITORY_RANGE / 2
        self.assertGreaterEqual(min(m.territory), lo - 1e-9)  # never beyond the central 60 %
        start = m.territory
        m.update(0.0, 0.0, [], 0.0)
        m2 = SceneMapping(max_count=10, terrain=self.terrain)
        m2.update(0.0, 0.0, hurried[:1], 0.0)  # calm, sparse
        t = 0.0
        for _ in range(60 * 5):  # 5 s later, now hurried and crowded: the bed has moved only part of the way
            t += 1 / 60
            m2.update(t, 1 / 60, hurried, 1.0)
        moved = abs(m2.territory[0] - 0.5 - TERRITORY_RANGE * (self.terrain.point(m2.hurry, m2.crowding)[0] - 0.5))
        self.assertGreater(moved, 0.05)
        self.assertNotEqual(start, (0.5, 0.5))

    def test_bad_orientation(self):
        d = json.loads(self.path.read_text(encoding="utf-8"))
        d["orientation"] = "row0_is_y0_top"
        bad = Path(tempfile.mkdtemp()) / "bad.json"
        bad.write_text(json.dumps(d), encoding="utf-8")
        with self.assertRaises(ValueError):
            Terrain.load(bad)


class Mapping(unittest.TestCase):
    def test_presence_fades_and_silence(self):
        m = SceneMapping(max_count=10)
        walkers = [person("a", 0, 0, 1.2), person("b", 0, 3, 1.2)]
        m.update(0.0, 0.0, walkers, 0.5)
        self.assertAlmostEqual(m.master, master_level(2, 10))
        t = 0.0
        for _ in range(int(60 * 8)):
            t += 1 / 60
            m.update(t, 1 / 60, [], 0.0)
        self.assertEqual(m.master, 0.0)
        m.update(t + 1 / 60, 1 / 60, walkers[:1], 0.5)
        self.assertGreater(m.master, 0.0)
        self.assertLess(m.master, master_level(1, 10) / 2)

    def test_cadence(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.3), person("b", 5, 5, 0.1)], 0.5)
        self.assertAlmostEqual(m.step_rate, 1.3 / 0.7)
        m.update(0.1, 0.1, [person("b", 5, 5, 0.1)], 0.0)
        self.assertEqual((m.step_rate, m.density), (0.0, 0.0))

    def test_agitation_is_averaged_for_l2(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.2)], 0.5, agitation_now=0.0)
        m.update(1.0, 1.0, [person("a", 0, 0, 1.2)], 0.5, agitation_now=1.0)  # a spike: L2 barely moves
        self.assertLess(m.temp, 0.4)

    def test_arrival_waits_until_visible(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.2)], 0.5)
        m.update(1.0, 1.0, [person("a", 0, 0, 1.2), person("b", 9, 0, 1.2, pan=0.97, visible=False)], 0.5)
        self.assertEqual(m.pad_samples, [])  # entering through the edge band: not yet
        m.update(2.0, 1.0, [person("a", 0, 0, 1.2), person("b", 8, 0, 1.2, pan=0.9, visible=True)], 0.5)
        self.assertEqual(m.pad_samples, [4])  # now clearly in the picture, on the right: slice 4
        self.assertEqual(m.events[-1][:1], ("arrival",))
        m.update(3.0, 1.0, [person("a", 0, 0, 1.2), person("c", 1, 1, 1.2, pan=0.1)], 0.5)
        self.assertEqual(m.pad_samples, [4])  # within EVENT_MIN_GAP_S: capped
        self.assertEqual((m.arrivals_seen, m.arrivals_sent), (2, 1))
        m.update(2.0 + EVENT_MIN_GAP_S + 0.1, 0.1, [person("d", 0, 0, 1.2, pan=0.0)], 0.5)
        self.assertEqual(m.pad_samples, [4, 1])

    def test_never_seen_newcomer_makes_no_sound(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.2)], 0.5)
        for k in range(5):  # walks along the edge band and leaves
            m.update(1.0 + k, 1.0, [person("a", 0, 0, 1.2), person("e", 9, k, 1.2, visible=False)], 0.5)
        m.update(7.0, 1.0, [person("a", 0, 0, 1.2)], 0.5)
        self.assertEqual((m.pad_samples, m.arrivals_seen), ([], 0))

    def test_walking_together_kick(self):
        m = SceneMapping(max_count=10)
        t = 0.0
        pair = [person("a", 0, 0, 1.25), person("b", 0, 1, 1.2)]
        m.update(t, 0.0, pair, 0.5)
        while not m.kicks and t < 5:
            t += 1 / 60
            m.update(t, 1 / 60, pair, 0.5)
        self.assertAlmostEqual(t, 2.0, delta=PAIR_STEP_S + 0.05)  # after 2 s of walking together (10 Hz check)
        self.assertEqual(m.kicks, 1)
        self.assertEqual(m.events[-1][0], "together")
        self.assertEqual(set(m.events[-1][2]), {"a", "b"})

    def test_pair_in_the_edge_band_is_silent(self):
        m = SceneMapping(max_count=10)
        pair = [person("a", 0, 0, 1.25), person("b", 0, 1, 1.2, visible=False)]
        t = 0.0
        m.update(t, 0.0, pair, 0.5)
        for _ in range(60 * 4):
            t += 1 / 60
            m.update(t, 1 / 60, pair, 0.5)
        self.assertEqual((m.kicks, m.together_seen), (0, 0))

    def test_pause_holds(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.2)], 0.9)
        state = (m.hurry, m.territory)
        for _ in range(100):
            m.update(0.0, 0.0, [], 0.0)
        self.assertEqual((m.hurry, m.territory), state)


class SourceAndOsc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = "".join(f"{10 * k}\t{i}\t{0.5 * k}\t{10.0 * (i - 1)}\n" for i in (1, 2) for k in range(60))  # y 0 and 10 m
        rows += "".join(f"{10 * k}\t3\t{30 - 0.5 * (k - 20)}\t5\n" for k in range(20, 60))  # arrives at 8 s
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", newline="")
        f.write(rows)
        f.close()
        cls.path = Path(f.name)

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)

    def test_synthfish_sends_the_standard_mapping(self):
        src = ReplaySource(self.path)
        self.assertEqual(src.profile, "synthfish")
        self.assertIsNone(src.mapping_values())
        self.assertIsNone(src.fish_x_values())
        self.assertEqual((src.take_pad_samples(), src.take_kicks(), src.event_marks()), ([], 0, []))

    def test_scene_values_arrival_and_mark(self):
        src = ReplaySource(self.path, profile="scene", events=False)
        c = RuntimeControls()
        samples, marks = [], []
        for _ in range(60 * 12):
            src.step(1.0, c, None, None)
            samples += src.take_pad_samples()
            marks += [m for m in src.event_marks() if m[0] == "arrival"]
        v = src.mapping_values()
        self.assertEqual(set(v), {"xy", "density", "master", "temp"})
        self.assertGreater(v["master"], 0.7)
        self.assertEqual(len(samples), 1)  # person 3 arrived (once clearly inside the data's box)
        self.assertTrue(marks)  # and the picture had something to flash
        self.assertTrue(all(0 <= age <= EVENT_FLASH_S for _, age, _ in marks))
        self.assertEqual(len(src.fish_x_values()), len(src.agents))
        self.assertIsNone(src.repeller)  # no pulse through the predator path any more
        with self.assertRaises(ValueError):
            ReplaySource(self.path, profile="jazz")

    def test_pan_follows_the_camera(self):
        # synthetic camera that mirrors x: world x (m) -> image column 720 - 20 x
        src = ReplaySource(self.path, profile="scene", events=False)
        cam = Camera(Path("x.png"), ((-20.0, 0.0, 720.0), (0.0, 20.0, 100.0), (0.0, 0.0, 1.0)))
        src.set_pan_camera(cam, (720, 576))
        c = RuntimeControls()
        for _ in range(60 * 3):
            src.step(1.0, c, None, None)
        b = src.agents[0]
        x_m, _ = src.transform.to_data(b.position.x, b.position.y)
        self.assertAlmostEqual(src.fish_x_values()[0], (720 - 20 * x_m) / 720, places=6)

    def test_osc_overrides(self):
        sent = []
        osc = OscSender("127.0.0.1", 9)
        osc.client.send_message = lambda addr, value: sent.append((addr, value))
        osc._send_plain = lambda *a: None
        d = {"center_x_raw": 0.1, "center_y_raw": 0.2, "fish_count": 0.06, "energy": 0.5}
        osc.send_plaud(d)
        osc.send_plaud(d, xy=(0.7, 0.3), temp=0.33)
        osc.send_granular(d, RuntimeControls())
        osc.send_granular(d, RuntimeControls(), density=18.5)
        osc.send_fish_x([], 1200, 32, values=[0.2, 1.4])
        osc.send_fx_strength(0.0)
        get = lambda a: [v for k, v in sent if k == a]  # noqa: E731
        self.assertEqual(get("/plaud/x"), [0.1, 0.7])
        self.assertEqual(get("/plaud/temp"), [min(1.0, 0.2 + 0.5 * 1.3), 0.33])
        self.assertEqual(get("/gran/density"), [6.0, 18.5])
        self.assertEqual(get("/gran/fish_x"), [[0.2, 1.0]])
        self.assertEqual(get("/fx/predator_strength"), [0.0])


if __name__ == "__main__":
    unittest.main()
