"""Offline tests for the replay sound profile `scene` (replay/mapping.py, ReplaySource, osc_io overrides).

Synthetic terrain and tracks only (no model or dataset files). No window, no sockets.
Run from the repository root:  python -m unittest discover -s tests -v
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.osc_io import OscSender  # noqa: E402
from aquarium_boids.replay.mapping import (  # noqa: E402
    EVENT_MIN_GAP_S, MASTER_DEFAULT_DB, PAIR_STEP_S, PULSE_S, Person, SceneMapping, Terrain, cadence_density, db_to_master,
    master_level, together_rule,
)
from aquarium_boids.sources import ReplaySource  # noqa: E402


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


def person(key, x, y, vx, vy=0.0, x01=0.5):
    return Person(key, x, y, vx, vy, x01, x * 50, y * 50)


class Curves(unittest.TestCase):
    def test_master_level(self):
        self.assertEqual(master_level(0, 20), 0.0)
        self.assertAlmostEqual(master_level(20, 20), db_to_master(MASTER_DEFAULT_DB))
        self.assertAlmostEqual(master_level(20, 20), 1 - 6 / 70)
        levels = [master_level(n, 20) for n in range(1, 21)]
        self.assertEqual(levels, sorted(levels))  # more people, louder
        self.assertGreater(70 * (levels[0] - 1), -16)  # one person stays audible (about -13 dB)
        self.assertEqual(master_level(40, 20), master_level(20, 20))  # capped at the default level

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
        x, y = self.terrain.point(hurry=1.0, crowding=1.0)  # hurried and crowded: left, bottom
        self.assertLess(x, 0.15)
        self.assertLess(y, 0.15)
        x, y = self.terrain.point(hurry=0.0, crowding=0.0)  # calm and sparse: right, top
        self.assertGreater(x, 0.85)
        self.assertGreater(y, 0.85)
        xm, _ = self.terrain.point(0.5, 0.5)
        self.assertAlmostEqual(xm, 0.5, delta=0.1)

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
        self.assertAlmostEqual(m.master, master_level(2, 10))  # starts at the scene's level, no fade
        t = 0.0
        for _ in range(int(60 * 8)):  # everybody leaves: fade to silence within a few seconds
            t += 1 / 60
            m.update(t, 1 / 60, [], 0.0)
        self.assertEqual(m.master, 0.0)
        m.update(t + 1 / 60, 1 / 60, walkers[:1], 0.5)  # someone arrives: soft fade-in, not a jump
        self.assertGreater(m.master, 0.0)
        self.assertLess(m.master, master_level(1, 10) / 2)

    def test_cadence(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.3), person("b", 5, 5, 0.1)], 0.5)  # b stands
        self.assertAlmostEqual(m.step_rate, 1.3 / 0.7)
        m.update(0.1, 0.1, [person("b", 5, 5, 0.1)], 0.0)
        self.assertEqual((m.step_rate, m.density), (0.0, 0.0))

    def test_arrivals_slices_and_cap(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.2)], 0.5)
        self.assertEqual(m.pad_samples, [])  # present at the start: not an arrival
        m.update(1.0, 1.0, [person("a", 0, 0, 1.2), person("b", 9, 0, 1.2, x01=0.9)], 0.5)
        self.assertEqual(m.pad_samples, [4])  # entered on the right: slice 4
        m.update(2.0, 1.0, [person("a", 0, 0, 1.2), person("b", 9, 0, 1.2), person("c", 1, 1, 1.2, x01=0.1)], 0.5)
        self.assertEqual(m.pad_samples, [4])  # within EVENT_MIN_GAP_S: capped
        self.assertEqual((m.arrivals_seen, m.arrivals_sent), (2, 1))
        m.update(1.0 + EVENT_MIN_GAP_S + 0.1, 0.1, [person("d", 0, 0, 1.2, x01=0.0)], 0.5)
        self.assertEqual(m.pad_samples, [4, 1])

    def test_walking_together_pulse(self):
        m = SceneMapping(max_count=10)
        t = 0.0
        pair = [person("a", 0, 0, 1.25), person("b", 0, 1, 1.2)]
        m.update(t, 0.0, pair, 0.5)
        while m.pulse is None and t < 5:
            t += 1 / 60
            m.update(t, 1 / 60, pair, 0.5)
        self.assertAlmostEqual(t, 2.0, delta=PAIR_STEP_S + 0.05)  # after 2 s of walking together (10 Hz check)
        self.assertEqual((m.pulse[0], m.pulse[1]), (0.0, 25.0))  # at their midpoint (world px)
        start = t
        while m.pulse is not None:
            t += 1 / 60
            m.update(t, 1 / 60, pair, 0.5)
        self.assertAlmostEqual(t - start, PULSE_S, delta=0.05)
        self.assertEqual(m.together_sent, 1)

    def test_pause_holds(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.2)], 0.9)
        state = (m.hurry, m.territory, m.master)
        for _ in range(100):
            m.update(0.0, 0.0, [], 0.0)  # no data time passes
        self.assertEqual((m.hurry, m.territory), state[:2])


class SourceAndOsc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = "".join(f"{10 * k}\t{i}\t{0.5 * k}\t{2.0 * i}\n" for i in (1, 2) for k in range(60))
        rows += "".join(f"{10 * k}\t3\t{30 - 0.5 * (k - 20)}\t9\n" for k in range(20, 60))  # arrives at 8 s
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
        self.assertEqual(src.take_pad_samples(), [])

    def test_scene_values_and_arrival(self):
        src = ReplaySource(self.path, profile="scene", events=False)
        c = RuntimeControls()
        samples = []
        for _ in range(60 * 12):
            src.step(1.0, c, None, None)
            samples += src.take_pad_samples()
        v = src.mapping_values()
        self.assertEqual(set(v), {"xy", "density", "master"})
        self.assertGreater(v["master"], 0.8)
        self.assertAlmostEqual(src.mapping.step_rate, 1.25 / 0.7, delta=0.05)
        self.assertEqual(len(samples), 1)  # person 3 arrived
        self.assertIn("sound scene", "\n".join(src.hud_lines(c, {})))
        with self.assertRaises(ValueError):
            ReplaySource(self.path, profile="jazz")

    def test_osc_overrides(self):
        sent = []
        osc = OscSender("127.0.0.1", 9)
        osc.client.send_message = lambda addr, value: sent.append((addr, value))
        osc._send_plain = lambda *a: None
        d = {"center_x_raw": 0.1, "center_y_raw": 0.2, "fish_count": 0.06}
        osc.send_plaud(d)
        osc.send_plaud(d, xy=(0.7, 0.3))
        osc.send_granular(d, RuntimeControls())
        osc.send_granular(d, RuntimeControls(), density=18.5)
        x = [v for a, v in sent if a == "/plaud/x"]
        dens = [v for a, v in sent if a == "/gran/density"]
        self.assertEqual(x, [0.1, 0.7])
        self.assertEqual(dens[0], 6.0)
        self.assertEqual(dens[1], 18.5)


if __name__ == "__main__":
    unittest.main()
