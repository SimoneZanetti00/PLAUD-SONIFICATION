"""Offline tests for the replay sound profile `scene` (replay/mapping.py, ReplaySource, osc_io overrides).

Synthetic terrain, tracks and homography only (no model or dataset files). No window, no sockets.
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
from aquarium_boids.replay.camera import Camera  # noqa: E402
from aquarium_boids.replay.mapping import (  # noqa: E402
    EVENT_MIN_GAP_S, LIMIT_ONE, LIMIT_TENSION, LOUDNESS_ONE, MASTER_RANGE_DB, MASTER_TOP_DB, PAIR_STEP_S,
    STRETCH_MAX, TERRITORY_TAU_S, Person, SceneMapping, Terrain, axial_disorder, cadence_density, centroid_level,
    closeness_level, count_fraction, count_range, db_to_master, limit_level, loudness_level, master_level,
    together_rule,
)
from aquarium_boids.sources import EVENT_FLASH_S, ReplaySource  # noqa: E402


def write_terrain(features: dict) -> Path:
    d = {"resolution": 128, "orientation": "row0_is_y0_bottom", "features": features}
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(d, f)
    f.close()
    return Path(f.name)


def synthetic_terrain() -> Path:
    """Like a real terrain map's main directions: onsets and rhythm high on the left; percussive / bright / noisy at
    the top, loud at the bottom."""
    n = 128
    left = [[1.0 - j / (n - 1) for j in range(n)] for _ in range(n)]
    up = [[i / (n - 1)] * n for i in range(n)]  # row 0 = bottom
    down = [[1.0 - i / (n - 1)] * n for i in range(n)]
    return write_terrain({"onset_density": left, "rhythmicity": left, "percussiveness": up, "brightness": up,
                          "noisiness": up, "loudness": down})


def two_sided_terrain() -> Path:
    """Onsets and rhythm high at both the left and the right edge, low in the middle; texture flat."""
    n = 128
    edges = [[abs(j / (n - 1) - 0.5) * 2 for j in range(n)] for _ in range(n)]
    flat = [[0.5] * n for _ in range(n)]
    return write_terrain({"onset_density": edges, "rhythmicity": edges, "percussiveness": flat,
                          "brightness": flat, "noisiness": flat, "loudness": flat})


def person(key, x, y, vx, vy=0.0, pan=0.5, visible=True):
    return Person(key, x, y, vx, vy, pan, x * 50, y * 50, visible)


def run(m, people, seconds, t=0.0, agitation=0.0):
    """Feed the same people for `seconds` of data time at 60 steps per second; returns the end time."""
    for _ in range(int(round(60 * seconds))):
        t += 1 / 60
        m.update(t, 1 / 60, people, agitation)
    return t


class Curves(unittest.TestCase):
    def test_master_level(self):
        self.assertEqual(master_level(0, 20), 0.0)
        self.assertAlmostEqual(master_level(20, 20), db_to_master(MASTER_TOP_DB))
        self.assertAlmostEqual(master_level(20, 20), 1 - 6 / 70)  # the device's Dry default
        levels = [master_level(n, 20) for n in range(1, 21)]
        self.assertEqual(levels, sorted(levels))  # more people, louder
        self.assertAlmostEqual(70 * (levels[-1] - levels[0]), MASTER_RANGE_DB * (1 - 0.228), delta=0.1)  # ~4.6 dB
        self.assertEqual(master_level(40, 20), master_level(20, 20))

    def test_count_is_relative_to_the_file(self):
        self.assertEqual(count_range(0, 11), (1.0, 11.0))  # hotel: empty at p5, still one person = 0
        self.assertEqual(count_range(27, 65), (27.0, 65.0))  # univ
        self.assertEqual(count_range(2, 2), (2.0, 4.0))  # a flat file still gets a range
        self.assertEqual(count_fraction(1, (1, 10)), 0.0)
        self.assertEqual(count_fraction(10, (1, 10)), 1.0)
        self.assertAlmostEqual(count_fraction(5, (1, 10)), math.log(5) / math.log(10))
        self.assertEqual(count_fraction(27, (27, 65)), 0.0)  # univ's quiet end
        self.assertAlmostEqual(count_fraction(48, (27, 65)), 0.66, delta=0.01)
        self.assertEqual(count_fraction(90, (27, 65)), 1.0)

    def test_loudness_and_limit(self):
        rng = (1.0, 10.0)
        self.assertEqual(loudness_level(0, rng), 0.0)
        self.assertAlmostEqual(loudness_level(1, rng), LOUDNESS_ONE)
        self.assertAlmostEqual(loudness_level(10, rng), 1.0)
        loud = [loudness_level(n, rng) for n in range(1, 11)]
        self.assertEqual(loud, sorted(loud))
        self.assertAlmostEqual(limit_level(0.0, 0.0), LIMIT_ONE)  # one person, nobody close: few partials
        self.assertEqual(limit_level(1.0, 0.0), 0.0)  # a full scene, spread out: all partials
        self.assertAlmostEqual(limit_level(1.0, 1.0), LIMIT_TENSION)  # a full scene pressed together: thinner
        self.assertEqual(limit_level(0.0, 1.0), 1.0)

    def test_centroid_from_walking_speed(self):
        self.assertEqual(centroid_level(0.0), 0.0)
        self.assertEqual(centroid_level(0.4), 0.0)
        self.assertAlmostEqual(centroid_level(0.95), 0.5)
        self.assertEqual(centroid_level(2.5), 1.0)

    def test_axial_disorder(self):
        self.assertAlmostEqual(axial_disorder([(1.2, 0), (-1.3, 0), (1.1, 0.05)]), 0.0, places=2)  # a two-way street
        self.assertAlmostEqual(axial_disorder([(1.2, 0), (0, 1.2), (-1.2, 0), (0, -1.2)]), 1.0, places=6)  # a crossing
        self.assertAlmostEqual(axial_disorder([(1.0, 1.0), (-1.0, 1.0)]), 1.0, places=6)  # two perpendicular lines
        self.assertIsNone(axial_disorder([(1.2, 0), (0.1, 0)]))  # one walker, one standing
        self.assertIsNone(axial_disorder([]))

    def test_closeness(self):
        self.assertEqual(closeness_level([(0, 0), (0.7, 0)]), 1.0)  # a pair side by side
        self.assertEqual(closeness_level([(0, 0), (5, 0), (10, 0)]), 0.0)  # spread out
        self.assertAlmostEqual(closeness_level([(0, 0), (1.6, 0)]), 0.5)
        self.assertIsNone(closeness_level([(0, 0)]))

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
        cls.two_path = two_sided_terrain()
        cls.two = Terrain.load(cls.two_path)

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)
        cls.two_path.unlink(missing_ok=True)

    def test_features_are_scaled(self):
        for k in (2, 3, 4):
            vals = [c[k] for c in self.terrain.cells]
            self.assertEqual((min(vals), max(vals)), (0.0, 1.0))
        onset, rhythm, texture = self.terrain.features_at(0.01, 0.99)  # top left
        self.assertEqual((onset, rhythm, texture), (1.0, 1.0, 1.0))
        self.assertEqual(self.terrain.features_at(0.99, 0.01), (0.0, 0.0, 0.0))  # bottom right

    def test_best_match(self):
        x, y = self.terrain.best((1.0, 1.0, 0.0), near=(0.0, 0.0))  # busy, hurried, ordered
        self.assertLess(x, 0.1)
        self.assertLess(y, 0.1)
        x, y = self.terrain.best((0.0, 0.0, 1.0), near=(1.0, 1.0))  # sparse, slow, disordered
        self.assertGreater(x, 0.9)
        self.assertGreater(y, 0.9)

    def test_no_far_jump_between_equal_patches(self):
        x, _ = self.two.best((1.0, 1.0, 0.5), near=(0.1, 0.5))
        self.assertLess(x, 0.3)  # the left edge, where the territory already is
        x, _ = self.two.best((1.0, 1.0, 0.5), near=(0.9, 0.5))
        self.assertGreater(x, 0.7)  # the right edge, as good, not across the map

    def test_territory_glides_in_seconds(self):
        m = SceneMapping(max_count=10, terrain=self.terrain, count_range=(1.0, 10.0))
        m.update(0.0, 0.0, [person("a", 0, 0, 0.3)], 0.0)  # one person standing: right, bottom
        start = m.territory
        self.assertGreater(start[0], 0.85)
        crowd = [person(str(i), 3 * i, 0, 1.6 if i % 2 else -1.6) for i in range(10)]  # busy two-way street
        t = run(m, crowd, TERRITORY_TAU_S)
        moved = (start[0] - m.territory[0]) / start[0]
        self.assertGreater(moved, 0.35)  # several seconds, not tens
        self.assertLess(moved, 0.9)  # but a glide, not a jump
        run(m, crowd, 16, t)
        self.assertLess(m.territory[0], 0.15)  # busy and hurried: left
        self.assertLess(m.territory[1], 0.15)  # ordered: down (fuller, louder)

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
        m.update(0.0, 0.0, walkers)
        self.assertAlmostEqual(m.master, master_level(2, 10))
        t = run(m, [], 8)
        self.assertEqual(m.master, 0.0)
        m.update(t + 1 / 60, 1 / 60, walkers[:1])
        self.assertGreater(m.master, 0.0)
        self.assertLess(m.master, master_level(1, 10) / 2)

    def test_cadence(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.3), person("b", 5, 5, 0.1)])
        self.assertAlmostEqual(m.step_rate, 1.3 / 0.7)
        m.update(0.1, 0.1, [person("b", 5, 5, 0.1)])
        self.assertEqual((m.step_rate, m.density), (0.0, 0.0))

    def test_agitation_is_averaged_for_l2(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.2)], agitation_now=0.0)
        m.update(0.25, 0.25, [person("a", 0, 0, 1.2)], agitation_now=1.0)  # a spike: L2 barely moves
        self.assertLess(m.temp, 0.3)
        run(m, [person("a", 0, 0, 1.2)], 12, 0.25, agitation=1.0)  # sustained: L2 follows within seconds
        self.assertGreater(m.temp, 0.95)

    def test_plaud_follows_count_speed_and_closeness(self):
        m = SceneMapping(max_count=20, count_range=(1.0, 15.0))
        m.update(0.0, 0.0, [person("a", 0, 0, 0.5)])
        self.assertAlmostEqual((m.loudness, m.limit, m.stretch), (LOUDNESS_ONE, LIMIT_ONE, 0.0))
        spread = [person(str(i), 3 * i, 0, 1.5) for i in range(15)]  # 15 brisk walkers, 3 m apart
        t = run(m, spread, 8)
        self.assertGreater(m.loudness, 0.95)
        self.assertLess(m.limit, 0.05)  # all partials
        self.assertGreater(m.centroid, 0.9)
        self.assertLess(m.stretch, 0.01)
        pressed = [person(str(i), 0.6 * i, 0, 1.5) for i in range(15)]  # the same people, shoulder to shoulder
        t = run(m, pressed, 10, t)
        self.assertAlmostEqual(m.stretch, STRETCH_MAX, delta=0.01)
        self.assertAlmostEqual(m.limit, LIMIT_TENSION, delta=0.01)  # tension thins the partials again
        run(m, [], 6, t)  # everybody leaves: loudness falls to 0, the rest is held
        self.assertLess(m.loudness, 0.05)
        self.assertAlmostEqual(m.limit, LIMIT_TENSION, delta=0.01)
        self.assertGreater(m.centroid, 0.9)

    def test_count_relative_to_a_dense_file(self):
        m = SceneMapping(max_count=80, count_range=(27.0, 65.0))  # univ
        quiet = [person(str(i), 3 * i, 0, 0.6) for i in range(27)]
        m.update(0.0, 0.0, quiet)
        self.assertAlmostEqual(m.loudness, LOUDNESS_ONE)  # univ's quiet end is low, not saturated
        run(m, [person(str(i), 3 * (i % 20), 3 * (i // 20), 0.6) for i in range(65)], 8)
        self.assertGreater(m.loudness, 0.95)

    def test_disorder_drives_texture(self):
        m = SceneMapping(max_count=10)  # no terrain: the plain axes, y_up = texture
        street = [person(str(i), 3 * i, 0, 1.2 if i % 2 else -1.2) for i in range(6)]
        t = run(m, street, 10)
        self.assertLess(m.disorder, 0.05)
        self.assertLess(m.territory[1], 0.1)  # ordered: down
        crossing = [person(str(i), 3 * i, 0, 1.2 * math.cos(i * math.pi / 3), 1.2 * math.sin(i * math.pi / 3))
                    for i in range(6)]
        run(m, crossing, 15, t)
        self.assertGreater(m.disorder, 0.9)
        self.assertGreater(m.territory[1], 0.9)  # everybody crossing every way: up (percussive, noisy)

    def test_arrival_waits_until_visible(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.2)])
        m.update(1.0, 1.0, [person("a", 0, 0, 1.2), person("b", 9, 0, 1.2, pan=0.97, visible=False)])
        self.assertEqual(m.pad_samples, [])  # entering through the edge band: not yet
        m.update(2.0, 1.0, [person("a", 0, 0, 1.2), person("b", 8, 0, 1.2, pan=0.9, visible=True)])
        self.assertEqual(m.pad_samples, [4])  # now clearly in the picture, on the right: slice 4
        self.assertEqual(m.events[-1][:1], ("arrival",))
        m.update(3.0, 1.0, [person("a", 0, 0, 1.2), person("c", 1, 1, 1.2, pan=0.1)])
        self.assertEqual(m.pad_samples, [4])  # within EVENT_MIN_GAP_S: capped
        self.assertEqual((m.arrivals_seen, m.arrivals_sent), (2, 1))
        m.update(2.0 + EVENT_MIN_GAP_S + 0.1, 0.1, [person("d", 0, 0, 1.2, pan=0.0)])
        self.assertEqual(m.pad_samples, [4, 1])

    def test_never_seen_newcomer_makes_no_sound(self):
        m = SceneMapping(max_count=10)
        m.update(0.0, 0.0, [person("a", 0, 0, 1.2)])
        for k in range(5):  # walks along the edge band and leaves
            m.update(1.0 + k, 1.0, [person("a", 0, 0, 1.2), person("e", 9, k, 1.2, visible=False)])
        m.update(7.0, 1.0, [person("a", 0, 0, 1.2)])
        self.assertEqual((m.pad_samples, m.arrivals_seen), ([], 0))

    def test_walking_together_kick(self):
        m = SceneMapping(max_count=10)
        t = 0.0
        pair = [person("a", 0, 0, 1.25), person("b", 0, 1, 1.2)]
        m.update(t, 0.0, pair)
        while not m.kicks and t < 5:
            t += 1 / 60
            m.update(t, 1 / 60, pair)
        self.assertAlmostEqual(t, 2.0, delta=PAIR_STEP_S + 0.05)  # after 2 s of walking together (10 Hz check)
        self.assertEqual(m.kicks, 1)
        self.assertEqual(m.events[-1][0], "together")
        self.assertEqual(set(m.events[-1][2]), {"a", "b"})

    def test_pair_in_the_edge_band_is_silent(self):
        m = SceneMapping(max_count=10)
        pair = [person("a", 0, 0, 1.25), person("b", 0, 1, 1.2, visible=False)]
        m.update(0.0, 0.0, pair)
        run(m, pair, 4)
        self.assertEqual((m.kicks, m.together_seen), (0, 0))

    def test_pause_holds(self):
        m = SceneMapping(max_count=10)
        two = [person("a", 0, 0, 1.2), person("b", 3, 0, 0.0, 1.2)]
        m.update(0.0, 0.0, two)
        run(m, two, 1)
        state = (m.count01, m.disorder, m.closeness, m.territory, m.limit, m.loudness)
        for _ in range(100):  # paused: data time does not move
            m.update(1.0, 0.0, two)
        self.assertEqual((m.count01, m.disorder, m.closeness, m.territory, m.limit, m.loudness), state)


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
        self.assertEqual(set(v), {"xy", "density", "master", "temp", "loudness", "limit", "centroid", "stretch"})
        self.assertEqual(src.mapping.count_range, (2.0, 4.0))  # the file: 2 people, then 3 (p5 2, p95 3)
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
        osc.send_plaud(d, xy=(0.7, 0.3), temp=0.33, loudness=0.42)
        osc.send_plaud_bend(0.25, 0.6, 0.3)
        osc.send_granular(d, RuntimeControls())
        osc.send_granular(d, RuntimeControls(), density=18.5)
        osc.send_fish_x([], 1200, 32, values=[0.2, 1.4])
        osc.send_fx_strength(0.0)
        get = lambda a: [v for k, v in sent if k == a]  # noqa: E731
        self.assertEqual(get("/plaud/x"), [0.1, 0.7])
        self.assertEqual(get("/plaud/temp"), [min(1.0, 0.2 + 0.5 * 1.3), 0.33])
        self.assertEqual(get("/plaud/loudness"), [0.6 + 0.06 * 0.8, 0.42])
        self.assertEqual((get("/plaud/limit"), get("/plaud/centroid"), get("/plaud/stretch")), ([0.25], [0.6], [0.3]))
        names = [k for k, _ in sent]
        self.assertLess(names.index("/plaud/loudness", 6), names.index("/plaud/limit"))  # limit wins in Max
        self.assertEqual(get("/gran/density"), [6.0, 18.5])
        self.assertEqual(get("/gran/fish_x"), [[0.2, 1.0]])
        self.assertEqual(get("/fx/predator_strength"), [0.0])


if __name__ == "__main__":
    unittest.main()
