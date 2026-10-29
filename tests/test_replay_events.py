"""Offline tests for replay events (replay/events.py and ReplaySource's repeller). No window.

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

from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.replay.events import EventDetector, EventParams  # noqa: E402
from aquarium_boids.sources import ReplaySource, SimulationSource  # noqa: E402

FPS = 60.0


def feed(det: EventDetector, signal, seconds: float, rate: float = FPS):
    """Run det over signal(t) sampled at `rate`; returns (onset times, release times)."""
    onsets, releases = [], []
    for i in range(int(seconds * rate)):
        t = i / rate
        det.update(t, signal(t))
        if det.onset:
            onsets.append(t)
        if det.released:
            releases.append(t)
    return onsets, releases


def noise(t: float) -> float:
    """Deterministic small wiggle around 0.3 (no random)."""
    return 0.3 + 0.03 * math.sin(7.1 * t) + 0.02 * math.sin(2.3 * t + 1.0)


def spikes(*starts: float, length: float = 0.4, height: float = 1.0):
    return lambda t: noise(t) + (height if any(s <= t < s + length for s in starts) else 0.0)


class Detector(unittest.TestCase):
    def test_one_spike_gives_one_onset_release_pair(self):
        det = EventDetector()
        onsets, releases = feed(det, spikes(20.0), 40.0)
        self.assertEqual(len(onsets), 1)
        self.assertEqual(len(releases), 1)
        self.assertAlmostEqual(onsets[0], 20.0, delta=0.05)
        self.assertGreaterEqual(releases[0] - onsets[0], 0.5)  # minimum hold
        self.assertFalse(det.active)

    def test_short_spike_still_holds_half_a_second(self):
        det = EventDetector()
        onsets, releases = feed(det, spikes(20.0, length=0.05), 40.0)
        self.assertEqual(len(onsets), 1)
        self.assertGreaterEqual(releases[0] - onsets[0], 0.5)

    def test_refractory_period(self):
        # second spike ~3 s after the first one ends: inside the default 5 s refractory -> ignored
        onsets, _ = feed(EventDetector(), spikes(20.0, 24.0), 40.0)
        self.assertEqual(len(onsets), 1)
        # with a 0.5 s refractory it counts
        onsets, _ = feed(EventDetector(EventParams(refractory=0.5)), spikes(20.0, 24.0), 40.0)
        self.assertEqual(len(onsets), 2)
        # far apart: two events
        onsets, _ = feed(EventDetector(), spikes(20.0, 30.0), 40.0)
        self.assertEqual(len(onsets), 2)

    def test_warm_up_and_flat_signal(self):
        onsets, _ = feed(EventDetector(), spikes(2.0), 12.0)  # before window / 2 = 5 s
        self.assertEqual(onsets, [])
        onsets, _ = feed(EventDetector(), noise, 60.0)
        self.assertEqual(onsets, [])

    def test_threshold(self):
        small = spikes(20.0, height=0.08)
        self.assertEqual(len(feed(EventDetector(EventParams(on=2.5)), small, 40.0)[0]), 1)
        self.assertEqual(len(feed(EventDetector(EventParams(on=6.0)), small, 40.0)[0]), 0)

    def test_defaults(self):
        self.assertEqual((EventParams().on, EventParams().refractory), (3.0, 5.0))

    def test_independent_of_the_sampling_rate(self):
        a, _ = feed(EventDetector(), spikes(20.0, 30.0), 40.0, rate=60.0)
        b, _ = feed(EventDetector(), spikes(20.0, 30.0), 40.0, rate=30.0)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, delta=0.05)

    def test_strength_has_a_floor(self):
        det = EventDetector()  # on = 3.0
        self.assertEqual(det.strength(), 0.0)  # idle
        det.active, det.peak_z = True, 3.0
        self.assertAlmostEqual(det.strength(), 0.9)  # at the onset: 0.3 of the full range, audible
        det.peak_z = 4.25
        self.assertAlmostEqual(det.strength(), 3.0 * (0.3 + 0.7 * 0.5))
        det.peak_z = 9.0
        self.assertAlmostEqual(det.strength(), 3.0)

    def test_reset(self):
        det = EventDetector()
        feed(det, spikes(20.0, length=5.0), 21.0)
        self.assertTrue(det.active)
        det.reset()
        self.assertFalse(det.active)
        self.assertEqual(det.z, 0.0)


class SourceEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # six people walking at 1.25 m/s with a slight sway (so the file's p95 |a| is not 0, as in
        # real data); person 0 turns back sharply at row 50 (20 s)
        rows = []
        for i in range(6):
            x, vx = 0.0, 0.5
            for k in range(100):
                if i == 0 and k == 50:
                    vx = -0.5
                y = 2.0 * i + 0.05 * math.sin(0.9 * k + i)
                rows.append(f"{10 * k}\t{i}\t{x:.3f}\t{y:.3f}\n")
                x += vx
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", newline="")
        f.write("".join(rows))
        f.close()
        cls.path = Path(f.name)

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)

    def test_turn_is_the_strong_event(self):
        # at z > 2.5 the six periodic sways also cross the threshold now and then (z ~2.5-2.6);
        # the turn stands far above them
        src = ReplaySource(self.path, event_params=EventParams(on=2.5, refractory=2.0))
        c = RuntimeControls()
        for _ in range(int(60 * 30)):
            src.step(1.0, c, None, None)
        strong = [e for e in src.event_log if e[3] > 5.0]
        self.assertEqual(len(strong), 1)
        self.assertAlmostEqual(strong[0][0], 20.0, delta=0.6)
        self.assertTrue(all(e[3] < 3.0 for e in src.event_log if e is not strong[0]))

    def test_turn_sets_and_clears_the_repeller(self):
        src = ReplaySource(self.path, event_params=EventParams(on=4.0))
        c = RuntimeControls()
        c.predator_strength = 1.7
        seen_repeller, strengths = None, []
        for _ in range(int(60 * 30)):
            src.step(1.0, c, None, None)
            if src.repeller is not None and seen_repeller is None:
                seen_repeller = src.repeller
            if src.detector.active:
                strengths.append(c.predator_strength)
        self.assertEqual(len(src.event_log), 1)
        t, (x, y), people, z = src.event_log[0]
        self.assertAlmostEqual(t, 20.0, delta=0.6)
        self.assertAlmostEqual(y, 0.0, delta=0.06)  # on the turning person (y = 0 +- sway), not a centroid
        self.assertEqual(people, 6)
        self.assertIsNotNone(seen_repeller)
        self.assertIsNone(src.repeller)  # released
        self.assertTrue(strengths)
        self.assertTrue(all(0.9 - 1e-9 <= s <= 3.0 for s in strengths))  # never below the floor
        self.assertEqual(c.predator_strength, 1.7)  # restored after the event

    def test_events_off_and_restart(self):
        src = ReplaySource(self.path, events=False)
        c = RuntimeControls()
        for _ in range(int(60 * 30)):
            src.step(1.0, c, None, None)
        self.assertIsNone(src.repeller)
        self.assertEqual(src.event_log, [])
        self.assertEqual(src.event_status(), "events off")
        src2 = ReplaySource(self.path)
        for _ in range(int(60 * 20.3)):
            src2.step(1.0, c, None, None)
        src2.reset(c.population)
        self.assertIsNone(src2.repeller)
        self.assertFalse(src2.detector.active)

    def test_simulation_has_no_repeller(self):
        self.assertIsNone(SimulationSource.repeller)


if __name__ == "__main__":
    unittest.main()
