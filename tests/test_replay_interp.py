"""Offline tests for src/aquarium_boids/replay/interp.py (stdlib unittest, no pygame, no window).

Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquarium_boids.replay.interp import (  # noqa: E402
    DataClock, Piece, State, Trajectories, per_frame, split_track, tangents,
)
from aquarium_boids.replay.loader import Dataset, Sample, load_csv  # noqa: E402

FPS = 2.5  # ETH/UCY-like keyframe rate
DT = 1 / FPS


def samples(fn, n, t0=0.0, dt=DT, skip=()):
    """Keyframes of a motion fn(t) -> (x, y), frames 0..n-1 except `skip`."""
    out = []
    for k in range(n):
        if k in skip:
            continue
        t = t0 + k * dt
        x, y = fn(t)
        out.append(Sample(t, k, x, y))
    return out


def piece(fn, n, **kw):
    return Piece.from_samples("p", 0, samples(fn, n, **kw))


def times_inside(p, per_interval=7):
    """Times strictly inside every keyframe interval, plus the keyframes themselves."""
    ts = list(p.t)
    for a, b in zip(p.t, p.t[1:]):
        ts += [a + (b - a) * j / per_interval for j in range(1, per_interval)]
    return sorted(ts)


class Tangents(unittest.TestCase):
    def test_uniform_spacing_is_catmull_rom(self):
        ts, ps = [0, 1, 2, 3], [0.0, 2.0, 1.0, 5.0]
        m = tangents(ts, ps)
        self.assertAlmostEqual(m[1], (ps[2] - ps[0]) / 2)
        self.assertAlmostEqual(m[2], (ps[3] - ps[1]) / 2)

    def test_quadratic_is_exact_with_uneven_spacing(self):
        ts = [0.0, 0.4, 1.2, 1.6, 2.4]
        f, df = (lambda t: 3 * t * t - 2 * t + 1), (lambda t: 6 * t - 2)
        for got, t in zip(tangents(ts, [f(t) for t in ts]), ts):
            self.assertAlmostEqual(got, df(t))

    def test_two_points_use_the_slope(self):
        self.assertEqual(tangents([0, 2], [1.0, 5.0]), [2.0, 2.0])


class PieceMotion(unittest.TestCase):
    def test_straight_line_constant_speed(self):
        vx, vy = 1.3, -0.7  # m/s
        p = piece(lambda t: (5 + vx * t, 2 + vy * t), 12)
        for t in times_inside(p):
            s = p.state(t)
            self.assertAlmostEqual(s.x, 5 + vx * t)
            self.assertAlmostEqual(s.y, 2 + vy * t)
            self.assertAlmostEqual(s.vx, vx)
            self.assertAlmostEqual(s.vy, vy)
            self.assertAlmostEqual(s.ax, 0.0)
            self.assertAlmostEqual(s.ay, 0.0)

    def test_constant_acceleration_is_exact_even_across_a_missing_row(self):
        g = (0.4, -1.1)
        fn = lambda t: (1 + 2 * t + 0.5 * g[0] * t * t, 3 - t + 0.5 * g[1] * t * t)  # noqa: E731
        p = piece(fn, 10, skip={4})
        for t in times_inside(p):
            s = p.state(t)
            self.assertAlmostEqual(s.ax, g[0])
            self.assertAlmostEqual(s.ay, g[1])
            self.assertAlmostEqual(s.vx, 2 + g[0] * t)
            self.assertAlmostEqual(s.x, fn(t)[0])

    def test_circle_gives_centripetal_acceleration(self):
        r, period, c = 4.0, 10.0, (10.0, 6.0)  # 25 keyframes per turn at 2.5 fps
        w = 2 * math.pi / period
        p = piece(lambda t: (c[0] + r * math.cos(w * t), c[1] + r * math.sin(w * t)), 40)
        # Measured at this sampling: pos 0.01 %, |v| 1.05 %, |a| 5.5 % max / 1.8 % mean,
        # direction of a 13.6 deg max / 7 deg mean (a is linear inside each 0.4 s interval).
        err = {"pos": [], "speed": [], "acc": [], "angle": []}
        for t in times_inside(p, 20):
            if not (p.t[2] <= t <= p.t[-3]):  # away from the one-sided ends
                continue
            s = p.state(t)
            to_centre = (c[0] - s.x, c[1] - s.y)
            a = math.hypot(s.ax, s.ay)
            cos = (s.ax * to_centre[0] + s.ay * to_centre[1]) / (a * math.hypot(*to_centre))
            err["pos"].append(abs(math.hypot(*to_centre) - r) / r)
            err["speed"].append(abs(math.hypot(s.vx, s.vy) - w * r) / (w * r))
            err["acc"].append(abs(a - w * w * r) / (w * w * r))
            err["angle"].append(math.degrees(math.acos(min(1.0, cos))))
        mean = {k: sum(v) / len(v) for k, v in err.items()}
        self.assertLess(max(err["pos"]), 0.0005)   # radius within 0.05 %
        self.assertLess(max(err["speed"]), 0.015)  # |v| = w r within 1.5 %
        self.assertLess(max(err["acc"]), 0.07)     # |a| = w^2 r within 7 %
        self.assertLess(mean["acc"], 0.025)
        self.assertLess(max(err["angle"]), 15.0)   # a points towards the centre
        self.assertLess(mean["angle"], 8.0)

    def test_passes_through_keyframes_and_velocity_is_continuous(self):
        fn = lambda t: (math.sin(1.7 * t) * 3, t * t - math.cos(t))  # noqa: E731
        p = piece(fn, 9)
        for k, t in enumerate(p.t):
            s = p.state(t)
            self.assertAlmostEqual((s.x, s.y), (p.x[k], p.y[k]))
        for t in p.t[1:-1]:
            left, right = p.state(t - 1e-9), p.state(t + 1e-9)
            self.assertAlmostEqual(left.vx, right.vx, places=6)
            self.assertAlmostEqual(left.vy, right.vy, places=6)

    def test_derivatives_match_finite_differences(self):
        p = piece(lambda t: (math.sin(1.7 * t) * 3, t * t - math.cos(t)), 9)
        e = 1e-5
        for t in times_inside(p, 5):
            if t - e < p.t0 or t + e > p.t1 or any(abs(t - k) < 2 * e for k in p.t):
                continue
            a, s, b = p.state(t - e), p.state(t), p.state(t + e)
            self.assertAlmostEqual(s.vx, (b.x - a.x) / (2 * e), places=5)
            self.assertAlmostEqual(s.ay, (b.vy - a.vy) / (2 * e), places=4)

    def test_time_is_clamped_to_the_piece(self):
        p = piece(lambda t: (t, 0.0), 3)
        self.assertEqual(p.state(-5).x, p.x[0])
        self.assertEqual(p.state(99).x, p.x[-1])


class Gaps(unittest.TestCase):
    def test_one_missing_row_is_bridged_two_split(self):
        s = samples(lambda t: (t, t), 10, skip={3, 6, 7})
        runs = split_track(tuple(s), frame_step=1, max_missing=1)
        self.assertEqual([[x.frame for x in r] for r in runs], [[0, 1, 2, 4, 5], [8, 9]])
        self.assertEqual(len(split_track(tuple(s), 1, max_missing=0)), 3)

    def test_frame_step_scales_the_gap(self):
        s = [Sample(k * DT, 780 + 10 * k, 0.0, 0.0) for k in (0, 1, 3, 6)]  # frames 780, 790, 810, 840
        runs = split_track(tuple(s), frame_step=10)
        self.assertEqual([[x.frame for x in r] for r in runs], [[780, 790, 810], [840]])


class TrajectoriesSet(unittest.TestCase):
    def make(self):
        # a: frames 0-4; b: frames 2-3, then 7-9 after a gap; c: single row (dropped)
        rows = {
            "a": [Sample(k * DT, k, k, 0.0) for k in range(5)],
            "b": [Sample(k * DT, k, 0.0, k) for k in (2, 3, 7, 8, 9)],
            "c": [Sample(6 * DT, 6, 1.0, 1.0)],
        }
        return Trajectories(Dataset({k: tuple(v) for k, v in rows.items()}, fps=FPS))

    def test_pieces_and_dropped(self):
        tr = self.make()
        self.assertEqual([(p.track_id, p.index) for p in tr.pieces], [("a", 0), ("b", 0), ("b", 1)])
        self.assertEqual(tr.dropped, 1)
        self.assertAlmostEqual(tr.t_start, 0.0)
        self.assertAlmostEqual(tr.t_end, 9 * DT)

    def test_active_set_over_time(self):
        tr = self.make()
        ids = lambda t: [(p.track_id, p.index) for p in tr.active(t)]  # noqa: E731
        self.assertEqual(ids(0.5 * DT), [("a", 0)])
        self.assertEqual(ids(2.5 * DT), [("a", 0), ("b", 0)])
        self.assertEqual(ids(5.0 * DT), [])
        self.assertEqual(ids(8.0 * DT), [("b", 1)])
        self.assertEqual(ids(99.0), [])

    def test_nothing_to_replay(self):
        with self.assertRaisesRegex(ValueError, "nothing to replay"):
            Trajectories(Dataset({"x": (Sample(0.0, 0, 0.0, 0.0),)}, fps=FPS))

    def test_from_a_loaded_eth_style_csv(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
        f.write("id,frame,x,y\n" + "".join(f"1,{780 + 10 * k},{0.5 * k},0\n" for k in range(6)))
        f.close()
        try:
            tr = Trajectories(load_csv(f.name, fps=2.5))
        finally:
            Path(f.name).unlink()
        (p,) = tr.pieces
        self.assertAlmostEqual(p.state(p.t0 + 0.2).vx, 0.5 / 0.4)  # 0.5 m per 0.4 s


class Units(unittest.TestCase):
    def test_per_frame_scales_v_by_k_and_a_by_k_squared(self):
        s = per_frame(State(1.0, 2.0, 60.0, -30.0, 3600.0, 7200.0), speed=2.0)
        self.assertEqual((s.x, s.y), (1.0, 2.0))
        self.assertAlmostEqual(s.vx, 2.0)
        self.assertAlmostEqual(s.vy, -1.0)
        self.assertAlmostEqual(s.ax, 4.0)
        self.assertAlmostEqual(s.ay, 8.0)

    def test_per_frame_velocity_matches_frame_displacement(self):
        p = piece(lambda t: (40 * t, 10 * t * t), 10)  # px / s
        speed, t = 1.5, 1.3
        a, b = p.state(t), p.state(t + speed / 60.0)
        v = per_frame(p.state(t + speed / 120.0), speed)
        self.assertAlmostEqual(v.vx, b.x - a.x, places=9)
        self.assertAlmostEqual(v.vy, b.y - a.y, places=9)


class Clock(unittest.TestCase):
    def test_speed_and_pause(self):
        c = DataClock(10.0, 20.0, speed=2.0)
        c.advance(1.0)
        self.assertAlmostEqual(c.t, 10.0 + 2.0 / 60.0)
        c.advance(1.0, paused=True)
        self.assertAlmostEqual(c.t, 10.0 + 2.0 / 60.0)
        for _ in range(60):
            c.advance(0.5)
        self.assertAlmostEqual(c.t, 10.0 + 2.0 / 60.0 + 1.0)

    def test_end_without_loop(self):
        c = DataClock(0.0, 1.0)
        for _ in range(100):
            c.advance(1.0)
        self.assertEqual(c.t, 1.0)
        self.assertTrue(c.finished)
        c.reset()
        self.assertEqual(c.t, 0.0)
        self.assertFalse(c.finished)

    def test_loop_wraps(self):
        c = DataClock(5.0, 6.0, loop=True)
        for _ in range(66):  # 1.1 s of data
            c.advance(1.0)
        self.assertAlmostEqual(c.t, 5.1)
        self.assertFalse(c.finished)
        wraps = 0
        for _ in range(120):
            c.advance(1.0)
            wraps += c.wrapped
        self.assertEqual(wraps, 2)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            DataClock(0.0, 1.0, speed=0.0)
        with self.assertRaises(ValueError):
            DataClock(2.0, 1.0)


if __name__ == "__main__":
    unittest.main()
