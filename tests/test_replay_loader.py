"""Offline tests for src/aquarium_boids/replay/loader.py (stdlib unittest, no pygame, no window).

Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aquarium_boids.replay.loader import Bounds, data_bounds, fit_to_world, load_csv, to_world  # noqa: E402


_TEMP_FILES: list[Path] = []


def write_csv(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    f.write(text)
    f.close()
    _TEMP_FILES.append(Path(f.name))
    return Path(f.name)


def tearDownModule():
    for p in _TEMP_FILES:
        p.unlink(missing_ok=True)


class LoadCsv(unittest.TestCase):
    def test_basic_tracks_sorted_and_timed_by_fps(self):
        p = write_csv("id,frame,x,y\nb,2,5,6\na,1,1,2\na,0,0,0\nb,1,3,4\n")
        ds = load_csv(p, fps=2.5)
        self.assertEqual(sorted(ds.tracks), ["a", "b"])
        self.assertEqual([s.frame for s in ds.tracks["a"]], [0, 1])
        self.assertAlmostEqual(ds.tracks["b"][1].t, 2 / 2.5)
        self.assertEqual((ds.tracks["b"][0].x, ds.tracks["b"][0].y), (3.0, 4.0))
        self.assertEqual(ds.max_population, 2)
        self.assertAlmostEqual(ds.duration, 2 / 2.5)

    def test_t_column_overrides_fps_and_header_is_flexible(self):
        p = write_csv("﻿X, Y ,ID,Frame,t,extra\n1,2,7,10,0.5,zz\n3,4,7,11,0.9,zz\n")
        ds = load_csv(p, fps=25)
        self.assertEqual(list(ds.tracks), ["7"])
        self.assertEqual([s.t for s in ds.tracks["7"]], [0.5, 0.9])

    def test_blank_lines_are_skipped(self):
        ds = load_csv(write_csv("id,frame,x,y\n1,0,0,0\n\n1,1,1,1\n"))
        self.assertEqual(len(ds.tracks["1"]), 2)

    def test_missing_column(self):
        with self.assertRaisesRegex(ValueError, "missing column"):
            load_csv(write_csv("id,frame,x\n1,0,0\n"))

    def test_bad_number_reports_line(self):
        with self.assertRaisesRegex(ValueError, r":3: bad value"):
            load_csv(write_csv("id,frame,x,y\n1,0,0,0\n1,1,oops,0\n"))

    def test_duplicate_id_in_frame(self):
        with self.assertRaisesRegex(ValueError, "twice in frame 0"):
            load_csv(write_csv("id,frame,x,y\n1,0,0,0\n1,0,1,1\n"))

    def test_empty_id(self):
        with self.assertRaisesRegex(ValueError, "empty id"):
            load_csv(write_csv("id,frame,x,y\n ,0,0,0\n"))

    def test_fps_must_be_positive(self):
        with self.assertRaises(ValueError):
            load_csv(write_csv("id,frame,x,y\n1,0,0,0\n"), fps=0)

    def test_frame_step_is_detected(self):
        # ETH/UCY style: every 10th video frame annotated, 2.5 rows per second -> 0.4 s apart.
        ds = load_csv(write_csv("id,frame,x,y\n1,780,0,0\n1,790,1,0\n2,790,5,5\n2,810,6,5\n"), fps=2.5)
        self.assertEqual(ds.frame_step, 10)
        a0, a1 = ds.tracks["1"]
        self.assertAlmostEqual(a1.t - a0.t, 0.4)
        b0, b1 = ds.tracks["2"]  # a missing row (gap) keeps its real time distance
        self.assertAlmostEqual(b1.t - b0.t, 0.8)

    def test_frame_step_override_and_single_rows(self):
        ds = load_csv(write_csv("id,frame,x,y\n1,0,0,0\n1,10,1,0\n"), fps=2.5, frame_step=5)
        self.assertEqual(ds.frame_step, 5)
        self.assertAlmostEqual(ds.tracks["1"][1].t, 10 / 5 / 2.5)
        self.assertEqual(load_csv(write_csv("id,frame,x,y\n1,7,0,0\n2,9,0,0\n")).frame_step, 1)
        with self.assertRaises(ValueError):
            load_csv(write_csv("id,frame,x,y\n1,0,0,0\n"), frame_step=0)

    def test_float_frame_numbers_are_accepted_if_integral(self):
        ds = load_csv(write_csv("id,frame,x,y\n1,780.0,0,0\n1,790.0,1,1\n"))
        self.assertEqual([s.frame for s in ds.tracks["1"]], [780, 790])
        with self.assertRaisesRegex(ValueError, r":2: bad value.*not an integer"):
            load_csv(write_csv("id,frame,x,y\n1,1.5,0,0\n"))

    def test_non_finite_values_are_rejected(self):
        for bad in ("nan", "inf", "-inf"):
            with self.assertRaisesRegex(ValueError, r":3: bad value"):
                load_csv(write_csv(f"id,frame,x,y\n1,0,0,0\n1,1,{bad},0\n"))

    def test_row_with_extra_fields_does_not_crash(self):
        ds = load_csv(write_csv("id,frame,x,y\n1,0,0,0,surplus,more\n1,1,1,1\n"))
        self.assertEqual(len(ds.tracks["1"]), 2)

    def test_same_time_twice_in_a_track(self):
        with self.assertRaisesRegex(ValueError, r":3: id '1' has two rows at t=0.5"):
            load_csv(write_csv("id,frame,x,y,t\n1,0,0,0,0.5\n1,1,1,1,0.5\n"))

    def test_header_only_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no data rows"):
            load_csv(write_csv("id,frame,x,y\n"))

    def test_tracks_ordered_by_first_time_then_numeric_id(self):
        ds = load_csv(write_csv("id,frame,x,y\n10,0,0,0\nb,1,0,0\n2,0,0,0\na,1,0,0\n"))
        self.assertEqual(list(ds.tracks), ["2", "10", "a", "b"])


class FitToWorld(unittest.TestCase):
    W, H, M = 1200.0, 750.0, 60.0

    def corners(self, bounds, **kw):
        tr = fit_to_world(bounds, (self.W, self.H), self.M, **kw)
        return tr, tr.apply(bounds.xmin, bounds.ymin), tr.apply(bounds.xmax, bounds.ymax)

    def assertInside(self, pt):
        self.assertGreaterEqual(pt[0], self.M - 1e-9)
        self.assertLessEqual(pt[0], self.W - self.M + 1e-9)
        self.assertGreaterEqual(pt[1], self.M - 1e-9)
        self.assertLessEqual(pt[1], self.H - self.M + 1e-9)

    def test_wide_data_fills_width_and_is_centred_vertically(self):
        tr, lo, hi = self.corners(Bounds(0, 0, 100, 10))
        self.assertAlmostEqual(lo[0], self.M)
        self.assertAlmostEqual(hi[0], self.W - self.M)
        self.assertAlmostEqual((lo[1] + hi[1]) / 2, self.H / 2)
        self.assertInside(lo)
        self.assertInside(hi)

    def test_tall_data_fills_height_and_is_centred_horizontally(self):
        tr, lo, hi = self.corners(Bounds(-5, -50, 5, 50))
        self.assertAlmostEqual(lo[1], self.M)
        self.assertAlmostEqual(hi[1], self.H - self.M)
        self.assertAlmostEqual((lo[0] + hi[0]) / 2, self.W / 2)

    def test_aspect_ratio_is_kept(self):
        b = Bounds(3, 7, 43, 37)  # 40 x 30
        tr, lo, hi = self.corners(b)
        self.assertAlmostEqual((hi[0] - lo[0]) / (hi[1] - lo[1]), 40 / 30)

    def test_flip_y_puts_data_top_at_world_top(self):
        b = Bounds(0, 0, 10, 10)
        tr = fit_to_world(b, (self.W, self.H), self.M, flip_y=True)
        top = tr.apply(5, 10)  # data y max = up
        bottom = tr.apply(5, 0)
        self.assertLess(top[1], bottom[1])
        self.assertAlmostEqual(top[1], self.M)

    def test_degenerate_axis_and_single_point(self):
        tr, lo, hi = self.corners(Bounds(0, 5, 10, 5))  # zero height
        self.assertAlmostEqual(lo[1], self.H / 2)
        self.assertAlmostEqual(hi[0] - lo[0], self.W - 2 * self.M)
        tr = fit_to_world(Bounds(2, 2, 2, 2), (self.W, self.H), self.M)
        self.assertEqual(tr.apply(2, 2), (self.W / 2, self.H / 2))

    def test_margin_too_large(self):
        with self.assertRaises(ValueError):
            fit_to_world(Bounds(0, 0, 1, 1), (100, 100), 60)

    def test_to_world_on_a_loaded_dataset(self):
        ds = load_csv(write_csv("id,frame,x,y\na,0,-1,-1\na,1,1,1\nb,0,1,-1\n"))
        tr = fit_to_world(data_bounds(ds), (self.W, self.H), self.M)
        world = to_world(ds, tr)
        pts = [(s.x, s.y) for samples in world.tracks.values() for s in samples]
        for p in pts:
            self.assertInside(p)
        a0, a1 = world.tracks["a"]
        self.assertTrue(math.isclose(a1.x - a0.x, a1.y - a0.y))  # square data stays square
        self.assertEqual([s.t for s in world.tracks["a"]], [s.t for s in ds.tracks["a"]])


if __name__ == "__main__":
    unittest.main()
