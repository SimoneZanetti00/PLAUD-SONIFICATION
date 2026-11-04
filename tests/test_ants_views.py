"""Offline tests for the ant views: video read-ahead and precomputed dark look (replay/video.py),
the dark video written by tools/convert_ants.py, and the N social graph in body lengths.

Synthetic files only. SDL dummy driver, no window. The video tests are skipped without OpenCV.
Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import pygame  # noqa: E402

import convert_ants as conv  # noqa: E402
from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.render import View  # noqa: E402
from aquarium_boids.render_data import GRAPH_NEAR_COLOR, GRAPH_TOGETHER_COLORS, DataRenderer  # noqa: E402
from aquarium_boids.replay.ant_mapping import ant_together  # noqa: E402
from aquarium_boids.replay.scene_config import load_scene_config  # noqa: E402
from aquarium_boids.replay.video import VideoFrames, import_cv2  # noqa: E402
from aquarium_boids.sources import ReplaySource  # noqa: E402

cv2, _ = import_cv2()
FRAMES, W, H = 40, 64, 48


def make_video(path: Path, value=lambda k: 20 + 3 * k) -> None:
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (W, H))
    for k in range(FRAMES):
        writer.write(np.full((H, W, 3), value(k), dtype=np.uint8))
    writer.release()


def wait(video: VideoFrames, index: int, size, look="normal", timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        got = video.latest()
        if got is not None and got[:3] == (index, size, look):
            return got
        time.sleep(0.005)
    raise AssertionError(f"frame {index} not decoded")


@unittest.skipIf(cv2 is None, "OpenCV not installed (optional, requirements-video.txt)")
class ReadAhead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "v.avi"
        make_video(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_frames_ahead_are_ready_at_once_and_wrap_with_loop(self):
        v = VideoFrames(self.path, ahead=4, loop=True)
        try:
            v.request(10, (W, H))
            wait(v, 10, (W, H))
            end = time.time() + 5
            while time.time() < end and len(v._cache) < 5:
                time.sleep(0.005)
            self.assertEqual(sorted(k[0] for k in v._cache), [10, 11, 12, 13, 14])
            v.request(12, (W, H))  # in the cache: served by request() itself
            self.assertEqual(v.latest()[0], 12)
            self.assertEqual(v.cache_hits, 1)
            v.request(FRAMES - 2, (W, H))
            wait(v, FRAMES - 2, (W, H))
            end = time.time() + 5
            while time.time() < end and (0, (W, H), "normal") not in v._cache:
                time.sleep(0.005)
            self.assertIn((0, (W, H), "normal"), v._cache)  # loop: frame 0 is read ahead before the wrap
            self.assertLessEqual(len(v._cache), 5)  # only the window ahead of the wanted frame
        finally:
            v.close()

    def test_off_by_default_and_without_loop_stops_at_the_end(self):
        v = VideoFrames(self.path)
        try:
            v.request(3, (W, H))
            wait(v, 3, (W, H))
            time.sleep(0.05)
            self.assertEqual(v._cache, {})
        finally:
            v.close()
        v = VideoFrames(self.path, ahead=4)
        try:
            v.request(FRAMES - 2, (W, H))
            wait(v, FRAMES - 2, (W, H))
            time.sleep(0.1)
            self.assertEqual(sorted(k[0] for k in v._cache), [FRAMES - 2, FRAMES - 1])
        finally:
            v.close()

    def test_dark_look_from_a_second_video(self):
        dark = Path(self.tmp.name) / "d.avi"
        make_video(dark, value=lambda k: 200)
        v = VideoFrames(self.path, threaded=False, dark_path=dark)
        try:
            v.request(5, (W, H), "dark")
            self.assertGreater(v.latest()[3][0], 180)  # read from the dark video, not computed
            v.request(5, (W, H), "normal")
            self.assertLess(v.latest()[3][0], 60)
        finally:
            v.close()

    def test_converter_dark_video_matches_the_live_look(self):
        out = Path(self.tmp.name) / "v.dark.mp4"
        n = conv.build_dark_video(self.path, 25.0, out)
        self.assertEqual(n, FRAMES)
        live = VideoFrames(self.path, threaded=False)
        baked = VideoFrames(self.path, threaded=False, dark_path=out)
        try:
            for i in range(3):  # consecutive frames, like the converter
                live.request(i, (W, H), "dark")
                baked.request(i, (W, H), "dark")
                a, b = live.latest()[3], baked.latest()[3]
                self.assertLess(max(abs(x - y) for x, y in zip(a[:300], b[:300])), 12)  # mp4 is lossy
        finally:
            live.close()
            baked.close()


class AntGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((1200, 750))
        cls.font = pygame.font.SysFont("consolas", 16)

    def test_rule_in_body_lengths(self):
        self.assertTrue(ant_together(2.0, (2.0, 0.0), (2.3, 0.2), 1.0))
        self.assertFalse(ant_together(3.5, (2.0, 0.0), (2.0, 0.0), 1.0))  # too far
        self.assertFalse(ant_together(1.0, (0.5, 0.0), (2.0, 0.0), 1.0))  # one standing
        self.assertFalse(ant_together(1.0, (2.0, 0.0), (0.0, 2.0), 1.0))  # other way
        self.assertFalse(ant_together(1.0, (1.2, 0.0), (3.0, 0.0), 1.0))  # different pace

    def test_graph_connects_ants_walking_together_and_not_far_ones(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = ["id,frame,t,x,y"]
            for f in range(40):  # a and b side by side, 1 BL (6 mm) apart, 2 BL/s; c far away
                t = f * 0.25
                rows += [f"a,{f},{t},{12 * t},100", f"b,{f},{t},{12 * t},106", f"c,{f},{t},{12 * t},400"]
            (d / "t.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            (d / "t.json").write_text(json.dumps({"units": "mm", "body_length": 6}), encoding="utf-8")
            src = ReplaySource(d / "t.csv", events=False, config=load_scene_config(d / "t.json"))
            r = DataRenderer(self.font, src, mode="art")
            r.graph = True
            view = View((1200, 750), (1200, 750))
            c = RuntimeControls()
            canvas = pygame.Surface((1200, 750))
            colours = set()
            for _ in range(60 * 4):  # 4 data seconds
                r.draw(canvas, view, src.step(1.0, c, None, None), None, None, c, {})
            a, b, cc = (next(bb for bb, p in zip(src.agents, src._pieces) if p.track_id == k) for k in "abc")
            mid = ((a.position.x + b.position.x) / 2, (a.position.y + b.position.y) / 2)
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    colours.add(tuple(canvas.get_at((round(mid[0]) + dx, round(mid[1]) + dy)))[:3])
            self.assertIn(GRAPH_TOGETHER_COLORS[1], colours)  # walking together for > 2 s: full line
            self.assertGreater(len(r._together), 0)
            self.assertEqual([k for k in r._together if "c" in k], [])
            far = {tuple(canvas.get_at((round((a.position.x + cc.position.x) / 2), round((a.position.y + cc.position.y) / 2))))[:3]}
            self.assertNotIn(GRAPH_NEAR_COLOR, far)


if __name__ == "__main__":
    unittest.main()
