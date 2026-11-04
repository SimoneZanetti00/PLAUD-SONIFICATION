"""Offline tests for the ant views: video read-ahead and precomputed dark look (replay/video.py),
and the dark video written by tools/convert_ants.py.

Synthetic files only. SDL dummy driver, no window. The video tests are skipped without OpenCV.
Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

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

import convert_ants as conv  # noqa: E402
from aquarium_boids.replay.video import VideoFrames, import_cv2  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
