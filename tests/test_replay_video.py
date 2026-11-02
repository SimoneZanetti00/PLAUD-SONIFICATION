"""Offline tests for the replay video view (replay/video.py, DataRenderer + main fallbacks).

A tiny synthetic video is written with OpenCV in a temp folder (no dataset files). Skipped where
OpenCV is not installed (it is optional: requirements-video.txt); the missing-OpenCV fallback is
tested everywhere. SDL dummy driver, no window.
Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pygame  # noqa: E402

from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.render import View  # noqa: E402
from aquarium_boids.render_data import DataRenderer  # noqa: E402
from aquarium_boids.replay.camera import Camera, find_video  # noqa: E402
from aquarium_boids.replay.video import VideoFrames, import_cv2  # noqa: E402
from aquarium_boids.sources import ReplaySource  # noqa: E402

cv2, _ = import_cv2()
FRAMES, W, H = 60, 64, 48
H_IMAGE_TO_WORLD = ((0.05, 0.0, 1.0), (0.0, 0.04, -2.0), (0.0, 0.0, 1.0))


def grey(k: int) -> int:
    return 20 + 3 * k  # frame k is a flat grey of this value


def make_video(path: Path) -> None:
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (W, H))
    for k in range(FRAMES):
        writer.write(np.full((H, W, 3), grey(k), dtype=np.uint8))
    writer.release()


def first_pixel(data: bytes) -> int:
    return data[0]


@unittest.skipIf(cv2 is None, "OpenCV not installed (optional, requirements-video.txt)")
class Frames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "video.avi"
        make_video(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_frame_for_data_time(self):
        v = VideoFrames(self.path, threaded=False)
        self.assertEqual((v.fps, v.count, v.size), (25.0, FRAMES, (W, H)))
        self.assertEqual(v.frame_for(0.0), 0)
        self.assertEqual(v.frame_for(1.0), 25)
        self.assertEqual(v.frame_for(0.98), 24)  # round(24.5) -> 24 (banker's), 0.99 -> 25
        self.assertEqual(v.frame_for(99.0), FRAMES - 1)  # clamped
        self.assertEqual(v.frame_for(-1.0), 0)
        v.close()

    def test_right_frame_sequential_and_seek(self):
        v = VideoFrames(self.path, threaded=False)
        for k in (3, 4, 10, 40):  # forward: decoded in order, no seek
            v.request(k, (W, H))
            index, size, look, data = v.latest()
            self.assertEqual((index, size, look), (k, (W, H), "normal"))
            self.assertAlmostEqual(first_pixel(data), grey(k), delta=3)
        self.assertEqual(v.seeks, 0)
        v.request(2, (W, H))  # backwards (R restart, loop wrap): a seek
        self.assertEqual(v.seeks, 1)
        self.assertAlmostEqual(first_pixel(v.latest()[3]), grey(2), delta=3)
        v.request(2, (W, H))  # same request again: nothing to do
        self.assertEqual(v.seeks, 1)
        v.close()

    def test_scaled_and_dimmed(self):
        v = VideoFrames(self.path, dim=128, threaded=False)
        v.request(30, (32, 24))
        index, size, look, data = v.latest()
        self.assertEqual(size, (32, 24))
        self.assertEqual(len(data), 32 * 24 * 3)
        self.assertAlmostEqual(first_pixel(data), grey(30) * 128 / 255, delta=3)
        v.close()

    def test_background_thread(self):
        v = VideoFrames(self.path)
        self.assertIsNone(v.latest())
        v.request(20, (W, H))
        deadline = time.time() + 5.0
        while (v.latest() is None or v.latest()[0] != 20) and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(v.latest()[0], 20)
        self.assertAlmostEqual(first_pixel(v.latest()[3]), grey(20), delta=3)
        v.close()


@unittest.skipIf(cv2 is None, "OpenCV not installed (optional, requirements-video.txt)")
class VideoLook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.canvas = pygame.display.set_mode((1200, 750))
        cls.font = pygame.font.SysFont("consolas", 16)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video_path = Path(cls.tmp.name) / "video.avi"
        make_video(cls.video_path)
        image = pygame.Surface((W, H))
        image.fill((0, 0, 0))
        cls.image_path = Path(cls.tmp.name) / "reference.png"
        pygame.image.save(image, str(cls.image_path))
        cls.track = Path(cls.tmp.name) / "biwi_hotel.txt"
        cls.track.write_text("".join(f"{10 * k}\t1\t{2.0 + 0.05 * k}\t{0.0 + 0.04 * k}\n" for k in range(20)),
                             encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        pygame.quit()

    def test_draws_the_frame_of_the_data_time(self):
        src = ReplaySource(self.track)
        camera = Camera.from_homography(self.image_path, H_IMAGE_TO_WORLD)
        video = VideoFrames(self.video_path, threaded=False)
        renderer = DataRenderer(self.font, src, camera=camera, video=video)
        view = View((1200, 750), (1200, 750))
        c = RuntimeControls()
        for _ in range(60):  # 1 s of data -> video frame 25
            src.step(1.0, c, None, None)
        renderer.draw(self.canvas, view, src.agents, None, None, c, {})
        self.assertEqual(video.latest()[0], video.frame_for(src.clock.t))
        self.assertEqual(video.latest()[0], 25)
        corner = self.canvas.get_at((self.canvas.get_width() // 2, 5))[0]  # top of the fitted frame
        self.assertAlmostEqual(corner, grey(25), delta=4)  # dim 255 here: the raw grey
        c.paused = True  # pause: the frame stays
        renderer.draw(self.canvas, view, src.step(1.0, c, None, None), None, None, c, {})
        self.assertEqual(video.latest()[0], 25)
        src.reset(c.population)  # R: back to frame 0
        renderer.draw(self.canvas, view, src.agents, None, None, c, {})
        self.assertEqual(video.latest()[0], 0)
        self.assertIn("OpenTraj video (Amirian et al. 2020)", self._caption(renderer, src))
        video.close()

    @staticmethod
    def _caption(renderer, src):
        return src.caption(" + OpenTraj video (Amirian et al. 2020)") if renderer.video else src.caption()


class Fallbacks(unittest.TestCase):
    def test_missing_opencv(self):
        with mock.patch.dict(sys.modules, {"cv2": None}):
            module, why = import_cv2()
            self.assertIsNone(module)
            self.assertIn("requirements-video.txt", why)
            with self.assertRaisesRegex(RuntimeError, "needs OpenCV"):
                VideoFrames("whatever.avi")

    def test_find_video(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("missing", find_video("biwi_hotel.txt", d)[1])
            (Path(d) / "seq_hotel").mkdir()
            (Path(d) / "seq_hotel" / "video.avi").write_bytes(b"x")
            self.assertEqual(find_video("biwi_hotel.txt", d)[0], Path(d) / "seq_hotel" / "video.avi")
            self.assertIsNone(find_video("students001.txt", d)[0])

    def test_main_falls_back_to_the_camera_view(self):
        import main  # src/main.py
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "seq_hotel"
            folder.mkdir()
            (folder / "reference.png").write_bytes(b"png")
            (folder / "H.txt").write_text(" ".join(str(v) for row in H_IMAGE_TO_WORLD for v in row), encoding="utf-8")
            (folder / "video.avi").write_bytes(b"x")
            args = argparse.Namespace(replay=str(Path(d) / "biwi_hotel.txt"), replay_view="video",
                                      replay_camera_dir=d, classic=False, replay_look="data")
            out = io.StringIO()
            with mock.patch.dict(sys.modules, {"cv2": None}), contextlib.redirect_stdout(out):
                camera, video = main.replay_camera(args)
            self.assertIsNotNone(camera)
            self.assertIsNone(video)
            self.assertIn("video view not available", out.getvalue())
            self.assertIn("using the camera view", out.getvalue())
            args.replay_view = "map"
            self.assertEqual(main.replay_camera(args), (None, None))


if __name__ == "__main__":
    unittest.main()
