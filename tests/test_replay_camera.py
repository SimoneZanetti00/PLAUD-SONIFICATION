"""Offline tests for the replay camera view (replay/camera.py, DataRenderer camera mode).

Synthetic homography and image only (no dataset files). SDL dummy driver, no window.
Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pygame  # noqa: E402

from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.render import View  # noqa: E402
from aquarium_boids.render_data import CAMERA_DIM, DataRenderer  # noqa: E402
from aquarium_boids.replay.camera import (  # noqa: E402
    ENV_DIR, SCENE_CAMERAS, ZARA01_FIX, Camera, find_camera, invert3, read_homography,
)
from aquarium_boids.sources import ReplaySource  # noqa: E402

# image -> world: x = 0.05 * row + 1, y = 0.04 * col - 2  (metres; rows/cols in pixels)
H_IMAGE_TO_WORLD = ((0.05, 0.0, 1.0), (0.0, 0.04, -2.0), (0.0, 0.0, 1.0))


def matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


class Projection(unittest.TestCase):
    def test_invert3(self):
        m = ((2.0, 0.5, 1.0), (0.1, 3.0, -1.0), (0.01, 0.02, 1.0))
        product = matmul(m, invert3(m))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(product[i][j], 1.0 if i == j else 0.0)
        with self.assertRaises(ValueError):
            invert3(((1, 2, 3), (2, 4, 6), (0, 0, 1)))

    def test_world_to_pixel_convention(self):
        cam = Camera.from_homography("x.png", H_IMAGE_TO_WORLD)
        col, row = cam.project(1.0 + 0.05 * 100, -2.0 + 0.04 * 300)  # the world point of row 100, col 300
        self.assertAlmostEqual(row, 100.0)
        self.assertAlmostEqual(col, 300.0)

    def test_point_behind_the_camera(self):
        cam = Camera(Path("x.png"), ((1, 0, 0), (0, 1, 0), (0, 0, -1)))
        self.assertIsNone(cam.project(1.0, 1.0))

    def test_zara01_correction_regression(self):
        # 55e6931 drew zara1 mirrored: the zara01 entry must carry the mirror/x1.2/shift correction,
        # applied after H^-1 and the (row, col) swap; hotel has none.
        self.assertEqual(SCENE_CAMERAS["crowds_zara01"][2], ZARA01_FIX)
        self.assertIsNone(SCENE_CAMERAS["biwi_hotel"][2])
        plain = Camera.from_homography("x.png", H_IMAGE_TO_WORLD)
        fixed = Camera.from_homography("x.png", H_IMAGE_TO_WORLD, ZARA01_FIX)
        for x, y in ((6.0, 10.0), (3.5, 1.0), (12.0, 7.5)):
            col, row = plain.project(x, y)
            fcol, frow = fixed.project(x, y)
            self.assertAlmostEqual(fcol, (719 - col - 360) * 1.2 + 360 + 10)
            self.assertAlmostEqual(frow, (row - 288) * 1.2 + 288 + 40)
        a, b = fixed.project(6.0, 4.0), fixed.project(6.0, 5.0)  # +y in the world moves left in the frame
        self.assertLess(b[0], a[0])

    def test_read_homography(self):
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "H.txt"
            good.write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
            self.assertEqual(read_homography(good)[2], (0.0, 0.0, 1.0))
            bad = Path(d) / "bad.txt"
            bad.write_text("1 2 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 9 numbers"):
                read_homography(bad)


class FindCamera(unittest.TestCase):
    def test_reasons(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_DIR, None)
            cam, why = find_camera("students001.txt", "C:/anything")
            self.assertIsNone(cam)
            self.assertIn("no reference frame known", why)
            cam, why = find_camera("crowds_zara01.txt", None)
            self.assertIn("--replay-camera-dir", why)
            with tempfile.TemporaryDirectory() as d:
                cam, why = find_camera("crowds_zara01.txt", d)
                self.assertIn("missing", why)

    def test_zara01_found_with_its_correction(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "zara01"
            folder.mkdir()
            (folder / "reference.png").write_bytes(b"png")
            (folder / "H-cam.txt").write_text(" ".join(str(v) for row in H_IMAGE_TO_WORLD for v in row),
                                             encoding="utf-8")
            cam, why = find_camera("crowds_zara01.txt", d)
            self.assertIsNotNone(cam, why)
            expected = Camera.from_homography("x.png", H_IMAGE_TO_WORLD, ZARA01_FIX)
            self.assertEqual(cam.project(6.0, 4.0), expected.project(6.0, 4.0))

    def test_found_by_flag_and_by_environment(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "seq_hotel"
            folder.mkdir()
            (folder / "reference.png").write_bytes(b"png")  # existence is enough here
            (folder / "H.txt").write_text(" ".join(str(v) for row in H_IMAGE_TO_WORLD for v in row), encoding="utf-8")
            cam, why = find_camera("C:/data/biwi_hotel.txt", d)
            self.assertIsNotNone(cam, why)
            self.assertEqual(cam.image_path, folder / "reference.png")
            with mock.patch.dict(os.environ, {ENV_DIR: d}):
                self.assertIsNotNone(find_camera("biwi_hotel.txt", None)[0])


class CameraLook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.canvas = pygame.display.set_mode((1200, 750))
        cls.font = pygame.font.SysFont("consolas", 16)
        cls.tmp = tempfile.TemporaryDirectory()
        image = pygame.Surface((400, 320))
        image.fill((200, 100, 50))
        cls.image_path = Path(cls.tmp.name) / "reference.png"
        pygame.image.save(image, str(cls.image_path))
        track = Path(cls.tmp.name) / "biwi_hotel.txt"
        # people inside the synthetic frame: rows 40..280, cols 40..360 -> x 3..15, y -0.4..12.4 m
        track.write_text("".join(f"{10 * k}\t{i}\t{4.0 + 0.3 * k}\t{2.0 + 3.0 * i}\n" for i in range(1, 4)
                                 for k in range(30)), encoding="utf-8")
        cls.track = track

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        pygame.quit()

    def test_draws_frame_people_and_credit_without_random(self):
        src = ReplaySource(self.track)
        camera = Camera.from_homography(self.image_path, H_IMAGE_TO_WORLD)
        renderer = DataRenderer(self.font, src, camera=camera)
        view = View((1200, 750), (1200, 750))
        c = RuntimeControls()
        random.seed(3)
        state = random.getstate()
        for _ in range(20):
            renderer.draw(self.canvas, view, src.step(1.0, c, None, None), None, None, c, {})
        self.assertEqual(random.getstate(), state)
        # the frame is fitted (400x320 -> 937x750, centred) and dimmed
        dimmed = tuple(v * CAMERA_DIM // 255 for v in (200, 100, 50))
        corner = self.canvas.get_at((140, 20))[:3]
        for got, want in zip(corner, dimmed):
            self.assertLessEqual(abs(got - want), 2)
        # a person is drawn where the homography puts it
        boid = src.agents[0]
        x, y = src.transform.to_data(boid.position.x, boid.position.y)
        col, row = camera.project(x, y)
        fit = 750 / 320
        px = (round((1200 - 400 * fit) / 2 + col * fit), round(row * fit))
        self.assertNotEqual(self.canvas.get_at(px)[:3], tuple(dimmed))
        self.assertIn("+ OpenTraj frame (Amirian et al. 2020)", src.caption(" + " + camera.credit))

    def test_map_view_is_the_default(self):
        src = ReplaySource(self.track)
        self.assertIsNone(DataRenderer(self.font, src).camera)
        self.assertEqual(src.caption(), src.caption(""))


if __name__ == "__main__":
    unittest.main()
