"""Offline tests for the replay data look (render_data.py) and its source helpers (SDL dummy driver).

Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pygame  # noqa: E402

from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.render import View  # noqa: E402
from aquarium_boids.render_data import (  # noqa: E402
    DATA_BACKGROUND, DataRenderer, id_color, nice_length, speed_color,
)
from aquarium_boids.replay.loader import Bounds, fit_to_world  # noqa: E402
from aquarium_boids.sources import ReplaySource, scene_citation  # noqa: E402


class Helpers(unittest.TestCase):
    def test_transform_round_trip(self):
        for flip in (False, True):
            tr = fit_to_world(Bounds(-3.0, 2.0, 12.0, 9.5), flip_y=flip)
            for x, y in ((-3.0, 2.0), (4.2, 7.7), (12.0, 9.5)):
                wx, wy = tr.apply(x, y)
                bx, by = tr.to_data(wx, wy)
                self.assertAlmostEqual(bx, x)
                self.assertAlmostEqual(by, y)

    def test_nice_length(self):
        self.assertEqual(nice_length(1 / 49.4, 260), 5)  # zara1: 49 px per metre
        self.assertEqual(nice_length(1 / 49.4, 180), 2)
        self.assertEqual(nice_length(1 / 10.0, 260), 20)
        self.assertEqual(nice_length(1.0, 10), 10)
        self.assertEqual(nice_length(1.0, 0.5), 1)  # nothing fits: the smallest length, never 0

    def test_colours_are_deterministic(self):
        self.assertEqual(id_color("17"), id_color("17"))
        self.assertEqual(speed_color(0.0), (72, 40, 120))
        self.assertEqual(speed_color(1.0), (253, 231, 37))
        self.assertEqual(speed_color(7.0), speed_color(1.0))

    def test_citations(self):
        self.assertEqual(scene_citation("crowds_zara01.txt", "auto")[0], "UCY: Lerner et al. 2007")
        self.assertEqual(scene_citation("biwi_hotel.txt", "auto")[0], "ETH: Pellegrini et al. 2009")
        self.assertIn("/", scene_citation("other.txt", "auto")[0])
        self.assertEqual(scene_citation("walk.csv", "auto")[0], "CSV")


class Look(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.canvas = pygame.display.set_mode((1200, 750))
        cls.font = pygame.font.SysFont("consolas", 16)
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", newline="")
        # three walkers, 0.5 m per 0.4 s, 30 s of data, in "crowds_zara01" style
        f.write("".join(f"{10 * k}\t{i}\t{0.5 * k}\t{2.0 * i}\n" for i in range(1, 4) for k in range(75)))
        f.close()
        cls.path = Path(f.name)

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)
        pygame.quit()

    def test_overlay_paths_cover_the_last_seconds(self):
        src = ReplaySource(self.path)
        c = RuntimeControls()
        for _ in range(60 * 15):  # 15 s of data
            src.step(1.0, c, None, None)
        overlay = src.overlay(10.0)
        self.assertEqual(len(overlay), 3)
        boid, track_id, path = overlay[0]
        self.assertEqual(path[-1], (boid.position.x, boid.position.y))  # ends at the agent
        self.assertEqual(len(path), 25 + 1)  # 10 s of 0.4 s keyframes, plus the current position
        self.assertLessEqual(len(src.overlay(100.0)[0][2]), 15 / 0.4 + 2)  # never more than what was recorded

    def test_speed_reference_in_metres_per_second(self):
        src = ReplaySource(self.path)
        self.assertAlmostEqual(src.speed_reference(), 0.5 / 0.4, places=3)

    def test_hud_and_caption(self):
        src = ReplaySource(self.path)
        c = RuntimeControls()
        text = "\n".join(src.hud_lines(c, {}))
        self.assertIn("source: ", text)
        self.assertNotIn("density_fader", text)
        self.assertNotIn("section", text)
        self.assertNotIn("MIDI", text)
        self.assertIn("real trajectories (ETH/UCY: ", src.caption())

    def test_draws_without_random_and_toggles(self):
        src = ReplaySource(self.path)
        renderer = DataRenderer(self.font, src, mode="art")
        view = View((1200, 750), (1200, 750))
        c = RuntimeControls()
        random.seed(5)
        state = random.getstate()
        for _ in range(30):
            agents = src.step(1.0, c, None, None)
            renderer.draw(self.canvas, view, agents, None, None, c, {})
        self.assertEqual(random.getstate(), state)
        self.assertEqual(self.canvas.get_at((600, 5))[:3], DATA_BACKGROUND)  # art mode: no HUD text up top
        self.assertTrue(renderer.handle_key(pygame.K_i))
        self.assertTrue(renderer.labels)
        self.assertTrue(renderer.handle_key(pygame.K_h))
        self.assertEqual(renderer.mode, "data")
        self.assertFalse(renderer.handle_key(pygame.K_t))  # trails/glow/particles keys do nothing here
        renderer.draw(self.canvas, view, src.agents, None, pygame.math.Vector2(600, 375), c, {})

    def test_scaled_window(self):
        src = ReplaySource(self.path)
        renderer = DataRenderer(self.font, src)
        canvas = pygame.Surface((1920, 1080))
        renderer.draw(canvas, View((1200, 750), (1920, 1080)), src.agents, None, None, RuntimeControls(), {})


if __name__ == "__main__":
    unittest.main()
