"""Offline tests for the replay look: title, caption, people sprites (SDL dummy driver, no window).

Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pygame  # noqa: E402

from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.render import ClassicRenderer, PeopleSprites, ThermalRenderer, View  # noqa: E402
from aquarium_boids.sources import ReplaySource, mmss, scene_title  # noqa: E402


class Titles(unittest.TestCase):
    def test_eth_ucy_scene_names(self):
        self.assertEqual(scene_title("C:/x/crowds_zara01.txt", "auto"),
                         "PLAUD-SONIFICATION - zara1 - real trajectories (ETH/UCY)")
        self.assertEqual(scene_title("biwi_hotel.txt", "eth-ucy"), "PLAUD-SONIFICATION - hotel - real trajectories (ETH/UCY)")
        self.assertEqual(scene_title("my_scene.txt", "auto"), "PLAUD-SONIFICATION - my_scene - real trajectories (ETH/UCY)")
        self.assertEqual(scene_title("walk.csv", "auto"), "PLAUD-SONIFICATION - walk - real trajectories (CSV)")

    def test_mmss(self):
        self.assertEqual(mmss(0), "00:00")
        self.assertEqual(mmss(83.9), "01:23")
        self.assertEqual(mmss(722.4), "12:02")
        self.assertEqual(mmss(-1), "00:00")


class Look(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.canvas = pygame.display.set_mode((1200, 750))
        cls.font = pygame.font.SysFont("consolas", 16)
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", newline="")
        f.write("".join(f"{10 * k}\t{i}\t{0.5 * k + i}\t{i}\n" for i in range(1, 4) for k in range(30)))
        f.close()
        cls.path = Path(f.name)

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)
        pygame.quit()

    def test_caption_tracks_data_time(self):
        src = ReplaySource(self.path)
        self.assertTrue(src.caption().endswith("00:00 / 00:11"))
        c = RuntimeControls()
        for _ in range(60 * 5):
            src.step(1.0, c, None, None)
        self.assertTrue(src.caption().endswith("00:05 / 00:11"), src.caption())

    def test_people_sprites_layout_matches_fish(self):
        sprites = PeopleSprites(1.0)
        self.assertEqual(len(sprites.body), 3)
        self.assertEqual(len(sprites.body[0]), 8)
        self.assertEqual(len(sprites.body[0][0]), 64)
        self.assertEqual(len(sprites.halo[2][7]), 64)

    def test_renderers_draw_people_and_caption(self):
        src = ReplaySource(self.path)
        view = View((1200, 750), (1200, 750))
        c = RuntimeControls()
        for renderer in (ThermalRenderer(self.font, people=True, caption=src.caption),
                         ClassicRenderer(self.font, people=True, caption=src.caption)):
            for _ in range(3):
                agents = src.step(1.0, c, None, None)
                renderer.draw(self.canvas, view, agents, None, None, c, {"energy": 0.2, "mean_speed": 0.4})
            # the caption sits at the bottom-left: some pixels there are lit
            corner = self.canvas.subsurface(pygame.Rect(10, 725, 300, 20))
            self.assertGreater(max(corner.get_at((x, y))[0] for x in range(0, 300, 3) for y in range(0, 20, 2)), 60)

    def test_replay_hud_text(self):
        src = ReplaySource(self.path)
        c = RuntimeControls()
        src.step(1.0, c, None, None)
        lines = src.hud_lines(c, {"fish_count": 0.03, "mean_speed": 0.5})
        text = "\n".join(lines)
        self.assertIn("PLAUD-SONIFICATION | ", lines[0])
        self.assertIn("real trajectories (ETH/UCY)", lines[0])
        self.assertIn("data time 00:00 / 00:11 | playing", text)
        self.assertIn("people on screen 3 (most at once in the file 3)", text)
        self.assertNotIn("MIDI", text)
        self.assertNotIn("align", text)  # boid weights have no effect in replay
        c.paused = True
        self.assertIn("| paused |", "\n".join(src.hud_lines(c, {})))

    def test_renderers_use_the_replay_hud(self):
        src = ReplaySource(self.path)
        seen = []
        hud = lambda c, d: seen.append(1) or src.hud_lines(c, d)  # noqa: E731
        view = View((1200, 750), (1200, 750))
        c = RuntimeControls()
        ClassicRenderer(self.font, people=True, hud=hud).draw(self.canvas, view, src.agents, None, None, c, {})
        ThermalRenderer(self.font, mode="art", people=True, hud=hud).draw(self.canvas, view, src.agents, None, None, c, {})
        ThermalRenderer(self.font, mode="data", people=True, hud=hud).draw(self.canvas, view, src.agents, None, None, c, {})
        self.assertEqual(len(seen), 2)  # classic (always) + thermal data mode; not thermal art mode

    def test_defaults_stay_fish_without_caption(self):
        self.assertFalse(ThermalRenderer(self.font).people)
        self.assertIsNone(ThermalRenderer(self.font).caption)
        self.assertFalse(ClassicRenderer(self.font).people)
        self.assertIsNone(ClassicRenderer(self.font).caption)
        self.assertIsNone(ThermalRenderer(self.font).hud)
        self.assertIsNone(ClassicRenderer(self.font).hud)


if __name__ == "__main__":
    unittest.main()
