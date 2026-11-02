"""Offline tests for the optional data-look layers: social graph (N), long exposure (L), video look (V).

Synthetic tracks and video only; SDL dummy driver, no window.
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
    DATA_BACKGROUND, GRAPH_NEAR_COLOR, DataRenderer, walking_together,
)
from aquarium_boids.replay.video import VideoFrames, import_cv2  # noqa: E402
from aquarium_boids.sources import ReplaySource  # noqa: E402

cv2, _ = import_cv2()
VIEW = View((1200, 750), (1200, 750))


def write_track(rows: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", newline="")
    f.write(rows)
    f.close()
    return Path(f.name)


def walkers(n_rows: int, people: list[tuple[int, float, float, float]]) -> str:
    """ETH/UCY rows: (id, y in m, x speed in m per 0.4 s, x start); frame step 10."""
    return "".join(f"{10 * k}\t{pid}\t{x0 + vx * k:.3f}\t{y:.3f}\n" for pid, y, vx, x0 in people for k in range(n_rows))


class Rule(unittest.TestCase):
    def test_walking_together(self):
        self.assertTrue(walking_together(1.0, (1.2, 0.0), (1.25, 0.05), 1.0))
        self.assertFalse(walking_together(2.5, (1.2, 0.0), (1.2, 0.0), 1.0))  # too far
        self.assertFalse(walking_together(1.0, (1.2, 0.0), (-1.2, 0.0), 1.0))  # opposite ways
        self.assertFalse(walking_together(1.0, (1.2, 0.0), (0.7, 0.0), 1.0))  # speeds differ by 0.5
        self.assertFalse(walking_together(1.0, (0.1, 0.0), (0.12, 0.0), 1.0))  # standing
        self.assertFalse(walking_together(1.0, (1.2, 0.0), (1.0, 1.0), 1.0))  # 45 degrees apart


class Layers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.canvas = pygame.display.set_mode((1200, 750))
        cls.font = pygame.font.SysFont("consolas", 16)
        # 1 and 2 walk together 1 m apart at 1.25 m/s; 3 walks the other way far away; 30 s
        cls.track = write_track(walkers(75, [(1, 2.0, 0.5, 0.0), (2, 3.0, 0.5, 0.0), (3, 9.0, -0.5, 37.0)]))

    @classmethod
    def tearDownClass(cls):
        cls.track.unlink(missing_ok=True)
        pygame.quit()

    def run_for(self, renderer, src, seconds):
        c = RuntimeControls()
        for _ in range(int(60 * seconds)):
            renderer.draw(self.canvas, VIEW, src.step(1.0, c, None, None), None, None, c, {})

    def midpoint(self, src, a, b):
        pa, pb = src.agents[a].position, src.agents[b].position
        return (round((pa.x + pb.x) / 2), round((pa.y + pb.y) / 2))

    def test_defaults_off_and_keys(self):
        r = DataRenderer(self.font, ReplaySource(self.track))
        self.assertEqual((r.graph, r.exposure, r.video_look), (False, False, "normal"))
        self.assertTrue(r.handle_key(pygame.K_n))
        self.assertTrue(r.handle_key(pygame.K_l))
        self.assertTrue(r.graph and r.exposure)
        self.assertFalse(r.handle_key(pygame.K_v))  # no video in the map view: V does nothing

    def test_graph_thin_then_together(self):
        src = ReplaySource(self.track)
        r = DataRenderer(self.font, src)
        r.graph = True
        self.run_for(r, src, 1.0)  # together for < 2 s: a thin line
        self.assertEqual(len(r._together), 1)
        self.assertLess(next(iter(r._together.values())), 2.0)
        thin = self.canvas.get_at(self.midpoint(src, 0, 1))[:3]
        self.assertGreater(sum(thin), sum(DATA_BACKGROUND) + 20)
        self.assertLessEqual(max(thin), max(GRAPH_NEAR_COLOR) + 10)
        self.run_for(r, src, 6.0)  # > 6 s: thick and fully bright
        self.assertGreater(next(iter(r._together.values())), 6.0)
        bright = self.canvas.get_at(self.midpoint(src, 0, 1))[:3]
        self.assertGreater(bright[0], 200)
        self.assertEqual(len(r._together), 1)  # person 3 is never linked

    def test_long_exposure_builds_up_and_fades(self):
        src = ReplaySource(self.track)
        r = DataRenderer(self.font, src)
        r.exposure = True
        self.run_for(r, src, 10.0)
        surf = r._exposure_surface
        x, y = (round(v) for v in src.agents[0].position)
        trail = surf.get_at((x - 60, y))[:3]  # behind person 1 (it walks towards +x)
        self.assertGreater(sum(trail), 0)
        r.exposure = True
        before = sum(trail)
        self.run_for(r, src, 15.0)  # they move on; the old trail fades by ~20 levels per 15 s
        after = sum(surf.get_at((x - 60, y))[:3])
        self.assertLess(after, before)

    def test_standing_adds_no_light(self):
        # person 5 stands at (5, 5) m the whole time (a tiny 1 cm sway); person 1 walks for scale
        rows = walkers(40, [(1, 2.0, 0.5, 0.0)]) + "".join(
            f"{10 * k}\t5\t{5.0 + 0.01 * (k % 2):.3f}\t5.000\n" for k in range(40))
        track = write_track(rows)
        self.addCleanup(track.unlink, True)
        src = ReplaySource(track)
        r = DataRenderer(self.font, src)
        r.exposure = True
        self.run_for(r, src, 12.0)
        standing = next(b for b, p in zip(src.agents, src._pieces) if p.track_id == "5")
        x, y = (round(v) for v in standing.position)
        self.assertEqual(sum(r._exposure_surface.get_at((x, y))[:3]), 0)

    def test_no_random_and_paused_layers_hold(self):
        src = ReplaySource(self.track)
        r = DataRenderer(self.font, src)
        r.graph = r.exposure = True
        random.seed(9)
        state = random.getstate()
        self.run_for(r, src, 3.0)
        self.assertEqual(random.getstate(), state)
        held = dict(r._together)
        c = RuntimeControls()
        c.paused = True
        for _ in range(30):
            r.draw(self.canvas, VIEW, src.step(1.0, c, None, None), None, None, c, {})
        self.assertEqual(r._together, held)  # paused: no data time, timers hold

    def test_restart_resets_timers(self):
        src = ReplaySource(self.track)
        r = DataRenderer(self.font, src)
        r.graph = True
        self.run_for(r, src, 4.0)
        src.reset(0)
        r.draw(self.canvas, VIEW, src.agents, None, None, RuntimeControls(), {})
        self.assertTrue(all(v == 0.0 for v in r._together.values()))


@unittest.skipIf(cv2 is None, "OpenCV not installed (optional, requirements-video.txt)")
class DarkVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "video.avi"
        writer = cv2.VideoWriter(str(cls.path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (96, 64))
        for k in range(30):  # a grey street with a red square moving right 3 px per frame
            img = np.full((64, 96, 3), 140, dtype=np.uint8)
            img[20:36, 10 + 3 * k:26 + 3 * k] = (40, 40, 220)  # BGR red
            writer.write(img)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dark_look_keeps_moving_colour(self):
        v = VideoFrames(self.path, threaded=False)
        self.addCleanup(v.close)
        v.request(10, (96, 64), "dark")
        v.request(11, (96, 64), "dark")
        index, size, look, data = v.latest()
        self.assertEqual((index, look), (11, "dark"))
        px = lambda x, y: data[(y * 96 + x) * 3:(y * 96 + x) * 3 + 3]  # noqa: E731 (RGB)
        street = px(80, 55)
        self.assertLess(max(street), 60)  # 140 grey x 0.32 ~ 45: darker
        self.assertLess(abs(street[0] - street[2]), 6)  # and grey
        square = px(10 + 3 * 11 + 14, 28)  # near the leading edge, where the frame difference sees motion
        self.assertGreater(square[0], 150)  # the moving red square keeps its colour
        v.request(11, (96, 64), "normal")  # same frame, other look: no seek, colour back
        self.assertEqual(v.latest()[2], "normal")
        self.assertEqual(v.seeks, 0)

    def test_bad_look(self):
        v = VideoFrames(self.path, threaded=False)
        self.addCleanup(v.close)
        with self.assertRaises(ValueError):
            v.request(0, (96, 64), "sepia")


if __name__ == "__main__":
    unittest.main()
