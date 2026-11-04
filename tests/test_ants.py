"""Offline tests for the ant datasets: converters (tools/convert_ants.py), scene config
(replay/scene_config.py), ReplaySource with a config, places drawn by the data look, pixel camera.

Synthetic files only (no dataset). SDL dummy driver, no window.
Run from the repository root:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ["SDL_VIDEODRIVER"] = "dummy"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import pygame  # noqa: E402

import convert_ants as conv  # noqa: E402
import main  # noqa: E402
from aquarium_boids.controls import RuntimeControls  # noqa: E402
from aquarium_boids.render import View  # noqa: E402
from aquarium_boids.render_data import FOOD_COLORS, NEST_COLOR, DataRenderer  # noqa: E402
from aquarium_boids.replay.camera import IDENTITY, Camera, pixel_camera  # noqa: E402
from aquarium_boids.replay.loader import load_trajectories  # noqa: E402
from aquarium_boids.replay.scene_config import load_scene_config  # noqa: E402
from aquarium_boids.sources import ReplaySource  # noqa: E402


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, data: dict) -> Path:
    return write(path, json.dumps(data))


class AntsConverter(unittest.TestCase):
    def test_box_centre_frame_from_zero_and_time(self):
        rows = conv.ants_rows(["1,7,100,200,64,64,1", "2,7,110.5,210,64,64,1", "", "1,8,10,20,64,64,1"], fps=25)
        self.assertEqual(rows, [("7", 0, 0.0, 132.0, 232.0), ("8", 0, 0.0, 42.0, 52.0), ("7", 1, 0.04, 142.5, 242.0)])
        self.assertEqual(conv.box_size(["1,7,100,200,94,94,1"]), 94.0)

    def test_bad_rows(self):
        with self.assertRaises(ValueError):
            conv.ants_rows(["1,7,100"], fps=25)
        with self.assertRaises(ValueError):
            conv.ants_rows(["0,7,100,200,64,64,1"], fps=25)  # frames start at 1

    def test_sequence_folder_to_csv_and_config(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "Seq0099Object2Image64"
            write(seq / "seqinfo.ini", "[Sequence]\nname=Seq0099Object10Image64\nframeRate=30\nseqLength=3\n")
            write(seq / "gt" / "gt.txt", "1,1,0,0,64,64,1\n2,1,3,0,64,64,1\n3,1,6,0,64,64,1\n1,2,100,100,64,64,1\n"
                                         "2,2,100,103,64,64,1\n")
            out = Path(d) / "out"
            out.mkdir()
            conv.convert_ants(seq, out, video=False)
            data = load_trajectories(out / f"{seq.name}.csv")
            self.assertEqual(sorted(data.tracks), ["1", "2"])
            self.assertAlmostEqual(data.tracks["1"][2].t, 2 / 30)
            self.assertEqual((data.tracks["1"][0].x, data.tracks["1"][0].y), (32.0, 32.0))
            config = load_scene_config(out / f"{seq.name}.scene.json")
            self.assertEqual((config.units, config.body_length, config.agents), ("px", 64.0, "ants"))
            self.assertTrue(config.body_length_is_estimate)
            self.assertIsNone(config.image)  # no img/ folder
            # a second run keeps a hand-edited config
            cfg = out / f"{seq.name}.scene.json"
            cfg.write_text(cfg.read_text(encoding="utf-8").replace('"ants"', '"workers"'), encoding="utf-8")
            conv.convert_ants(seq, out, video=False)
            self.assertEqual(load_scene_config(cfg).agents, "workers")


HEX = "x,y,node,old_node\n" + "".join(  # a small honeycomb strip; node 634 is the nest
    f"{x},{y},{n},{n}\n" for x, y, n in [(0, 990, 1), (0, 1040, 2), (43.3, 1065, 3), (86.6, 1040, 4),
                                         (86.6, 990, 5), (43.3, 1015, 634), (200, 1500, 7)])


class LiquidConverter(unittest.TestCase):
    def test_row_names_dropped(self):
        rows = [["N_ind", "Frame", "Xmm", "Ymm", "Crossings", "Time_sec", "node"],
                ["1", "5", "620", "10.5", "1100", "0", "310", "3"],
                ["2", "5", "621", "11", "1101", "1", "310.5", "3"]]
        self.assertEqual(conv.liquid_rows(rows), [("5", 620, 310.0, 10.5, 1100.0), ("5", 621, 310.5, 11.0, 1101.0)])
        rows[1] = rows[1][1:]  # without row names it works too
        self.assertEqual(conv.liquid_rows(rows)[0], ("5", 620, 310.0, 10.5, 1100.0))
        with self.assertRaises(ValueError):
            conv.liquid_rows([rows[0], ["1", "2"]])
        with self.assertRaises(ValueError):
            conv.liquid_rows([["N_ind", "Frame"], ["1", "2"]])

    def test_maze_edges_arena_only(self):
        nodes = conv.maze_nodes(list(csv.DictReader(HEX.splitlines())))
        edges = conv.maze_edges(nodes)
        # nodes above y 1000: 2, 3, 4, 634 (7 is alone); edges 2-3, 3-4, 2-634, 4-634? distances ~50
        joined = {(e[0], e[1], e[2], e[3]) for e in edges}
        self.assertIn((0.0, 1040.0, 43.3, 1065.0), joined)
        self.assertTrue(all(e[1] > 1000 and e[3] > 1000 for e in edges))
        self.assertTrue(all(abs(((e[2] - e[0]) ** 2 + (e[3] - e[1]) ** 2) ** 0.5 - 50) < 1 for e in edges))
        self.assertEqual(len(edges), len(joined))  # each pair once

    def test_food_items_in_seconds(self):
        rows = [{"exp": "1", "patch": "2", "item": "7", "x": "1453", "y": "1465", "t": "1969"},
                {"exp": "1", "patch": "1", "item": "1", "x": "500", "y": "1465", "t": "NA"},
                {"exp": "2", "patch": "1", "item": "1", "x": "500", "y": "1465", "t": "10"}]
        items = conv.food_items(rows, 1)
        self.assertEqual([(f["patch"], f["t"]) for f in items], [("GP1", None), ("GP2", 984.5)])
        with self.assertRaises(ValueError):
            conv.food_items(rows, 3)

    def test_trial_to_csv_places_and_config(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            trial = write(d / "Food_Exp_3_data.csv",
                          '"N_ind","Frame","Xmm","Ymm","Crossings","Time_sec","node"\n'
                          '"1","4",620,50,1050,0,310,2\n"2","4",621,60,1060,0,310.5,2\n"3","4",622,70,1070,1,311,3\n')
            hex_csv = write(d / "hex.csv", HEX)
            food = write(d / "food.csv", "exp,patch,item,x,y,t\n3,1,1,50,1100,630\n3,2,1,80,1100,NA\n")
            out = d / "out"
            out.mkdir()
            conv.convert_liquid(trial, hex_csv, out, food, 3)
            data = load_trajectories(out / "Food_Exp_3.csv")
            self.assertEqual([s.t for s in data.tracks["4"]], [310.0, 310.5, 311.0])
            config = load_scene_config(out / "Food_Exp_3.scene.json")
            self.assertEqual((config.units, config.body_length, config.flip_y), ("mm", 6.0, True))
            self.assertEqual((config.nest.x, config.nest.y, config.nest.r), (43.3, 1015.0, 60.0))
            self.assertEqual([(f.patch, f.t) for f in config.food], [("GP1", 315.0), ("GP2", None)])
            self.assertTrue(config.maze_edges)
            self.assertTrue(config.event_on("food_discovery") and config.events["arrivals"] == "nest")
            with self.assertRaises(ValueError):
                conv.convert_liquid(trial, hex_csv, out, food, None)


class SceneConfigFile(unittest.TestCase):
    def load(self, data: dict, places: dict | None = None):
        with tempfile.TemporaryDirectory() as d:
            if places is not None:
                write_json(Path(d) / "p.json", places)
                data = {**data, "places_file": "p.json"}
            return load_scene_config(write_json(Path(d) / "c.json", data))

    def test_defaults_and_grid(self):
        c = self.load({"units": "mm", "body_length": 6})
        self.assertEqual((c.agents, c.grid_step, c.standing_bl_s, c.speed_bl_s), ("agents", 60.0, 0.3, (0.5, 5.0)))
        self.assertIsNone(self.load({}).grid_step)
        self.assertEqual(self.load({"grid": 25, "body_length": 6}).grid_step, 25.0)

    def test_nest_from_places_file_with_config_radius(self):
        c = self.load({"body_length": 6, "places": {"nest": {"r": 60}}}, {"nest": {"x": 1, "y": 2, "node": 634}})
        self.assertEqual((c.nest.x, c.nest.y, c.nest.r), (1.0, 2.0, 60.0))
        c = self.load({"body_length": 6}, {"nest": {"x": 1, "y": 2}})
        self.assertEqual(c.nest.r, 60.0)  # default 10 body lengths

    def test_relative_paths(self):
        with tempfile.TemporaryDirectory() as d:
            c = load_scene_config(write_json(Path(d) / "sub" / "c.json", {"image": "f.jpg", "video": "../v.mp4"}))
            self.assertEqual(c.image, Path(d) / "sub" / "f.jpg")
            self.assertEqual(c.video, Path(d) / "sub" / ".." / "v.mp4")

    def test_errors(self):
        bad = [
            {"colour": 1},  # unknown key
            {"thresholds": {"close_bl": [0.5, 3]}},  # thresholds need a body length
            {"body_length": 6, "thresholds": {"speed_bl_s": [5, 1]}},
            {"body_length": 6, "thresholds": {"speed": [1, 5]}},
            {"body_length": -1},
            {"events": {"arrivals": "nest"}},  # no nest
            {"events": {"food_discovery": True}},  # no food times
            {"events": {"dance": True}},
            {"events": {"encounters": "yes"}},
            {"places": {"nest": {"x": 1, "y": 2}}},  # no radius, no body length
            {"places": {"hive": {}}},
        ]
        for data in bad:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load(data)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_scene_config(write(Path(d) / "c.json", "{not json"))
            with self.assertRaises(OSError):
                load_scene_config(Path(d) / "missing.json")


class SourceAndViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((1200, 750))
        cls.font = pygame.font.SysFont("consolas", 16)

    def make(self, d: Path, **config):
        rows = ["id,frame,t,x,y"] + [f"a,{f},{300 + f},{100 + 10 * f},1100" for f in range(20)] + \
            [f"b,{f},{300 + f},{300},{1100 + 20 * f}" for f in range(20)]
        csv_path = write(d / "trial.csv", "\n".join(rows) + "\n")
        places = {"nest": {"x": 200, "y": 1050}, "food": [{"x": 150, "y": 1300, "patch": "GP1", "t": 305}],
                  "maze_edges": [[100, 1050, 300, 1050]]}
        write_json(d / "trial.places.json", places)
        cfg = {"name": "Trial", "citation": "Test 2025", "reference": "Test, long reference", "agents": "ants",
               "units": "mm", "body_length": 6, "flip_y": True, "places_file": "trial.places.json",
               "places": {"nest": {"r": 30}}, **config}
        return ReplaySource(csv_path, events=False, config=load_scene_config(write_json(d / "trial.scene.json", cfg)))

    def test_names_units_flip_and_places(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.make(Path(d))
            self.assertEqual(src.title, "PLAUD-SONIFICATION - Trial - real trajectories (Test 2025)")
            self.assertEqual((src.unit, src.noun, src.grid_step, src.reference), ("mm", "ants", 60.0,
                                                                                 "Test, long reference"))
            self.assertTrue(src.transform.flip_y)
            x, y, r = src.nest_world
            self.assertEqual(src.transform.to_data(x, y), (200.0, 1050.0))
            self.assertAlmostEqual(r, 30 * src.transform.scale)
            self.assertEqual(len(src.maze_world), 1)
            item = src.food_world[0][2]
            self.assertFalse(src.food_found(item))
            c = RuntimeControls()
            for _ in range(60 * 6):  # 6 data seconds
                src.step(1.0, c, None, None)
            self.assertTrue(src.food_found(item))
            lines = src.hud_lines(c, {})
            self.assertTrue(any(line.startswith("ants on screen 2") for line in lines))
            self.assertTrue(any("food found 1/1" in line and "body length 6 mm" in line for line in lines))

    def test_map_draws_nest_and_food(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.make(Path(d))
            r = DataRenderer(self.font, src, mode="art")
            view = View((1200, 750), (1200, 750))
            c = RuntimeControls()
            canvas = pygame.Surface((1200, 750))

            def colours():
                r.draw(canvas, view, src.agents, None, None, c, {})
                fx, fy, _ = src.food_world[0]
                return {tuple(canvas.get_at((round(fx) + dx, round(fy) + dy)))[:3]
                        for dx in range(-4, 5) for dy in range(-4, 5)}

            before = colours()
            self.assertIn(FOOD_COLORS[0], before)
            self.assertNotIn(FOOD_COLORS[1], before)
            for _ in range(60 * 6):
                src.step(1.0, c, None, None)
            self.assertIn(FOOD_COLORS[1], colours())  # filled once found
            nx, ny, nr = src.nest_world
            ring = {tuple(canvas.get_at((round(nx + dx), round(ny - nr + dy))))[:3] for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)}
            self.assertIn(NEST_COLOR, ring)

    def test_pixel_camera(self):
        cam = Camera.pixels("frame.jpg")
        self.assertEqual(cam.to_image, IDENTITY)
        self.assertEqual(cam.project(12.5, 40.0), (12.5, 40.0))
        self.assertEqual(cam.credit, "frame frame.jpg")
        self.assertEqual(pixel_camera(None)[0], None)
        self.assertIn("missing", pixel_camera("nope.jpg")[1])

    def test_main_pixel_views_and_fallbacks(self):
        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "f.png"
            pygame.image.save(pygame.Surface((64, 48)), str(image))
            src = self.make(Path(d), units="px", image="f.png", image_credit="Test frame")
            args = argparse.Namespace(replay_view="camera", classic=False, replay_look="data", replay="x.csv",
                                      replay_camera_dir=None, replay_image=None, replay_video=None)
            camera, video = main.replay_camera(args, src)
            self.assertEqual((camera.image_path, camera.credit, video), (image, "Test frame", None))
            args.replay_view = "video"  # no video: the camera view stays
            camera, video = main.replay_camera(args, src)
            self.assertEqual((camera.image_path, video), (image, None))
            args.replay_video = str(Path(d) / "missing.mp4")
            self.assertIsNone(main.replay_camera(args, src)[1])
            args.replay_view, args.replay_image, args.replay_video = "camera", str(Path(d) / "nope.png"), None
            self.assertEqual(main.replay_camera(args, src), (None, None))  # missing image: map view

    def test_flags_parse(self):
        args = main.parse_args(["--replay", "x.csv", "--replay-config", "c.json", "--replay-image", "i.jpg",
                                "--replay-video", "v.mp4"])
        self.assertEqual((args.replay_config, args.replay_image, args.replay_video), ("c.json", "i.jpg", "v.mp4"))
        args = main.parse_args(["--replay", "x.csv"])
        self.assertEqual((args.replay_config, args.replay_image, args.replay_video), (None, None, None))


if __name__ == "__main__":
    unittest.main()
