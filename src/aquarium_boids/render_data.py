"""Replay "data look": real trajectories drawn like tracking output (--replay, default look).

Neutral dark grey, a metric grid (1 m minor, 5 m major) with a scale bar, and per person a dot
with a heading tick, coloured by speed, plus a thin line through its recorded keyframes of the
last ~10 s. No stars, particles, glow or fish. Read-only like render.py: never writes to the
agents, the controls or the descriptors, never calls the global `random`.
Keys: H toggles the HUD (as in the thermal look), I toggles track id labels.

Camera view (--replay-view camera): the scene's OpenTraj reference frame drawn as is (dimmed), with
the same people, paths, rings and labels projected into it through the frame's homography
(replay/camera.py). Display only: the agents, descriptors and OSC stay in world metres. No grid
or scale bar there (the perspective changes the scale across the frame).
"""
from __future__ import annotations

import math
import zlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pygame
from pygame.math import Vector2

from .controls import RuntimeControls
from .render import View, draw_caption, draw_hud

if TYPE_CHECKING:
    from .boids import Boid
    from .replay.camera import Camera
    from .sources import ReplaySource

Projector = Callable[[float, float], "tuple[float, float] | None"]  # world px -> canvas px (None: not drawable)

DATA_BACKGROUND = (26, 27, 30)
GRID_MINOR = (34, 35, 39)
GRID_MAJOR = (46, 48, 54)
GRID_LABEL = (78, 80, 88)
GRID_MINOR_STEP = 1.0  # data units (metres for ETH/UCY)
GRID_MAJOR_EVERY = 5
TEXT_DIM = (150, 152, 160)
PATH_SECONDS = 10.0
PATH_BRIGHTNESS = 0.55  # path colour = id colour scaled down
LABEL_COLOR = (170, 172, 180)
FOOD_RING = (70, 160, 120)
EVENT_RING = (190, 80, 90)
# Speed colour ramp (0 .. calibrated p95), viridis-like: readable on dark grey, no red/green pair.
SPEED_STOPS = ((72, 40, 120), (49, 104, 142), (31, 158, 137), (110, 206, 88), (253, 231, 37))
ID_HUES = 12
DOT_RADIUS = 3.2  # canvas px at view scale 1
TICK = (4.5, 10.5)  # heading tick from/to, canvas px at view scale 1
NICE_LENGTHS = (1, 2, 5, 10, 20, 50, 100)
CAMERA_DIM = 115  # reference frame brightness x115/255
CAMERA_HEADING_STEP = 0.25  # metres ahead used to find the heading direction in the frame
SCALE_BAR_MAX_PX = 260  # long enough for 5 m (one major grid cell) in every ETH/UCY scene


def _lerp(stops: tuple[tuple[int, int, int], ...], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    a, b = stops[i], stops[i + 1]
    return tuple(int(round(a[c] + (b[c] - a[c]) * f)) for c in range(3))


def speed_color(speed01: float) -> tuple[int, int, int]:
    return _lerp(SPEED_STOPS, speed01)


def id_color(track_id: str) -> tuple[int, int, int]:
    """A muted colour per track id (deterministic, no random)."""
    hue = (zlib.crc32(track_id.encode("utf-8")) % ID_HUES) / ID_HUES
    color = pygame.Color(0)
    color.hsva = (hue * 360.0, 45, 85, 100)
    return (color.r, color.g, color.b)


def nice_length(units_per_px: float, max_px: float = SCALE_BAR_MAX_PX) -> float:
    """Largest 1/2/5 x 10^k length (data units) that fits in max_px."""
    best = NICE_LENGTHS[0]
    for length in NICE_LENGTHS:
        if length / units_per_px <= max_px:
            best = length
    return best


class DataRenderer:
    """Tracking-style look for replay mode. Same draw() signature as the other renderers."""

    MODES = ("art", "data")

    def __init__(self, font: pygame.font.Font, source: "ReplaySource", mode: str = "art",
                 camera: "Camera | None" = None):
        self.font = font
        self.source = source
        self.mode = mode
        self.camera = camera
        self.labels = False
        self._frame: pygame.Surface | None = None
        self._frame_key: tuple | None = None
        self._small: pygame.font.Font | None = None
        self._grid: pygame.Surface | None = None
        self._grid_key: tuple | None = None
        self._label_cache: dict[str, pygame.Surface] = {}

    def handle_key(self, key: int) -> bool:
        if key == pygame.K_h:
            self.mode = "data" if self.mode == "art" else "art"
            return True
        if key == pygame.K_i:
            self.labels = not self.labels
            return True
        return False

    def draw(
        self,
        canvas: pygame.Surface,
        view: View,
        flock: list["Boid"],
        attractor: Vector2 | None,
        repeller: Vector2 | None,
        controls: RuntimeControls,
        descriptors: dict[str, float],
    ) -> None:
        if self._small is None:
            self._small = pygame.font.SysFont("consolas", 13)
        canvas.fill(DATA_BACKGROUND)
        if self.camera is None:
            canvas.blit(self._grid_for(view), view.rect.topleft)
            rect, k = view.rect, view.scale
            ox, oy = rect.topleft

            def project(x: float, y: float) -> tuple[float, float] | None:
                return (ox + x * k, oy + y * k)
        else:
            rect, k, project = self._camera_frame(canvas)
        canvas.set_clip(rect)
        self._draw_agents(canvas, project, k)
        if attractor is not None:
            centre = project(attractor.x, attractor.y)
            if centre is not None:
                pygame.draw.circle(canvas, FOOD_RING, centre, 18 * k, 1)
        if repeller is not None:
            centre = project(repeller.x, repeller.y)
            if centre is not None:
                pygame.draw.circle(canvas, EVENT_RING, centre, 34 * k, 1)
                if repeller is self.source.repeller and self.source.detector is not None:
                    text = self._small.render(f"acceleration event  z {self.source.detector.peak_z:.1f}", True,
                                              EVENT_RING)
                    canvas.blit(text, (centre[0] + 38 * k, centre[1] - text.get_height() / 2))
        canvas.set_clip(None)
        self._draw_scale_and_legend(canvas, rect, None if self.camera else self.source.transform.scale * view.scale)
        if self.mode == "data":
            draw_hud(canvas, self.font, controls, descriptors, origin=rect.topleft,
                     lines=self.source.hud_lines(controls, descriptors))
        extra = f" + {self.camera.credit}" if self.camera is not None else ""
        draw_caption(canvas, self._small, self.source.caption(extra), rect)

    def _draw_agents(self, canvas: pygame.Surface, project: Projector, k: float) -> None:
        """Paths, then dots with heading ticks (and labels), through `project`; k = px per world px."""
        overlay = self.source.overlay(PATH_SECONDS)
        for boid, track_id, path in overlay:  # paths first, under every dot
            points = [q for q in (project(x, y) for x, y in path) if q is not None]
            if len(points) >= 2:
                color = tuple(int(c * PATH_BRIGHTNESS) for c in id_color(track_id))
                pygame.draw.aalines(canvas, color, False, points)
        ref = self.source.calibration.max_speed
        dot = max(2.0, DOT_RADIUS * k)
        step = CAMERA_HEADING_STEP * self.source.transform.scale  # world px
        for boid, track_id, _ in overlay:
            here = project(boid.position.x, boid.position.y)
            if here is None:
                continue
            p = Vector2(here)
            color = speed_color(boid.velocity.length() / ref if ref > 0 else 0.0)
            if boid.velocity.length_squared() > 1e-12:
                ahead_world = boid.position + boid.velocity.normalize() * step
                ahead = project(ahead_world.x, ahead_world.y)
                if ahead is not None and (Vector2(ahead) - p).length_squared() > 1e-12:
                    heading = (Vector2(ahead) - p).normalize()
                    pygame.draw.line(canvas, color, p + heading * TICK[0] * k, p + heading * TICK[1] * k,
                                     max(1, round(1.6 * k)))
            pygame.draw.circle(canvas, color, p, dot)
            if self.labels:
                label = self._label(track_id)
                canvas.blit(label, (p.x + dot + 3, p.y - label.get_height() - 1))

    def _camera_frame(self, canvas: pygame.Surface) -> tuple[pygame.Rect, float, Projector]:
        """Draw the dimmed reference frame fitted into the canvas; returns its rect, scale and projector."""
        camera = self.camera
        key = (canvas.get_size(), camera.image_path)
        if self._frame is None or self._frame_key != key:
            image = pygame.image.load(str(camera.image_path)).convert()
            iw, ih = image.get_size()
            cw, ch = canvas.get_size()
            fit = min(cw / iw, ch / ih)
            frame = pygame.transform.smoothscale(image, (round(iw * fit), round(ih * fit)))
            frame.fill((CAMERA_DIM,) * 3, special_flags=pygame.BLEND_RGB_MULT)
            self._frame, self._frame_key = frame, key
            self._frame_fit = fit
        frame, fit = self._frame, self._frame_fit
        rect = frame.get_rect(center=canvas.get_rect().center)
        canvas.blit(frame, rect)
        to_data = self.source.transform.to_data

        def project(x: float, y: float) -> tuple[float, float] | None:
            pixel = camera.project(*to_data(x, y))
            if pixel is None:
                return None
            return (rect.x + pixel[0] * fit, rect.y + pixel[1] * fit)

        return rect, fit, project

    def _label(self, track_id: str) -> pygame.Surface:
        img = self._label_cache.get(track_id)
        if img is None:
            img = self._small.render(track_id, True, LABEL_COLOR)
            self._label_cache[track_id] = img
        return img

    def _grid_for(self, view: View) -> pygame.Surface:
        """Metric grid over the view, pre-rendered once per view size."""
        tr = self.source.transform
        key = (view.rect.size, view.scale, tr)
        if self._grid is not None and self._grid_key == key:
            return self._grid
        grid = pygame.Surface(view.rect.size).convert()
        grid.fill(DATA_BACKGROUND)
        ww, wh = view.world_size
        xs = [tr.to_data(0, 0)[0], tr.to_data(ww, 0)[0]]
        ys = [tr.to_data(0, 0)[1], tr.to_data(0, wh)[1]]
        s = view.scale
        for major in (False, True):  # majors on top
            color = GRID_MAJOR if major else GRID_MINOR
            for k in range(math.floor(min(xs)), math.ceil(max(xs)) + 1):
                if (k % GRID_MAJOR_EVERY == 0) == major:
                    wx = tr.apply(k * GRID_MINOR_STEP, 0.0)[0] * s
                    pygame.draw.line(grid, color, (wx, 0), (wx, grid.get_height()))
            for k in range(math.floor(min(ys)), math.ceil(max(ys)) + 1):
                if (k % GRID_MAJOR_EVERY == 0) == major:
                    wy = tr.apply(tr.origin_x, k * GRID_MINOR_STEP)[1] * s
                    pygame.draw.line(grid, color, (0, wy), (grid.get_width(), wy))
        self._grid, self._grid_key = grid, key
        return grid

    def _draw_scale_and_legend(self, canvas: pygame.Surface, rect: pygame.Rect, px_per_unit: float | None) -> None:
        """Scale bar (map view: px_per_unit given) and speed legend at the bottom-right of rect."""
        unit = self.source.unit
        right = rect.right - 16
        y = rect.bottom - 18
        if px_per_unit is not None:
            length = nice_length(1.0 / px_per_unit)
            bar = length * px_per_unit
            pygame.draw.line(canvas, TEXT_DIM, (right - bar, y), (right, y), 2)
            for x in (right - bar, right):
                pygame.draw.line(canvas, TEXT_DIM, (x, y - 4), (x, y + 4), 2)
            label = self._small.render(f"{length:g} {unit}", True, TEXT_DIM)
            canvas.blit(label, (right - bar / 2 - label.get_width() / 2, y - label.get_height() - 5))
        # speed legend: ramp 0 .. p95, above the scale bar
        ref = self.source.speed_reference()
        width, top = 120, y - 46
        for i in range(width):
            pygame.draw.line(canvas, speed_color(i / (width - 1)), (right - width + i, top), (right - width + i, top + 6))
        text = self._small.render(f"speed 0 .. {ref:.1f} {unit}/s (p95)", True, TEXT_DIM)
        canvas.blit(text, (right - text.get_width(), top - text.get_height() - 3))
