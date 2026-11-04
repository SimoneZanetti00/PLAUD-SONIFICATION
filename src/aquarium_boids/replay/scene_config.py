"""Per-dataset scene config (--replay-config FILE.json): what the replay needs to know about a
dataset that is not pedestrians in metres. Pure Python, no pygame.

The file is kept with the data, not in this repository (an example: docs/ANTS.md). Without it the
replay behaves as before (pedestrians, metres, ETH/UCY names and citations).

Read here:
- name, citation (short, for the caption), reference (long, for the HUD), agents ("ants");
- units of x and y ("mm", "px", "m"), body_length (in those units) and whether it is an estimate;
- flip_y (data y points up), grid (minor grid step in data units, map view);
- smooth_s: a centred moving average of the positions over this many seconds before the spline
  (dense, jittery tracking such as ANTS box centres); path_s: seconds of path drawn behind each one;
- thresholds in body lengths: standing (BL/s), speed range (BL/s), closeness range (BL), and the
  centre distance of an encounter (BL);
- places: the nest (x, y, r) and food items (x, y, patch, discovery time t in data seconds), and
  maze edges to draw, from an optional places_file (written by tools/convert_ants.py) and/or the
  config's own "places" (its nest keys, e.g. only r, win over the file's; its food list replaces it);
- events: which events apply to this dataset (used by the ant sound mapping, next step);
- image / video: a still frame and a video in the data's pixel coordinates (no homography), for
  the camera / video views; --replay-image / --replay-video override them. video_dark: the same
  video with the dark look already applied (tools/convert_ants.py), read for the V dark look;
Relative paths are relative to the config file.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

EVENTS = ("arrivals", "together", "food_discovery", "recruitment", "encounters")
ARRIVALS = ("anywhere", "nest", "off")
DEFAULT_THRESHOLDS = {"standing_bl_s": 1.0, "speed_bl_s": (1.0, 4.0), "close_bl": (0.5, 3.0), "encounter_bl": 1.0}
DEFAULT_NEST_RADIUS_BL = 10.0  # nest radius when none is given, in body lengths
DEFAULT_GRID_BL = 10.0  # map grid step when none is given, in body lengths
KEYS = {"name", "citation", "reference", "agents", "units", "body_length", "body_length_is_estimate",
        "flip_y", "grid", "thresholds", "places_file", "places", "events", "image", "video",
        "image_credit", "video_credit", "video_dark", "smooth_s", "path_s", "comment"}


@dataclass(frozen=True)
class Nest:
    x: float
    y: float
    r: float  # data units


@dataclass(frozen=True)
class FoodItem:
    x: float
    y: float
    patch: str
    t: float | None  # data seconds of discovery, None = never found (or unknown)


@dataclass(frozen=True)
class SceneConfig:
    path: Path
    name: str | None = None
    citation: str | None = None
    reference: str | None = None
    agents: str = "agents"
    units: str = "unit"
    body_length: float | None = None
    body_length_is_estimate: bool = False
    flip_y: bool = False
    grid: float | None = None
    smooth_s: float | None = None
    path_s: float | None = None
    standing_bl_s: float = DEFAULT_THRESHOLDS["standing_bl_s"]
    speed_bl_s: tuple[float, float] = DEFAULT_THRESHOLDS["speed_bl_s"]
    close_bl: tuple[float, float] = DEFAULT_THRESHOLDS["close_bl"]
    encounter_bl: float = DEFAULT_THRESHOLDS["encounter_bl"]
    nest: Nest | None = None
    food: tuple[FoodItem, ...] = ()
    maze_edges: tuple[tuple[float, float, float, float], ...] = ()
    events: dict[str, object] = field(default_factory=dict)
    image: Path | None = None
    video: Path | None = None
    video_dark: Path | None = None
    image_credit: str | None = None
    video_credit: str | None = None

    @property
    def grid_step(self) -> float | None:
        """Minor grid step of the map view (data units): the config's, else 10 body lengths."""
        if self.grid is not None:
            return self.grid
        return DEFAULT_GRID_BL * self.body_length if self.body_length else None

    def event_on(self, name: str) -> bool:
        value = self.events.get(name, False)
        return value not in (False, "off", None)


def _number(value, what: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} must be a number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{what} must be > 0, got {value}")
    return float(value)


def _pair(value, what: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{what} must be [low, high], got {value!r}")
    lo, hi = (_number(v, what) for v in value)
    if not 0 <= lo < hi:
        raise ValueError(f"{what} must be 0 <= low < high, got {value!r}")
    return lo, hi


def _text(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty text, got {value!r}")
    return value.strip()


def _nest(value, what: str, default_r: float | None) -> Nest:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be an object with x, y (and r), got {value!r}")
    r = value.get("r", default_r)
    if r is None:
        raise ValueError(f"{what}: no radius r (and no body_length to take the default from)")
    return Nest(_number(value.get("x"), f"{what}.x"), _number(value.get("y"), f"{what}.y"),
                _number(r, f"{what}.r", positive=True))


def _food(items, what: str) -> tuple[FoodItem, ...]:
    if not isinstance(items, list):
        raise ValueError(f"{what} must be a list, got {items!r}")
    out = []
    for i, item in enumerate(items):
        where = f"{what}[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object, got {item!r}")
        t = item.get("t")
        out.append(FoodItem(_number(item.get("x"), f"{where}.x"), _number(item.get("y"), f"{where}.y"),
                            str(item.get("patch", "")), None if t is None else _number(t, f"{where}.t")))
    return tuple(out)


def _edges(edges, what: str) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(edges, list):
        raise ValueError(f"{what} must be a list of [x1, y1, x2, y2], got {edges!r}")
    out = []
    for i, e in enumerate(edges):
        if not isinstance(e, (list, tuple)) or len(e) != 4:
            raise ValueError(f"{what}[{i}] must be [x1, y1, x2, y2], got {e!r}")
        out.append(tuple(_number(v, f"{what}[{i}]") for v in e))
    return tuple(out)  # type: ignore[return-value]


def _read_json(path: Path, what: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} {path.name}: not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise ValueError(f"{what} {path.name}: expected a JSON object")
    return data


def load_scene_config(path: str | Path) -> SceneConfig:
    """Read and validate a scene config. Raises ValueError (what is wrong) or OSError (missing file)."""
    path = Path(path)
    data = _read_json(path, "scene config")
    unknown = sorted(set(data) - KEYS)
    if unknown:
        raise ValueError(f"scene config {path.name}: unknown key(s) {', '.join(unknown)} (known: {', '.join(sorted(KEYS))})")
    base = path.parent

    def rel(value, what: str) -> Path | None:
        return None if value is None else base / _text(value, what)

    kw: dict = {"path": path}
    for key in ("name", "citation", "reference", "agents", "units", "image_credit", "video_credit"):
        if data.get(key) is not None:
            kw[key] = _text(data[key], key)
    bl = data.get("body_length")
    if bl is not None:
        kw["body_length"] = bl = _number(bl, "body_length", positive=True)
    kw["body_length_is_estimate"] = bool(data.get("body_length_is_estimate", False))
    kw["flip_y"] = bool(data.get("flip_y", False))
    for key in ("grid", "smooth_s", "path_s"):
        if data.get(key) is not None:
            kw[key] = _number(data[key], key, positive=True)

    th = data.get("thresholds", {})
    if not isinstance(th, dict):
        raise ValueError(f"thresholds must be an object, got {th!r}")
    bad = sorted(set(th) - set(DEFAULT_THRESHOLDS))
    if bad:
        raise ValueError(f"thresholds: unknown key(s) {', '.join(bad)} (known: {', '.join(DEFAULT_THRESHOLDS)})")
    for key in ("standing_bl_s", "encounter_bl"):
        if key in th:
            kw[key] = _number(th[key], f"thresholds.{key}", positive=True)
    for key in ("speed_bl_s", "close_bl"):
        if key in th:
            kw[key] = _pair(th[key], f"thresholds.{key}")
    if th and bl is None:
        raise ValueError("thresholds are in body lengths: body_length is needed")

    default_r = DEFAULT_NEST_RADIUS_BL * bl if bl else None
    nest_raw: dict = {}
    nest_what = "places.nest"
    food = edges = None
    if data.get("places_file") is not None:
        side_path = rel(data["places_file"], "places_file")
        side = _read_json(side_path, "places file")
        if side.get("nest") is not None:
            if not isinstance(side["nest"], dict):
                raise ValueError(f"{side_path.name}: nest must be an object with x, y, got {side['nest']!r}")
            nest_raw, nest_what = dict(side["nest"]), f"{side_path.name} + places.nest"
        if side.get("food") is not None:
            food = _food(side["food"], f"{side_path.name}: food")
        if side.get("maze_edges") is not None:
            edges = _edges(side["maze_edges"], f"{side_path.name}: maze_edges")
    places = data.get("places", {})
    if not isinstance(places, dict):
        raise ValueError(f"places must be an object, got {places!r}")
    bad = sorted(set(places) - {"nest", "food"})
    if bad:
        raise ValueError(f"places: unknown key(s) {', '.join(bad)} (known: nest, food)")
    if places.get("nest") is not None:  # the config's keys (e.g. only r) win over the places file's
        if not isinstance(places["nest"], dict):
            raise ValueError(f"places.nest must be an object, got {places['nest']!r}")
        nest_raw.update(places["nest"])
    nest = _nest(nest_raw, nest_what, default_r) if nest_raw else None
    if places.get("food") is not None:
        food = _food(places["food"], "places.food")
    kw["nest"], kw["food"], kw["maze_edges"] = nest, food or (), edges or ()

    events = data.get("events", {})
    if not isinstance(events, dict):
        raise ValueError(f"events must be an object, got {events!r}")
    bad = sorted(set(events) - set(EVENTS))
    if bad:
        raise ValueError(f"events: unknown key(s) {', '.join(bad)} (known: {', '.join(EVENTS)})")
    for key, value in events.items():
        if key == "arrivals":
            if value not in ARRIVALS:
                raise ValueError(f"events.arrivals must be one of {', '.join(ARRIVALS)}, got {value!r}")
            if value == "nest" and nest is None:
                raise ValueError("events.arrivals is 'nest' but no nest is given")
        elif not isinstance(value, bool):
            raise ValueError(f"events.{key} must be true or false, got {value!r}")
    if events.get("food_discovery") and not any(f.t is not None for f in (food or ())):
        raise ValueError("events.food_discovery is on but no food item has a discovery time t")
    kw["events"] = dict(events)
    kw["image"] = rel(data.get("image"), "image")
    kw["video"] = rel(data.get("video"), "video")
    kw["video_dark"] = rel(data.get("video_dark"), "video_dark")
    return SceneConfig(**kw)
