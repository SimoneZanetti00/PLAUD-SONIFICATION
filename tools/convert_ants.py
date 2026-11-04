"""Convert ant tracking datasets to the replay CSV (id,frame,t,x,y) plus a places file and a starter
scene config. The data stays outside the repository; write the output next to it.

ANTS (Wu et al. 2022, GigaScience; CC0): one sequence folder with seqinfo.ini, gt/gt.txt, img/
    python tools/convert_ants.py ants <SEQ_DIR> [<SEQ_DIR> ...] --out <OUT_DIR> [--video]
  gt.txt rows are frame,id,left,top,w,h,1 in pixels (top-left corner of a fixed-size box). The ant
  is the box centre (left + w/2, top + h/2), in image pixels, y down (no flip, no homography).
  frame = gt frame - 1, so frame 0 is img/000001.jpg and is video frame 0; t = frame / frameRate
  of the sequence's seqinfo.ini (25 indoor, 30 outdoor). --video builds <name>.mp4 from img/
  (needs OpenCV, requirements-video.txt), at the same frame rate.

Foraging Ants as Liquid Brains (Fernandez-Lopez et al. 2025, PNAS; data CC BY 4.0):
    python tools/convert_ants.py liquid <TRIAL.csv> --hex <hex.csv> --out <OUT_DIR>
                                 [--food <food_items.csv> --exp N]
  The trial CSVs are R write.csv output with row names: each row has one more field than the
  header, and the first one (the row name) is dropped. Columns kept: N_ind -> id, Frame -> frame,
  Time_sec -> t, Xmm, Ymm -> x, y (mm). The places file has the nest (maze node 634, as in the
  authors' figure code), the maze edges of the arena (nodes with y > 1000 mm closer than 51 mm,
  as their compute_edges) and, with --food, the food items of experiment N with their discovery
  time in seconds (frame / 2). food_items.csv (exp, patch, item, x, y, t in frames) is made from
  the authors' processed_data/det.RData with base R, for example:
    e <- new.env(); load("det.RData", envir = e)
    write.csv(do.call(rbind, lapply(seq_along(e$det), function(i) { f <- e$det[[i]]$food
      do.call(rbind, lapply(seq_along(f), function(k) cbind(exp = i, patch = k,
        item = seq_len(nrow(f[[k]])), as.data.frame(f[[k]])))) })), "food_items.csv", row.names = FALSE)
  det[[i]] is Food_Exp_<i> (same row count, checked for i = 1).

A starter <name>.scene.json is written only when none exists, so a hand-edited one is kept.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import sys
from pathlib import Path

LB_NEST_NODE = "634"
LB_ARENA_Y_MIN = 1000.0  # mm: the arena is the half of the maze above this (authors' code)
LB_EDGE_MM = 51.0  # mm: maze nodes closer than this are joined (authors' compute_edges)
LB_FPS = 2.0
LB_BODY_LENGTH_MM = 6.0  # Aphaenogaster senilis worker: an estimate, to be checked
ANTS_CITATION = "ANTS: Wu et al. 2022"
ANTS_REFERENCE = ("ANTS: Wu, Cao, Yang, Cao & Guo, A dataset of ant colonies' motion trajectories in indoor "
                  "and outdoor scenes to study clustering behavior, GigaScience 2022 (data CC0)")
LB_CITATION = "Liquid Brains: Fernandez-Lopez et al. 2025"
LB_REFERENCE = ("Fernandez-Lopez, Oro, Lloret-Cabot, Genovart, Garriga & Bartumeus, Foraging ants as liquid "
                "brains: movement heterogeneity shapes collective efficiency, PNAS 2025 (data CC BY 4.0)")
EVENTS_ANTS = {"arrivals": "anywhere", "together": False, "food_discovery": False, "recruitment": False,
               "encounters": True}


def read_seqinfo(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read_string(path.read_text(encoding="utf-8"))
    if "Sequence" not in parser:
        raise ValueError(f"{path}: no [Sequence] section")
    return dict(parser["Sequence"])


def ants_rows(lines, fps: float) -> list[tuple[str, int, float, float, float]]:
    """gt.txt lines -> (id, frame from 0, t, centre x, centre y), sorted by frame then id."""
    out = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(f"gt.txt:{n}: expected frame,id,left,top,w,h,..., got {line!r}")
        frame = int(float(parts[0])) - 1
        if frame < 0:
            raise ValueError(f"gt.txt:{n}: frames start at 1, got {parts[0]}")
        left, top, w, h = (float(v) for v in parts[2:6])
        out.append((parts[1].strip(), frame, frame / fps, left + w / 2.0, top + h / 2.0))
    out.sort(key=lambda r: (r[1], int(r[0]) if r[0].isdigit() else 0, r[0]))
    return out


def box_size(lines) -> float:
    """The (fixed) box width of an ANTS gt.txt, a stand-in for one body length in pixels."""
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 6:
            return float(parts[4])
    raise ValueError("gt.txt: no rows")


def liquid_rows(rows: list[list[str]]) -> list[tuple[str, int, float, float, float]]:
    """Liquid Brains CSV rows (header first) -> (id, frame, t, x mm, y mm)."""
    header = [h.strip() for h in rows[0]]
    need = ("N_ind", "Frame", "Xmm", "Ymm", "Time_sec")
    missing = [c for c in need if c not in header]
    if missing:
        raise ValueError(f"missing column(s) {', '.join(missing)} (header: {header})")
    col = {c: header.index(c) for c in need}
    out = []
    for n, row in enumerate(rows[1:], 2):
        if not row:
            continue
        if len(row) == len(header) + 1:  # R row names
            row = row[1:]
        elif len(row) != len(header):
            raise ValueError(f"line {n}: {len(row)} fields for a header of {len(header)}")
        out.append((row[col["N_ind"]].strip(), int(float(row[col["Frame"]])), float(row[col["Time_sec"]]),
                    float(row[col["Xmm"]]), float(row[col["Ymm"]])))
    out.sort(key=lambda r: (r[1], int(r[0]) if r[0].isdigit() else 0, r[0]))
    return out


def maze_nodes(rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    return {r["node"].strip(): (float(r["x"]), float(r["y"])) for r in rows}


def maze_edges(nodes: dict[str, tuple[float, float]], y_min: float = LB_ARENA_Y_MIN,
               max_len: float = LB_EDGE_MM) -> list[list[float]]:
    """Segments between arena nodes (y > y_min) closer than max_len, each pair once, sorted."""
    pts = sorted({xy for xy in nodes.values() if xy[1] > y_min})
    cell = max_len
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for p in pts:
        grid.setdefault((math.floor(p[0] / cell), math.floor(p[1] / cell)), []).append(p)
    edges = []
    for a in pts:
        cx, cy = math.floor(a[0] / cell), math.floor(a[1] / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for b in grid.get((cx + dx, cy + dy), []):
                    if a < b and math.dist(a, b) < max_len:
                        edges.append([round(a[0], 2), round(a[1], 2), round(b[0], 2), round(b[1], 2)])
    edges.sort()
    return edges


def food_items(rows: list[dict[str, str]], exp: int, fps: float = LB_FPS) -> list[dict]:
    """Food items of one experiment: x, y (mm), patch, t = discovery frame / fps (seconds)."""
    out = []
    for r in rows:
        if int(r["exp"]) != exp:
            continue
        t = r.get("t", "").strip()
        out.append({"patch": f"GP{r['patch']}", "item": int(r["item"]), "x": float(r["x"]), "y": float(r["y"]),
                    "t": None if t in ("", "NA") else round(float(t) / fps, 2)})
    if not out:
        raise ValueError(f"no food items for experiment {exp}")
    return sorted(out, key=lambda f: (f["patch"], f["item"]))


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "frame", "t", "x", "y"])
        for tid, frame, t, x, y in rows:
            w.writerow([tid, frame, repr(t), f"{x:.3f}", f"{y:.3f}"])  # t exact: even steps at 25 / 30 fps


def write_json_once(path: Path, data: dict) -> bool:
    if path.exists():
        return False
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def build_video(img_dir: Path, fps: float, out: Path) -> int:
    """img/*.jpg (name order) -> an mp4 at fps. Returns the number of frames written."""
    import cv2  # noqa: PLC0415  (optional, requirements-video.txt)
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise ValueError(f"no .jpg in {img_dir}")
    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {out}")
    for p in images:
        img = cv2.imread(str(p))
        if img is None or img.shape[:2] != (h, w):
            raise ValueError(f"{p}: unreadable or not {w}x{h}")
        writer.write(img)
    writer.release()
    return len(images)


def convert_ants(seq_dir: Path, out_dir: Path, video: bool) -> None:
    info = read_seqinfo(seq_dir / "seqinfo.ini")
    fps = float(info["framerate"])
    name = seq_dir.name  # the folder name (seqinfo's name is wrong for the outdoor sequences)
    lines = (seq_dir / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines()
    rows = ants_rows(lines, fps)
    write_csv(out_dir / f"{name}.csv", rows)
    box = box_size(lines)
    first = sorted((seq_dir / "img").glob("*.jpg")) if (seq_dir / "img").is_dir() else []
    config = {
        "name": name, "citation": ANTS_CITATION, "reference": ANTS_REFERENCE, "agents": "ants", "units": "px",
        "body_length": box, "body_length_is_estimate": True,
        "comment": "pixel coordinates of the image (y down); body length = the annotation box size, an estimate",
        "smooth_s": 0.2, "path_s": 2,
        "thresholds": {"standing_bl_s": 0.3, "speed_bl_s": [0.5, 5.0], "close_bl": [0.5, 3.0]},
        "events": EVENTS_ANTS,
        "image_credit": "ANTS frame (Wu et al. 2022)", "video_credit": "ANTS video (Wu et al. 2022)",
    }
    if first:
        config["image"] = str(first[0].resolve().relative_to(out_dir.resolve(), walk_up=True)).replace("\\", "/")
    print(f"{name}: {len(rows)} rows, {len({r[0] for r in rows})} ids, {fps:g} fps, box {box:g} px -> {name}.csv")
    if video:
        mp4 = out_dir / f"{name}.mp4"
        n = build_video(seq_dir / "img", fps, mp4) if not mp4.exists() else None
        print(f"  video {mp4.name}: " + ("kept (exists)" if n is None else f"{n} frames at {fps:g} fps"))
        config["video"] = mp4.name
    if write_json_once(out_dir / f"{name}.scene.json", config):
        print(f"  starter config {name}.scene.json")


def convert_liquid(trial: Path, hex_csv: Path, out_dir: Path, food_csv: Path | None, exp: int | None) -> None:
    with trial.open(newline="", encoding="utf-8") as f:
        rows = liquid_rows(list(csv.reader(f)))
    name = trial.stem.removesuffix("_data")
    write_csv(out_dir / f"{name}.csv", rows)
    with hex_csv.open(newline="", encoding="utf-8") as f:
        nodes = maze_nodes(list(csv.DictReader(f)))
    nx, ny = nodes[LB_NEST_NODE]
    places = {"units": "mm", "source": f"{trial.name}, {hex_csv.name}" + (f", {food_csv.name} exp {exp}" if food_csv else ""),
              "nest": {"x": nx, "y": ny, "node": int(LB_NEST_NODE)}, "food": [], "maze_edges": maze_edges(nodes)}
    if food_csv is not None:
        if exp is None:
            raise ValueError("--food needs --exp N (the Food experiment number)")
        with food_csv.open(newline="", encoding="utf-8") as f:
            places["food"] = food_items(list(csv.DictReader(f)), exp)
    (out_dir / f"{name}.places.json").write_text(json.dumps(places, indent=1) + "\n", encoding="utf-8")
    food = bool(places["food"])
    config = {
        "name": name, "citation": LB_CITATION, "reference": LB_REFERENCE, "agents": "ants", "units": "mm",
        "body_length": LB_BODY_LENGTH_MM, "body_length_is_estimate": True,
        "comment": "Aphaenogaster senilis in a Y-maze; body length 6 mm is an estimate; y flipped so the nest is at the bottom as in the paper",
        "flip_y": True, "grid": 50,
        "thresholds": {"standing_bl_s": 0.3, "speed_bl_s": [0.5, 5.0], "close_bl": [0.5, 3.0]},
        "places_file": f"{name}.places.json",
        "places": {"nest": {"r": 60}},
        "events": {"arrivals": "nest", "together": False, "food_discovery": food, "recruitment": True,
                   "encounters": True},
    }
    print(f"{name}: {len(rows)} rows, {len({r[0] for r in rows})} ids, frames {rows[0][1]}-{rows[-1][1]} -> {name}.csv; "
          f"places: nest node {LB_NEST_NODE}, {len(places['maze_edges'])} maze edges, {len(places['food'])} food items")
    if write_json_once(out_dir / f"{name}.scene.json", config):
        print(f"  starter config {name}.scene.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="dataset", required=True)
    a = sub.add_parser("ants", help="ANTS sequence folder(s)")
    a.add_argument("seq", nargs="+", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--video", action="store_true", help="also build <name>.mp4 from img/ (OpenCV)")
    lb = sub.add_parser("liquid", help="a Liquid Brains trial CSV")
    lb.add_argument("trial", type=Path)
    lb.add_argument("--hex", type=Path, required=True, help="processed_data/hex.csv (maze nodes)")
    lb.add_argument("--food", type=Path, help="food items CSV made from det.RData (see the docstring)")
    lb.add_argument("--exp", type=int, help="Food experiment number for --food")
    lb.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.dataset == "ants":
            for seq in args.seq:
                convert_ants(seq, args.out, args.video)
        else:
            convert_liquid(args.trial, args.hex, args.out, args.food, args.exp)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        sys.exit(f"convert_ants: {exc}")


if __name__ == "__main__":
    main()
