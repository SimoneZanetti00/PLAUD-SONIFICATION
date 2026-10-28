"""Load a trajectory file and fit it into the simulation world. Pure functions, no pygame.

Formats: CSV (below) or the ETH/UCY text files of Social GAN / Trajectron++ (`load_eth_ucy`).
CSV: a header row with at least `id, frame, x, y` (any order, any case, extra columns ignored).
An optional `t` column (seconds) gives the time of each row. Without it, time comes from the
frame number: `fps` is data rows per second, and the frame numbers may advance in steps
(e.g. 0, 10, 20 when every 10th video frame was annotated); the step is detected as the GCD
of the frame differences, so t = frame / frame_step / fps.
Ids are kept as strings. Units of x and y are whatever the dataset uses: `fit_to_world`
maps them uniformly (aspect kept) into the world rectangle.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ("id", "frame", "x", "y")


@dataclass(frozen=True)
class Sample:
    t: float  # seconds
    frame: int
    x: float
    y: float


@dataclass(frozen=True)
class Dataset:
    tracks: dict[str, tuple[Sample, ...]]  # id -> samples sorted by time; ids ordered by first time, then id
    fps: float  # data rows per second (used when there is no t column)
    frame_step: int = 1  # frame-number distance between consecutive rows of one id

    @property
    def duration(self) -> float:
        times = [s.t for samples in self.tracks.values() for s in samples]
        return max(times) - min(times) if times else 0.0

    @property
    def max_population(self) -> int:
        """Largest number of ids present in the same data frame."""
        per_frame: dict[int, int] = {}
        for samples in self.tracks.values():
            for s in samples:
                per_frame[s.frame] = per_frame.get(s.frame, 0) + 1
        return max(per_frame.values(), default=0)


@dataclass(frozen=True)
class Bounds:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class Transform:
    """world = (data - origin) * scale + offset, with y optionally flipped inside the bounds."""
    scale: float
    origin_x: float
    origin_y: float
    offset_x: float
    offset_y: float
    flip_y: bool
    span_y: float  # data height, used by the flip

    def apply(self, x: float, y: float) -> tuple[float, float]:
        dy = y - self.origin_y
        if self.flip_y:
            dy = self.span_y - dy
        return ((x - self.origin_x) * self.scale + self.offset_x, dy * self.scale + self.offset_y)


def id_sort_key(track_id: str) -> tuple[int, int | str]:
    """Numeric ids in numeric order (2 before 10), then the other ids as text."""
    try:
        return (0, int(track_id))
    except ValueError:
        return (1, track_id)


def detect_frame_step(frames_per_id: list[list[int]]) -> int:
    """GCD of the differences between consecutive (sorted) frame numbers of each id; 1 if none."""
    step = 0
    for frames in frames_per_id:
        for a, b in zip(frames, frames[1:]):
            step = math.gcd(step, b - a)
    return step or 1


def _finite(text: str | None, name: str) -> float:
    value = float(text)  # type: ignore[arg-type]  # None -> TypeError, caught by the caller
    if not math.isfinite(value):
        raise ValueError(f"{name} is {value}")
    return value


FORMATS = ("auto", "csv", "eth-ucy")

def load_trajectories(path: str | Path, fmt: str = "auto", fps: float = 2.5, frame_step: int | None = None) -> Dataset:
    """Load any supported file. fmt "auto": `.txt` = eth-ucy, anything else = csv."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r} (expected one of {', '.join(FORMATS)})")
    if fmt == "auto":
        fmt = "eth-ucy" if Path(path).suffix.lower() == ".txt" else "csv"
    return load_eth_ucy(path, fps, frame_step) if fmt == "eth-ucy" else load_csv(path, fps, frame_step)


def load_csv(path: str | Path, fps: float = 2.5, frame_step: int | None = None) -> Dataset:
    """Read a trajectory CSV. Raises ValueError (with the line number) on bad input.

    frame_step: None = detect it from the frame numbers; ignored for time when a t column exists.
    """
    _check_rate(fps, frame_step)
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            raise ValueError(f"{path.name}: missing column(s) {', '.join(missing)} (header: {header})")
        reader.fieldnames = header
        has_t = "t" in header

        def rows():
            for row in reader:
                if not any(isinstance(v, str) and v.strip() for v in row.values()):
                    continue  # blank line
                track_id = row["id"].strip() if isinstance(row["id"], str) else ""
                yield (reader.line_num, track_id, row["frame"], row["x"], row["y"], row["t"] if has_t else None)

        return _build(path, rows(), fps, frame_step)


def load_eth_ucy(path: str | Path, fps: float = 2.5, frame_step: int | None = None) -> Dataset:
    """Read an ETH/UCY file as distributed with Social GAN / Trajectron++ (experiments/pedestrians/raw).

    No header; one row per line, `frame id x y` separated by tabs or spaces; x, y in metres.
    Ids written as floats ("1.0") become "1".
    """
    _check_rate(fps, frame_step)
    path = Path(path)
    with path.open(encoding="utf-8-sig") as f:

        def rows():
            for line_no, text in enumerate(f, start=1):
                parts = text.split()
                if not parts:
                    continue
                if len(parts) != 4:
                    raise ValueError(f"{path.name}:{line_no}: expected 4 fields (frame id x y), got {len(parts)}")
                frame, track_id, x, y = parts
                try:
                    as_float = float(track_id)
                    if as_float.is_integer():
                        track_id = str(int(as_float))
                except ValueError:
                    pass
                yield (line_no, track_id, frame, x, y, None)

        return _build(path, rows(), fps, frame_step)


def _check_rate(fps: float, frame_step: int | None) -> None:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if frame_step is not None and frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")


def _build(path: Path, rows, fps: float, frame_step: int | None) -> Dataset:
    """Validate rows (line, id, frame, x, y, t-or-None) and turn them into time-sorted tracks."""
    raw: dict[str, dict[int, tuple[float | None, float, float, int]]] = {}
    for line, track_id, frame_text, x_text, y_text, t_text in rows:
        try:
            frame_f = _finite(frame_text, "frame")
            if not frame_f.is_integer():
                raise ValueError(f"frame {frame_f} is not an integer")
            frame = int(frame_f)
            x = _finite(x_text, "x")
            y = _finite(y_text, "y")
            t = _finite(t_text, "t") if t_text is not None else None
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"{path.name}:{line}: bad value ({exc})") from None
        if not track_id:
            raise ValueError(f"{path.name}:{line}: empty id")
        frames = raw.setdefault(track_id, {})
        if frame in frames:
            raise ValueError(f"{path.name}:{line}: id {track_id!r} appears twice in frame {frame}")
        frames[frame] = (t, x, y, line)
    if not raw:
        raise ValueError(f"{path.name}: no data rows")

    step = frame_step or detect_frame_step([sorted(frames) for frames in raw.values()])
    tracks: dict[str, tuple[Sample, ...]] = {}
    for tid, frames in raw.items():
        samples = []
        for frame, (t, x, y, _) in frames.items():
            samples.append(Sample(t if t is not None else frame / step / fps, frame, x, y))
        samples.sort(key=lambda s: (s.t, s.frame))
        for a, b in zip(samples, samples[1:]):
            if b.t <= a.t:
                line = frames[b.frame][3]
                raise ValueError(f"{path.name}:{line}: id {tid!r} has two rows at t={b.t}")
        tracks[tid] = tuple(samples)
    ordered = dict(sorted(tracks.items(), key=lambda kv: (kv[1][0].t, id_sort_key(kv[0]))))
    return Dataset(tracks=ordered, fps=fps, frame_step=step)


def data_bounds(dataset: Dataset) -> Bounds:
    xs = [s.x for samples in dataset.tracks.values() for s in samples]
    ys = [s.y for samples in dataset.tracks.values() for s in samples]
    if not xs:
        raise ValueError("empty dataset")
    return Bounds(min(xs), min(ys), max(xs), max(ys))


def fit_to_world(bounds: Bounds, world: tuple[float, float] = (1200.0, 750.0), margin: float = 60.0,
                 flip_y: bool = False) -> Transform:
    """Uniform scale that fits the bounds inside the world minus a margin, centred, aspect kept.

    A zero-size axis takes its scale from the other one; a single point maps to the centre.
    """
    width, height = world
    avail_w, avail_h = width - 2.0 * margin, height - 2.0 * margin
    if avail_w <= 0 or avail_h <= 0:
        raise ValueError(f"margin {margin} leaves no room in world {world}")
    span_x, span_y = bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin
    candidates = [avail_w / span_x if span_x > 0 else None, avail_h / span_y if span_y > 0 else None]
    scales = [c for c in candidates if c is not None]
    scale = min(scales) if scales else 1.0
    offset_x = (width - span_x * scale) / 2.0
    offset_y = (height - span_y * scale) / 2.0
    return Transform(scale, bounds.xmin, bounds.ymin, offset_x, offset_y, flip_y, span_y)


def to_world(dataset: Dataset, transform: Transform) -> Dataset:
    """The same dataset with x, y mapped into world coordinates."""
    tracks = {
        tid: tuple(Sample(s.t, s.frame, *transform.apply(s.x, s.y)) for s in samples)
        for tid, samples in dataset.tracks.items()
    }
    return Dataset(tracks=tracks, fps=dataset.fps, frame_step=dataset.frame_step)
