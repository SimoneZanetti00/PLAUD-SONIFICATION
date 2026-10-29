"""Camera view for replay (display only): project world metres into an OpenTraj reference frame.

The frames and homographies are NOT part of this repository. They come from OpenTraj
(github.com/crowdbotp/OpenTraj, Amirian et al. 2020), downloaded by the user into a folder given
with --replay-camera-dir (or the PLAUD_OPENTRAJ_DIR environment variable), laid out as in OpenTraj:
  <dir>/zara01/reference.png, <dir>/zara01/H-cam.txt
  <dir>/seq_hotel/reference.png, <dir>/seq_hotel/H.txt

Convention, checked by overlaying the Trajectron++ / Social GAN tracks on each frame: H maps image
pixels to world metres, so world -> image uses H^-1; the projected pair is (row, col).
Pure Python, no pygame, no numpy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_DIR = "PLAUD_OPENTRAJ_DIR"
CREDIT = "OpenTraj frame (Amirian et al. 2020)"

# Trajectron++ file stem -> (OpenTraj folder, homography file). Only names, no data.
SCENE_CAMERAS = {
    "crowds_zara01": ("zara01", "H-cam.txt"),
    "biwi_hotel": ("seq_hotel", "H.txt"),
}
IMAGE_NAME = "reference.png"

Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def invert3(m: Matrix) -> Matrix:
    """Inverse of a 3x3 matrix (adjugate / determinant)."""
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-18:
        raise ValueError("homography is singular")
    return (
        ((e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det),
        ((f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det),
        ((d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det),
    )


def read_homography(path: str | Path) -> Matrix:
    values = [float(v) for v in Path(path).read_text(encoding="utf-8").split()]
    if len(values) != 9:
        raise ValueError(f"{Path(path).name}: expected 9 numbers, got {len(values)}")
    return (tuple(values[0:3]), tuple(values[3:6]), tuple(values[6:9]))  # type: ignore[return-value]


@dataclass(frozen=True)
class Camera:
    """World metres -> pixel (col, row) of one reference frame."""
    image_path: Path
    to_image: Matrix  # H^-1
    credit: str = CREDIT

    @classmethod
    def from_homography(cls, image_path: str | Path, h_image_to_world: Matrix) -> "Camera":
        return cls(Path(image_path), invert3(h_image_to_world))

    def project(self, x: float, y: float) -> tuple[float, float] | None:
        """(col, row) in image pixels, or None for a point on or behind the camera plane."""
        (a, b, c), (d, e, f), (g, h, i) = self.to_image
        w = g * x + h * y + i
        if w <= 1e-9:
            return None
        row = (a * x + b * y + c) / w
        col = (d * x + e * y + f) / w
        return (col, row)


def find_camera(replay_path: str | Path, directory: str | Path | None) -> tuple[Camera | None, str]:
    """The camera for a replay file, or (None, why not). `directory` None -> $PLAUD_OPENTRAJ_DIR."""
    stem = Path(replay_path).stem
    if stem not in SCENE_CAMERAS:
        known = ", ".join(sorted(SCENE_CAMERAS))
        return None, f"no reference frame known for {Path(replay_path).name} (camera view exists for: {known})"
    directory = directory or os.environ.get(ENV_DIR)
    if not directory:
        return None, f"no OpenTraj folder: pass --replay-camera-dir DIR or set {ENV_DIR}"
    folder, h_name = SCENE_CAMERAS[stem]
    image = Path(directory) / folder / IMAGE_NAME
    h_file = Path(directory) / folder / h_name
    for needed in (image, h_file):
        if not needed.is_file():
            return None, f"missing {needed}"
    try:
        return Camera.from_homography(image, read_homography(h_file)), ""
    except (OSError, ValueError) as exc:
        return None, f"{h_file.name}: {exc}"
