"""Camera view for replay (display only): project world metres into an OpenTraj reference frame.

The frames and homographies are NOT part of this repository. They come from OpenTraj
(github.com/crowdbotp/OpenTraj, Amirian et al. 2020), downloaded by the user into a folder given
with --replay-camera-dir (or the PLAUD_OPENTRAJ_DIR environment variable), laid out as in OpenTraj:
  <dir>/zara01/reference.png, <dir>/zara01/H-cam.txt
  <dir>/seq_hotel/reference.png, <dir>/seq_hotel/H.txt

Convention, checked by overlaying the Trajectron++ / Social GAN tracks on each frame: H maps image
pixels to world metres, so world -> image uses H^-1; the projected pair is (row, col).
zara01 also needs an image-space correction (mirror left-right, x1.2 about the frame centre, shift
(+10, +40) px): the Trajectron++ zara coordinates are not exactly in OpenTraj's H-cam world frame.
Found by syncing the data with OpenTraj's video.avi at the same frame number (people under the
dots: motion hit rate 0.29 -> 0.86 at offset 0, as good as hotel's 0.85 without correction); the
still reference frame alone could not tell left from right.
Pure Python, no pygame, no numpy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_DIR = "PLAUD_OPENTRAJ_DIR"
CREDIT = "OpenTraj frame (Amirian et al. 2020)"
VIDEO_CREDIT = "OpenTraj video (Amirian et al. 2020)"
VIDEO_NAME = "video.avi"

# Image-space correction applied after H^-1: (col, row) -> (col', row'), as a 3x3 affine matrix.
# zara01: col' = (719 - col - 360) x 1.2 + 360 + 10, row' = (row - 288) x 1.2 + 288 + 40.
ZARA01_FIX = ((-1.2, 0.0, 800.8), (0.0, 1.2, -17.6), (0.0, 0.0, 1.0))

# Trajectron++ file stem -> (OpenTraj folder, homography file, image correction or None). No data.
SCENE_CAMERAS = {
    "crowds_zara01": ("zara01", "H-cam.txt", ZARA01_FIX),
    "biwi_hotel": ("seq_hotel", "H.txt", None),
}
SWAP_ROW_COL = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
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


def matmul3(a: Matrix, b: Matrix) -> Matrix:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))  # type: ignore


def read_homography(path: str | Path) -> Matrix:
    values = [float(v) for v in Path(path).read_text(encoding="utf-8").split()]
    if len(values) != 9:
        raise ValueError(f"{Path(path).name}: expected 9 numbers, got {len(values)}")
    return (tuple(values[0:3]), tuple(values[3:6]), tuple(values[6:9]))  # type: ignore[return-value]


@dataclass(frozen=True)
class Camera:
    """World metres -> pixel (col, row) of one reference frame."""
    image_path: Path
    to_image: Matrix  # world (x, y, 1) -> image (col, row, 1), homogeneous
    credit: str = CREDIT

    @classmethod
    def from_homography(cls, image_path: str | Path, h_image_to_world: Matrix,
                        fix: Matrix | None = None) -> "Camera":
        """H^-1, then (row, col) -> (col, row), then the optional affine image correction."""
        m = matmul3(SWAP_ROW_COL, invert3(h_image_to_world))
        if fix is not None:
            m = matmul3(fix, m)
        return cls(Path(image_path), m)

    def project(self, x: float, y: float) -> tuple[float, float] | None:
        """(col, row) in image pixels, or None for a point on or behind the camera plane."""
        (a, b, c), (d, e, f), (g, h, i) = self.to_image
        w = g * x + h * y + i
        if w <= 1e-9:
            return None
        return ((a * x + b * y + c) / w, (d * x + e * y + f) / w)


def find_video(replay_path: str | Path, directory: str | Path | None) -> tuple[Path | None, str]:
    """<dir>/<scene folder>/video.avi for a replay file, or (None, why not)."""
    stem = Path(replay_path).stem
    directory = directory or os.environ.get(ENV_DIR)
    if stem not in SCENE_CAMERAS or not directory:
        return None, "no camera for this scene"
    video = Path(directory) / SCENE_CAMERAS[stem][0] / VIDEO_NAME
    return (video, "") if video.is_file() else (None, f"missing {video}")


def find_camera(replay_path: str | Path, directory: str | Path | None) -> tuple[Camera | None, str]:
    """The camera for a replay file, or (None, why not). `directory` None -> $PLAUD_OPENTRAJ_DIR."""
    stem = Path(replay_path).stem
    if stem not in SCENE_CAMERAS:
        known = ", ".join(sorted(SCENE_CAMERAS))
        return None, f"no reference frame known for {Path(replay_path).name} (camera view exists for: {known})"
    directory = directory or os.environ.get(ENV_DIR)
    if not directory:
        return None, f"no OpenTraj folder: pass --replay-camera-dir DIR or set {ENV_DIR}"
    folder, h_name, fix = SCENE_CAMERAS[stem]
    image = Path(directory) / folder / IMAGE_NAME
    h_file = Path(directory) / folder / h_name
    for needed in (image, h_file):
        if not needed.is_file():
            return None, f"missing {needed}"
    try:
        return Camera.from_homography(image, read_homography(h_file), fix), ""
    except (OSError, ValueError) as exc:
        return None, f"{h_file.name}: {exc}"
