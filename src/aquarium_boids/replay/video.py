"""Video view for replay (display only): the scene's original video behind the projected people.

Optional dependency: OpenCV (requirements-video.txt). It is imported only here, only for
--replay-view video; without it the app explains how to install it and uses the camera view.

Frame = round(25 x data time): the replay's data time is the video's frame number / 25 for the
ETH/UCY files (frame step 10 at 2.5 rows/s), checked against OpenTraj's video.avi (offset 0 for
zara01 and seq_hotel). Pause, R, loop and --replay-speed follow from the data time.

Decoding runs in a background thread: the main thread asks for (frame, size) and blits the latest
finished frame (OpenCV releases the GIL while decoding and resizing). Forward steps up to
MAX_GRAB frames are decoded sequentially (cheap grabs); anything else (restart, loop wrap, big
jumps) is a seek, which can take ~0.1-0.3 s on zara01: meanwhile the previous frame stays up.

Looks: "normal" (dimmed colour) or "dark" (grey and darker, with what moves since the previous
decoded frame brought back in colour, so the walking people stand out from the street).

Optional, off by default (the ETH/UCY videos use neither):
- dark_path: a second video with the dark look already applied (tools/convert_ants.py writes it,
  with this module's own dark function on consecutive frames), read instead of computing the look
  live, which costs ~60-100 ms per 1280x720 frame;
- ahead > 0: read-ahead. Once the wanted frame is decoded, the thread keeps decoding the next
  `ahead` frames (same size and look; wrapping to frame 0 with loop=True) into a small cache, so a
  short stall of the decoder does not show and a loop wrap needs no seek.
"""
from __future__ import annotations

import threading
from pathlib import Path

MAX_GRAB = 150  # forward jumps up to this many frames are decoded, not seeked (6 s at 25 fps)
LOOKS = ("normal", "dark")
DARK_LEVEL = 0.32  # dark look: grey street at this brightness
MOTION_MAX_GAP = 4  # frames: the motion mask needs the previous decoded frame to be this close
MOTION_THRESHOLD = 7  # grey levels of change below this are noise (compression, light flicker)
MOTION_GAIN = 14.0  # how fast the mask reaches full colour above the threshold
MOTION_GROW = 11  # px: the mask is dilated by this square, so whole bodies (not only their edges) keep colour
INSTALL_HINT = "pip install -r requirements-video.txt (OpenCV)"
AHEAD_FRAMES = 8  # read-ahead used for the videos in image pixels (ants)


def import_cv2():
    """(cv2, "") or (None, why)."""
    try:
        import cv2  # noqa: PLC0415  (optional dependency, imported on demand)
    except ImportError as exc:
        return None, f"the video view needs OpenCV, not installed ({exc}): {INSTALL_HINT}"
    return cv2, ""


class VideoFrames:
    """Decodes frames of one video on request, scaled, dimmed and converted to RGB bytes."""

    def __init__(self, path: str | Path, dim: int = 255, threaded: bool = True,
                 dark_path: str | Path | None = None, ahead: int = 0, loop: bool = False):
        cv2, why = import_cv2()
        if cv2 is None:
            raise RuntimeError(why)
        self._cv2 = cv2
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open {self.path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.dim = dim
        self._pos = 0  # index of the next frame the capture will decode
        self._want: tuple[int, tuple[int, int], str] | None = None
        self._done: tuple[int, tuple[int, int], str] | None = None
        self._latest: tuple[int, tuple[int, int], str, bytes] | None = None
        self._raw: tuple[int, object] | None = None  # (index, last decoded frame): a look change reuses it
        self._gray_now: tuple[int, object] | None = None  # (index, grey) of the last frame seen by _dark
        self._gray_before: tuple[int, object] | None = None  # (index, grey) of the frame before it
        self.seeks = 0
        self._dark_cap = None  # dark_path: the dark look read from a second video
        if dark_path is not None:
            self._dark_cap = cv2.VideoCapture(str(dark_path))
            if not self._dark_cap.isOpened():
                raise RuntimeError(f"cannot open {dark_path}")
        self._dark_pos = 0
        self._dark_raw: tuple[int, object] | None = None
        self.ahead = max(0, int(ahead))
        self.loop = loop
        self._cache: dict[tuple[int, tuple[int, int], str], bytes] = {}  # read-ahead frames
        self.cache_hits = 0
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._closed = False
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="replay-video", daemon=True)
            self._thread.start()

    def frame_for(self, data_time: float) -> int:
        """round(fps x data time), clamped to the video."""
        return max(0, min(self.count - 1, round(self.fps * data_time)))

    def request(self, index: int, size: tuple[int, int], look: str = "normal") -> None:
        """Ask for frame `index` scaled to `size` in `look`; returns at once (threaded) or after decoding."""
        if look not in LOOKS:
            raise ValueError(f"unknown look {look!r}")
        want = (index, (int(size[0]), int(size[1])), look)
        if self._thread is None:
            if want != self._done:
                self._latest = (*want, self._decode(*want))
                self._done = want
            return
        with self._wake:
            if want != self._want:
                self._want = want
                cached = self._cache.get(want)
                if cached is not None:  # read ahead: ready at once
                    self._latest = (*want, cached)
                    self._done = want
                    self.cache_hits += 1
                self._wake.notify()

    def latest(self) -> tuple[int, tuple[int, int], str, bytes] | None:
        """(index, size, look, RGB bytes) of the newest finished frame, or None before the first one."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._wake.notify()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()
        if self._dark_cap is not None:
            self._dark_cap.release()

    def _run(self) -> None:
        while True:
            with self._wake:
                job = None
                while not self._closed and (self._want is None or self._want == self._done):
                    job = self._next_ahead()
                    if job is not None:
                        break
                    self._wake.wait()
                if self._closed:
                    return
                want = self._want
                if want != self._done:
                    job = want
            data = self._decode(*job)
            with self._lock:
                if job == want:
                    self._latest = (*want, data)
                    self._done = want
                if self.ahead:
                    self._keep(job, data)

    def _next_ahead(self) -> tuple[int, tuple[int, int], str] | None:
        """Read ahead (under the lock): the first of the next `ahead` frames not decoded yet, or None."""
        if not self.ahead or self._done is None:
            return None
        index, size, look = self._done
        for k in range(1, self.ahead + 1):
            j = index + k
            if j >= self.count:
                if not self.loop:
                    return None
                j -= self.count
            if (j, size, look) not in self._cache:
                return (j, size, look)
        return None

    def _keep(self, job: tuple[int, tuple[int, int], str], data: bytes) -> None:
        """Cache a decoded frame (under the lock); drop frames outside the window ahead of the wanted one."""
        self._cache[job] = data
        if self._want is None:
            return
        index, size, look = self._want
        window = {((index + k) % self.count if self.loop else index + k, size, look) for k in range(self.ahead + 1)}
        for key in [key for key in self._cache if key not in window]:
            del self._cache[key]

    def _decode(self, index: int, size: tuple[int, int], look: str = "normal") -> bytes:
        cv2 = self._cv2
        if look == "dark" and self._dark_cap is not None:
            img = self._read_dark(index)
        else:
            img = self._read(index)
            if look == "dark":
                img = self._dark(index, img)
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        if look == "normal" and self.dim < 255:
            img = cv2.convertScaleAbs(img, alpha=self.dim / 255.0)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()

    def _read(self, index: int):
        """The raw frame `index` (BGR, full size): reused, decoded forward, or seeked."""
        if self._raw is not None and self._raw[0] == index:
            return self._raw[1]
        cv2, cap = self._cv2, self._cap
        step = index - self._pos
        if 0 <= step <= MAX_GRAB:
            for _ in range(step):
                cap.grab()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.seeks += 1
        ok, img = cap.read()
        self._pos = index + 1
        if not ok:
            img = self._blank()
        self._raw = (index, img)
        return img

    def _read_dark(self, index: int):
        """Frame `index` of the dark video (dark_path), like _read."""
        if self._dark_raw is not None and self._dark_raw[0] == index:
            return self._dark_raw[1]
        cv2, cap = self._cv2, self._dark_cap
        step = index - self._dark_pos
        if 0 <= step <= MAX_GRAB:
            for _ in range(step):
                cap.grab()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.seeks += 1
        ok, img = cap.read()
        self._dark_pos = index + 1
        if not ok:
            img = self._blank()
        self._dark_raw = (index, img)
        return img

    def _dark(self, index: int, img):
        """Grey, darker street; pixels that changed since the previous decoded frame keep their colour."""
        import numpy as np  # noqa: PLC0415  (comes with OpenCV)
        cv2 = self._cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = cv2.cvtColor(cv2.convertScaleAbs(gray, alpha=DARK_LEVEL), cv2.COLOR_GRAY2BGR)
        if self._gray_now is None or self._gray_now[0] != index:  # a new frame: shift the pair
            self._gray_before, self._gray_now = self._gray_now, (index, gray)
        prev = self._gray_before
        if prev is None or not 0 < index - prev[0] <= MOTION_MAX_GAP:
            return base
        diff = cv2.GaussianBlur(cv2.absdiff(gray, prev[1]), (0, 0), 2.5)
        mask = np.clip((diff.astype(np.float32) - MOTION_THRESHOLD) * (MOTION_GAIN / 255.0), 0.0, 1.0)
        mask = cv2.dilate(mask, np.ones((MOTION_GROW, MOTION_GROW), np.uint8))[..., None]
        out = base.astype(np.float32) * (1.0 - mask) + img.astype(np.float32) * mask
        return out.astype(np.uint8)

    def _blank(self):
        import numpy as np  # noqa: PLC0415  (comes with OpenCV)
        return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
