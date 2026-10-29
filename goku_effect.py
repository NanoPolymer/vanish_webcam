"""Two fingers on the forehead start a teleport. Vertical slits open in the body."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


EFFECT_SECONDS = 0.65
HOLD_SECONDS = 1.0


def _finger(points: np.ndarray, base: int) -> tuple[float, float]:
    """Return the path length and the straightness of a finger."""
    finger = points[base:base + 4]
    length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
    if length < 1e-6:
        return 0.0, 0.0
    return length, float(np.linalg.norm(finger[3] - finger[0])) / length


def fingers_on_forehead(hands, face: tuple[float, float, float, float, float] | None,
                        width: int, height: int) -> bool:
    """Find a straight index and middle finger together, with the tips on the forehead.

    The face is the bounding box (left, top, width, height) and the eye height, in pixels.
    """
    if face is None:
        return False
    left, top, face_width, face_height, eye_y = face
    for hand in hands:
        if len(hand) != 21:
            continue
        points = np.array([(p.x * width, p.y * height) for p in hand])
        if not np.isfinite(points).all():
            continue
        index_length, index_straight = _finger(points, 5)
        middle_length, middle_straight = _finger(points, 9)
        if min(index_straight, middle_straight) < 0.8 or min(index_length, middle_length) < 8:
            continue
        # The tips touch each other. A V sign has a much larger gap.
        if np.linalg.norm(points[8] - points[12]) > 0.45 * min(index_length, middle_length):
            continue
        tip_x, tip_y = (points[8] + points[12]) / 2
        if (left - 0.15 * face_width <= tip_x <= left + 1.15 * face_width
                and top - 0.45 * face_height <= tip_y <= eye_y):
            return True
    return False


@dataclass
class HoldTimer:
    """Start the effect when the pose stays for the full hold time.

    After a trigger, the pose must stop before the timer can start again.
    """

    seconds: float = HOLD_SECONDS
    started: float | None = None
    armed: bool = True

    def update(self, pose: bool, now: float) -> bool:
        if not pose:
            self.started = None
            self.armed = True
            return False
        if not self.armed:
            return False
        if self.started is None:
            self.started = now
        if now - self.started >= self.seconds:
            self.started = None
            self.armed = False
            return True
        return False


def make_teleport_field(height: int, width: int, seed: int = 4721) -> np.ndarray:
    """Make the time in [0, 1) at which each pixel of the body becomes transparent.

    The columns make vertical stripes of random width. Each stripe opens at a
    random height and time. Then it extends up and down.
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(height, dtype=np.float32)
    field = np.empty((height, width), dtype=np.float32)
    x = 0
    while x < width:
        stripe = int(rng.integers(2, max(3, width // 60)))
        start = rng.uniform(0.0, 0.7)
        center = rng.uniform(0.0, height)
        growth = rng.uniform(0.4, 1.1)
        field[:, x:x + stripe] = (start + np.abs(rows - center) / height * growth)[:, None]
        x += stripe
    # The last parts of the body go at the end of the effect.
    return np.minimum(field, 0.98)


def render_teleport(frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
                    field: np.ndarray, progress: float) -> np.ndarray:
    """Show the background through vertical slits in the body.

    At progress 0 the body is complete. At progress 1 only the background shows.
    """
    if progress >= 1.0:
        return background.copy()
    points = cv2.findNonZero((mask > 0.01).astype(np.uint8))
    if points is None:
        return background.copy()
    x, y, width, height = cv2.boundingRect(points)
    area = np.s_[y:y + height, x:x + width]
    solid = (field[area] >= progress).astype(np.float32)
    alpha = mask[area] * solid
    output = (
        frame[area].astype(np.float32) * alpha[:, :, None]
        + background[area].astype(np.float32) * (1.0 - alpha[:, :, None])
    )
    if progress > 0.0:
        # Thin dark lines at the slit edges, like the ink lines in the anime frames.
        change = np.abs(np.diff(solid, axis=1))
        edge = np.zeros_like(solid)
        edge[:, 1:] = change
        edge[:, :-1] = np.maximum(edge[:, :-1], change)
        edge = (edge * alpha)[:, :, None]
        output *= 1.0 - 0.55 * edge
    canvas = background.copy()
    canvas[area] = output.astype(np.uint8)
    return canvas
