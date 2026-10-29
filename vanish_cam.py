"""Local webcam: a V sign with the index and middle finger starts the dissolve."""

from __future__ import annotations

import argparse
import os
import platform
import struct
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from .vanish_sound import SOUND_PATH, VanishSound
    from .meeting_audio import MIC_SOURCE, MeetingAudio
    from .goku_effect import (
        EFFECT_SECONDS as GOKU_SECONDS, HoldTimer, fingers_on_forehead,
        make_teleport_field, render_teleport,
    )
except ImportError:
    from naruto_effect import (
        EFFECT_SECONDS, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from vanish_sound import SOUND_PATH, VanishSound
    from meeting_audio import MIC_SOURCE, MeetingAudio
    from goku_effect import (
        EFFECT_SECONDS as GOKU_SECONDS, HoldTimer, fingers_on_forehead,
        make_teleport_field, render_teleport,
    )


APP_DIR = Path(__file__).resolve().parent

MODEL_DIR = APP_DIR / "models"

# Text in the window. The terminal output is always in English.
# Use only ASCII characters, so that all fonts can show the text.
LANGUAGES = ("en", "it")
LANGUAGE_NAMES = ("English", "Italiano")
MODES = ("normal", "naruto", "goku", "peace")
MODE_NAMES = {
    "normal": "Normal Vanish", "naruto": "Naruto Vanish",
    "goku": "Goku Vanish", "peace": "Peace Out",
}
TEXT = {
    "en": {
        "webcam": "Webcam",
        "video_mode": "Video mode",
        "effect": "Effect",
        "language": "Language",
        "normal_detail": "V sign  /  dissolve",
        "naruto_detail": "Joined hands  /  log and smoke",
        "goku_detail": "Two fingers on the forehead for 1 s  /  teleport",
        "peace_detail": "V sign  /  freeze frame and fade",
        "capture": "Capture background",
        "capturing": "Leave the view: {0}",
        "recapture": "New background",
        "vanish": "Vanish  (V)",
        "reappear": "Reappear  (R)",
        "fullscreen": "Full screen  (F)",
        "keys": "Space: background   Esc: leave full screen   Q: quit",
        "status_no_background": "Capture the background first.",
        "status_calibrating": "Move out of the camera view.",
        "status_ready": "Ready. Do the gesture.",
        "status_effect": "Vanishing...",
        "status_vanished": "Vanished. Press R to reappear.",
        "status_loading": "Loading the effect...",
        "status_camera_error": "Cannot open this webcam.",
        "status_mode_error": "Cannot load this effect.",
        "meeting": "Meeting microphone (VanishMic)",
        "status_meeting_error": "Cannot make VanishMic. Read the terminal.",
    },
    "it": {
        "webcam": "Webcam",
        "video_mode": "Modalita video",
        "effect": "Effetto",
        "language": "Lingua",
        "normal_detail": "Segno a V  /  dissolvenza",
        "naruto_detail": "Mani unite  /  tronco e fumo",
        "goku_detail": "Due dita sulla fronte per 1 s  /  teletrasporto",
        "peace_detail": "Segno a V  /  fermo immagine e dissolvenza",
        "capture": "Cattura sfondo",
        "capturing": "Esci dall'inquadratura: {0}",
        "recapture": "Nuovo sfondo",
        "vanish": "Sparisci  (V)",
        "reappear": "Riappari  (R)",
        "fullscreen": "Schermo intero  (F)",
        "keys": "Spazio: sfondo   Esc: esci da schermo intero   Q: esci",
        "status_no_background": "Prima cattura lo sfondo.",
        "status_calibrating": "Esci dall'inquadratura.",
        "status_ready": "Pronto. Fai il gesto.",
        "status_effect": "Sto sparendo...",
        "status_vanished": "Sparito. Premi R per riapparire.",
        "status_loading": "Carico l'effetto...",
        "status_camera_error": "Impossibile aprire questa webcam.",
        "status_mode_error": "Impossibile caricare questo effetto.",
        "meeting": "Microfono per riunioni (VanishMic)",
        "status_meeting_error": "Impossibile creare VanishMic. Leggi il terminale.",
    },
}
SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = APP_DIR / "hand_landmarker.task"
FACE_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
# The repository does not contain this sound. Without it, the Goku mode has no audio.
TELEPORT_SOUND = APP_DIR / "assets" / "teleport.MP3"
# The repository does not contain this sound. Without it, the Peace mode has no audio.
PEACE_SOUND = APP_DIR / "assets" / "peace.mp3"
MODE_SOUNDS = {"naruto": SOUND_PATH, "goku": TELEPORT_SOUND, "peace": PEACE_SOUND}

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
    FACE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    ),
}
# The models that each mode needs.
MODE_MODELS = {
    "normal": (SEGMENTER_MODEL, GESTURE_MODEL),
    "naruto": (SEGMENTER_MODEL,),
    "goku": (SEGMENTER_MODEL, FACE_MODEL),
    "peace": (SEGMENTER_MODEL, GESTURE_MODEL),
}

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
VANISH_SECONDS = 1.8
PEACE_SECONDS = 1.4
GESTURE_INTERVAL_SECONDS = 0.14


@dataclass
class AppState:
    phase: str = "calibrating"
    calibration_started: float = 0.0
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    fullscreen: bool = False


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("The FPS value must be between 1 and 120.")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a vanish effect on the webcam: Normal Vanish, Naruto Vanish, Goku Vanish or Peace Out."
    )
    parser.add_argument("--width", type=int, default=960, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=540, help="Video height in pixels")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Start in full screen"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Do not mirror the webcam image"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="Maximum FPS of the virtual output (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require the virtual webcam. By default, the app uses it when it is available. --no-virtual-camera disables it.",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Driver of the virtual webcam",
    )
    parser.add_argument("--virtual-device", help="Name of the virtual device")
    parser.add_argument(
        "--mode", choices=MODES,
        help="Effect at start (default: normal). The window can change it.",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Show the gesture recognition status in the terminal. The video shows no text.",
    )
    parser.add_argument(
        "--meeting-audio", action="store_true",
        help="Start with the virtual microphone VanishMic on (Linux). It mixes your microphone and the effect sound. The window can change it.",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def list_cameras(limit: int = 10) -> list[tuple[int, str]]:
    """Return the index and the name of each webcam that can supply video."""
    sysfs = Path("/sys/class/video4linux")
    if sysfs.is_dir():
        cameras = []
        for node in sorted(sysfs.glob("video*"), key=lambda path: int(path.name[5:])):
            try:
                # Index 0 is the video node of a device. Other indexes contain metadata.
                # v4l2loopback devices have no "device" link and are outputs, not webcams.
                if (node / "index").read_text().strip() != "0" or not (node / "device").exists():
                    continue
                cameras.append((int(node.name[5:]), (node / "name").read_text().strip()))
            except (OSError, ValueError):
                continue
        return cameras
    # ponytail: other systems give no names to OpenCV. Try each index until one does not open.
    cameras = []
    for index in range(limit):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        capture = cv2.VideoCapture(index, backend)
        opened = capture.isOpened()
        capture.release()
        if not opened:
            break
        cameras.append((index, f"Webcam {index}"))
    return cameras


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "The virtual webcam is not available. Install OBS Studio (with the "
            "OBS Virtual Camera driver) or Unity Capture. If you use OBS Virtual Camera, "
            "stop 'Start Virtual Camera' in OBS. "
            "Start the app with AVVIA_LOCALE.bat to install pyvirtualcam. "
            f"Details: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Webcam] {message}\n[Webcam] The app continues with the local preview only.")
        return None
    print(f"[Webcam] The virtual output is active: {camera.device} ({width}x{height}, maximum {args.fps:g} FPS)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination in MODE_MODELS[mode]:
        url = MODEL_URLS[destination]
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[Vanish Lab] The app downloads {destination.name}.")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Vanish-Lab/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Cannot download the MediaPipe models. "
                "Examine the internet connection. Then try again."
            ) from error


# The V4L2 ioctl requests (_IOWR('V', number, size)) that list the formats of a webcam.
def _v4l2_request(number: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("V") << 8) | number


VIDIOC_ENUM_FMT = _v4l2_request(2, 64)
VIDIOC_ENUM_FRAMESIZES = _v4l2_request(74, 44)
VIDIOC_ENUM_FRAMEINTERVALS = _v4l2_request(75, 52)
V4L2_DISCRETE = 1
VIDEO_FORMATS = ("MJPG", "YUYV")
# ponytail: other systems give no format list to OpenCV. These usual modes cover most webcams.
FALLBACK_VIDEO_MODES = [
    ("MJPG", 1920, 1080, 30.0), ("MJPG", 1280, 720, 30.0),
    ("MJPG", 960, 540, 30.0), ("MJPG", 640, 480, 30.0),
]


def _v4l2_list(ioctl, device: int, request: int, size: int, layout: str, *values: int):
    """Give the result buffers for the index 0, 1, 2 ... until the driver refuses the index."""
    index = 0
    while True:
        buffer = bytearray(size)
        struct.pack_into(layout, buffer, 0, index, *values)
        try:
            ioctl(device, request, buffer)
        except OSError:
            return
        yield buffer
        index += 1


def list_video_modes(index: int) -> list[tuple[str, int, int, float]]:
    """Return the video modes of a webcam as (FOURCC, width, height, fps), the best mode first."""
    path = f"/dev/video{index}"
    if not sys.platform.startswith("linux") or not os.path.exists(path):
        return list(FALLBACK_VIDEO_MODES)
    import fcntl
    try:
        device = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return list(FALLBACK_VIDEO_MODES)
    try:
        return read_video_modes(fcntl.ioctl, device) or list(FALLBACK_VIDEO_MODES)
    finally:
        os.close(device)


def read_video_modes(ioctl, device: int) -> list[tuple[str, int, int, float]]:
    """Ask the driver for the modes. Use MJPG and YUYV only, with a height of 360 or more.

    If both formats have the same size and fps, keep MJPG.
    """
    modes: dict[tuple[int, int, float], str] = {}
    for description in _v4l2_list(ioctl, device, VIDIOC_ENUM_FMT, 64, "<II", 1):
        pixel_format = struct.unpack_from("<I", description, 44)[0]
        fourcc = pixel_format.to_bytes(4, "little").decode("ascii", "replace")
        if fourcc not in VIDEO_FORMATS:
            continue
        for frame_size in _v4l2_list(ioctl, device, VIDIOC_ENUM_FRAMESIZES, 44, "<II", pixel_format):
            kind, width, height = struct.unpack_from("<III", frame_size, 8)
            if kind != V4L2_DISCRETE:
                continue
            for interval in _v4l2_list(ioctl, device, VIDIOC_ENUM_FRAMEINTERVALS, 52, "<IIII",
                                       pixel_format, width, height):
                kind, numerator, denominator = struct.unpack_from("<III", interval, 16)
                if kind != V4L2_DISCRETE or not numerator or height < 360:
                    continue
                key = (width, height, round(denominator / numerator, 2))
                if modes.get(key) != "MJPG":
                    modes[key] = fourcc
    # A webcam can give each size at many frame rates. Keep 30 and 60 fps, or the highest rate.
    kept = []
    for width, height in {(width, height) for width, height, _ in modes}:
        rates = sorted(fps for size_w, size_h, fps in modes if (size_w, size_h) == (width, height))
        usual = [fps for fps in rates if fps in (30.0, 60.0)] or rates[-1:]
        kept += [(modes[(width, height, fps)], width, height, fps) for fps in usual]
    return sorted(kept, key=lambda mode: (-mode[1] * mode[2], -mode[3]))


def default_video_mode(modes: list[tuple[str, int, int, float]], width: int, height: int,
                       fps: float) -> tuple[str, int, int, float]:
    """Select the highest fps that is not above fps. Then select the size nearest to the requested size."""
    fitting = [mode for mode in modes if mode[3] <= fps + 0.01] or modes
    return min(fitting, key=lambda mode: (-mode[3], abs(mode[1] * mode[2] - width * height), mode[0] != "MJPG"))


def video_mode_name(mode: tuple[str, int, int, float]) -> str:
    fourcc, width, height, fps = mode
    return f"{width}x{height}  {fps:g} fps  {fourcc}"


def open_camera(index: int, width: int, height: int, fourcc: str = "MJPG",
                fps: float | None = None) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} is not available. Close the other apps that use it, or select a different webcam in the window."
        )

    # Many webcams supply large YUYV frames only at a low frame rate. MJPG gives the full rate.
    # Set the format before the size, because the driver selects the size for the format.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def create_tasks(mode: str = "normal") -> tuple[
    vision.ImageSegmenter, MultiViewGestureRecognizer | MultiViewHandLandmarker | GokuGestureDetector
]:
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    if mode in ("naruto", "goku"):
        if not HAND_MODEL.is_file():
            raise RuntimeError(f"Cannot find the hand model: {HAND_MODEL}")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        task_class = vision.HandLandmarker
        wrapper_class = MultiViewHandLandmarker
    else:
        classifier_options = mp.tasks.components.processors.ClassifierOptions(
            score_threshold=0.55,
            category_allowlist=["Victory"],
        )
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=classifier_options,
        )
        task_class = vision.GestureRecognizer
        wrapper_class = MultiViewGestureRecognizer

    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        if mode == "goku":
            hands = vision.HandLandmarker.create_from_options(options)
            try:
                faces = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                    base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
                    running_mode=vision.RunningMode.VIDEO,
                ))
            except Exception:
                hands.close()
                raise
            return segmenter, GokuGestureDetector(hands, faces)
        primary = task_class.create_from_options(options)
        try:
            alternate = task_class.create_from_options(
                replace(options, running_mode=vision.RunningMode.IMAGE)
            )
        except Exception:
            primary.close()
            raise
    except Exception:
        segmenter.close()
        raise
    return segmenter, wrapper_class(primary, alternate)


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3 = alpha[:, :, None]
    return (
        person.astype(np.float32) * alpha_3
        + background.astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)


def apply_dissolve(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    progress: float,
    field: np.ndarray,
    sparkle_noise: np.ndarray,
) -> np.ndarray:
    if progress <= 0.002:
        return alpha_composite(frame, background, mask)
    if progress >= 0.999:
        return background.copy()

    points = cv2.findNonZero((mask > 0.01).astype(np.uint8))
    if points is None:
        return background.copy()
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x - 18), max(0, y - 18)
    right = min(mask.shape[1], x + width + 18)
    bottom = min(mask.shape[0], y + height + 18)
    area = np.s_[top:bottom, left:right]
    mask = mask[area]
    field = field[area]
    sparkle_noise = sparkle_noise[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    alpha = mask * survival
    output = alpha_composite(frame[area], background[area], alpha)
    edge = (
        (np.abs(field - eased) < 0.018)
        & (mask > 0.48)
        & (sparkle_noise > 0.54)
    ).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = np.clip(glow * 5.0 + edge, 0.0, 1.0)[:, :, None]
    cyan = np.full_like(output, (255, 228, 92), dtype=np.uint8)
    result = np.clip(
        output.astype(np.float32) * (1.0 - glow * 0.8)
        + cyan.astype(np.float32) * glow * 0.8,
        0,
        255,
    ).astype(np.uint8)
    canvas = background.copy()
    canvas[area] = result
    return canvas


def victory_landmark_score(points: np.ndarray) -> float:
    """Find an extended index finger and middle finger. Use 3D distances, not the orientation."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP and tip: index, middle, ring and little finger.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # From the side, folded fingers can look straight but very short.
    # Thus, do not require clearly visible joints for the ring and little finger.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # The two fingertips must be apart. The thumb can stay hidden.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Return the shape quality of the extended fingers, not the probability of the classifier.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # The result contains landmarks also when the classifier does not find
    # "Victory", for example when the back of the hand points to the webcam.
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # Always examine the visible shape also. The estimated depth can be
    # incorrect when the palm and the folded fingers are hidden.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


class MultiViewGestureRecognizer:
    """Try the detection again on rotated frames. Do not change the video tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # The IMAGE mode always does a new detection. The VIDEO model stays on the
        # original frame. This prevents incorrect timestamps and tracking between rotations.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # In the side pose, the palm is frequently too small in the full frame.
        # The two lower crops overlap. Together they contain both hands, also in a mirror image.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Divide the slow IMAGE attempts across a sequence of video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(rotated)
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class MultiViewHandLandmarker:
    """Track two hands in the video. Try difficult views again with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(views[crop], rotation)),
            )
            result = self.alternate.detect(rotated)
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        return False

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class GokuGestureDetector:
    """Find two fingers on the forehead with the hand model and the face model."""

    def __init__(self, hands, faces):
        self.hands = hands
        self.faces = faces
        self.face_found = False

    def detect(self, image: mp.Image, timestamp_ms: int) -> bool:
        hands = self.hands.detect_for_video(image, timestamp_ms).hand_landmarks
        detections = self.faces.detect_for_video(image, timestamp_ms).detections
        self.face_found = bool(detections)
        if not detections:
            return False
        box = detections[0].bounding_box
        # Keypoints 0 and 1 are the eyes. Their positions are normalized.
        eyes = detections[0].keypoints[:2]
        eye_y = sum(point.y for point in eyes) / len(eyes) * image.height if eyes else box.origin_y + box.height * 0.4
        face = (box.origin_x, box.origin_y, box.width, box.height, eye_y)
        return fingers_on_forehead(hands, face, image.width, image.height)

    def close(self) -> None:
        try:
            self.hands.close()
        finally:
            self.faces.close()


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


def start_vanish_on_gesture(state: AppState, name: str, score: float, now: float) -> bool:
    """Start the dissolve when the app first finds a reliable V sign."""
    if state.phase == "ready" and name == "Victory" and score >= 0.62:
        state.phase = "vanishing"
        state.vanish_started = now
        return True
    return False


def start_naruto_on_join(state: AppState, joined: bool, now: float) -> bool:
    if state.phase == "ready" and joined:
        state.phase = "naruto_vanishing"
        state.vanish_started = now
        return True
    return False


def self_test() -> int:
    """Make sure that the bundled resources and the models are available. Do not open a webcam."""
    required = (SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL, SOUND_PATH)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"These resources are missing: {', '.join(missing)}")
    load_log()
    import pygame.mixer  # The distributed build must contain its MP3 decoder.
    from PIL import ImageTk  # The window needs Pillow to show the video.
    for mode in MODE_MODELS:
        segmenter, recognizer = create_tasks(mode)
        try:
            recognizer.close()
        finally:
            segmenter.close()
    print("[Webcam] The self-test is complete: the models, assets and audio are available.")
    return 0


class VanishEngine:
    """Keep the background, the models and the effect state. Change a camera frame into the output frame."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.mode: str | None = None
        self.segmenter = None
        self.recognizer = None
        self.sound: VanishSound | None = None
        self.log_sprite: np.ndarray | None = None
        self.state = AppState(phase="no_background")
        self.background: np.ndarray | None = None
        self.background_samples: list[np.ndarray] = []
        self.calibration_remaining: int | None = None
        self.last_gesture_report = 0.0
        self.started = time.perf_counter()
        self.dissolve_field: np.ndarray | None = None
        self.sparkle_noise: np.ndarray | None = None
        self.teleport_field: np.ndarray | None = None
        self.log_cache: dict[int, np.ndarray] = {}
        self.reset_effect()

    def reset_effect(self) -> None:
        self.state.vanish_started = None
        self.last_mask: np.ndarray | None = None
        self.last_mask_at = float("-inf")
        self.goku_hold = HoldTimer()
        self.log_anchor: tuple[int, int, int] | None = None
        self.vanished_frame: np.ndarray | None = None
        # The Peace mode keeps the first frame of each vanish, with its start time.
        self.frozen: tuple[float, np.ndarray] | None = None

    def set_mode(self, mode: str) -> None:
        """Load the models and the sound of the mode. Keep the background."""
        if mode == self.mode:
            return
        download_models(mode)
        segmenter, recognizer = create_tasks(mode)
        self.close()
        self.segmenter, self.recognizer = segmenter, recognizer
        self.log_sprite = load_log() if mode == "naruto" else None
        self.log_cache = {}
        self.mode = mode
        self.reload_sound()
        self.reset_effect()
        if self.background is not None and self.state.phase != "calibrating":
            self.state.phase = "ready"

    def close(self) -> None:
        tasks = (self.recognizer, self.segmenter)
        self.segmenter = self.recognizer = None
        for task in tasks:
            if task is not None:
                task.close()
        if self.sound is not None:
            self.sound.close()
            self.sound = None

    def capture_background(self, now: float) -> None:
        self.state.phase = "calibrating"
        self.state.calibration_started = now
        self.background_samples.clear()
        self.calibration_remaining = None
        self.reset_effect()
        print("[Webcam] Move out of the camera view for 4 seconds. The app records the background.")

    def reload_sound(self) -> None:
        """Open the sound of the mode again. The audio output reads PULSE_SINK only when it opens."""
        if self.sound is not None:
            self.sound.close()
            self.sound = None
        if self.mode in MODE_SOUNDS:
            # The effect works without audio, so a missing audio device must not stop the app.
            try:
                self.sound = VanishSound(MODE_SOUNDS[self.mode])
            except RuntimeError as error:
                print(f"[Webcam] The audio is off: {error}")

    def forget_background(self) -> None:
        """Remove the background, for example after a change of the webcam."""
        self.background = None
        self.background_samples.clear()
        self.state.phase = "no_background"
        self.reset_effect()

    def seconds_left(self, now: float) -> int:
        """Return the remaining seconds of the background capture."""
        return max(0, int(np.ceil(CALIBRATION_SECONDS - (now - self.state.calibration_started))))

    def play_sound(self) -> None:
        if self.sound is not None:
            self.sound.play()

    def vanish(self, now: float) -> None:
        if self.background is None or self.state.phase == "calibrating":
            return
        self.reset_effect()
        self.state.phase = {"naruto": "naruto_vanishing", "goku": "goku_vanishing"}.get(self.mode, "vanishing")
        self.state.vanish_started = now
        self.play_sound()

    def reappear(self, now: float) -> None:
        if self.mode == "goku" and self.state.phase == "vanished":
            # In the Goku mode, R plays the teleport in reverse.
            self.reset_effect()
            self.state.phase = "goku_returning"
            self.state.vanish_started = now
            self.play_sound()
        elif self.background is not None and self.state.phase != "calibrating":
            self.state.phase = "ready"
            self.reset_effect()

    def process(self, camera_frame: np.ndarray, now: float) -> np.ndarray:
        """Return the output frame for one camera frame. The frame is already a mirror image if necessary."""
        state, mode, args = self.state, self.mode, self.args
        height, width = camera_frame.shape[:2]
        if mode == "normal" and (self.dissolve_field is None or self.dissolve_field.shape != (height, width)):
            self.dissolve_field, self.sparkle_noise = make_dissolve_fields(height, width)
        if mode == "goku" and (self.teleport_field is None or self.teleport_field.shape != (height, width)):
            self.teleport_field = make_teleport_field(height, width)

        elapsed_ms = int((now - self.started) * 1000)
        mp_image = None

        if (
            mode == "goku" and self.background is not None
            and state.phase in ("ready", "vanished")
            and now - state.last_gesture_check >= 0.08
        ):
            state.last_gesture_check = now
            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose = self.recognizer.detect(mp_image, elapsed_ms)
            if args.gesture_debug and now - self.last_gesture_report >= 1.0:
                print(f"[Goku] face={self.recognizer.face_found}, fingers on forehead={pose}")
                self.last_gesture_report = now
            if self.goku_hold.update(pose, now):
                state.phase = "goku_vanishing" if state.phase == "ready" else "goku_returning"
                state.vanish_started = now
                self.vanished_frame = None
                self.play_sound()

        if state.phase == "no_background":
            return camera_frame

        if state.phase == "calibrating":
            remaining = self.seconds_left(now)
            if remaining != self.calibration_remaining:
                print(f"[Webcam] Background capture: {remaining} seconds.")
                self.calibration_remaining = remaining
            elapsed = now - state.calibration_started
            if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                self.background_samples.append(camera_frame.copy())
            if elapsed >= CALIBRATION_SECONDS:
                if self.background_samples:
                    stack = np.stack(self.background_samples[-18:], axis=0)
                    self.background = np.median(stack, axis=0).astype(np.uint8)
                else:
                    self.background = camera_frame.copy()
                self.background_samples.clear()
                state.phase = "ready"
                print("[Webcam] The background is ready. You can move back into the camera view.")
            return camera_frame

        background = self.background
        if state.phase == "vanished":
            if self.vanished_frame is None:
                self.vanished_frame = background.copy()
                if mode == "naruto" and self.log_anchor is not None:
                    overlay_log(self.vanished_frame, self.log_sprite,
                                landed_log_anchor(self.log_anchor, height), cache=self.log_cache)
            return self.vanished_frame

        mask = self.last_mask
        if (
            state.phase == "ready" and mode != "goku"
            and now - state.last_gesture_check >=
            (0.08 if mode == "naruto" else GESTURE_INTERVAL_SECONDS)
        ):
            state.last_gesture_check = now
            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if mode in ("normal", "peace"):
                name, score = recognize_gesture(self.recognizer, mp_image, elapsed_ms)
                if args.gesture_debug and now - self.last_gesture_report >= 1.0:
                    print(
                        f"[Gesture] {name}, score={score:.2f}, "
                        f"crop={self.recognizer.preferred_crop or 'full'}, "
                        f"rotation={self.recognizer.preferred_rotation or 0}"
                    )
                    self.last_gesture_report = now
                if start_vanish_on_gesture(state, name, score, now):
                    self.play_sound()
            else:
                joined = self.recognizer.recognize_joined_hands(mp_image, elapsed_ms)
                if args.gesture_debug and now - self.last_gesture_report >= 1.0:
                    print(f"[Hands] found={self.recognizer.last_hand_count}, "
                          f"joined={joined}, view={self.recognizer.preferred_view or 'full'}")
                    self.last_gesture_report = now
                if start_naruto_on_join(state, joined, now):
                    self.play_sound()

        if state.phase in ("vanishing", "naruto_vanishing", "goku_vanishing", "goku_returning"):
            # The normal dissolve updates the silhouette at 20 Hz. The Naruto
            # effect needs it only when the person is still visible.
            needs_mask = (
                (mode in ("normal", "goku") and (mask is None or now - self.last_mask_at >= 0.05))
                or (mode == "naruto" and (self.log_anchor is None or now - state.vanish_started < 0.13))
            )
            if needs_mask:
                if mp_image is None:
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mask = person_mask(self.segmenter, mp_image, elapsed_ms, width, height)
                self.last_mask = mask
                self.last_mask_at = now
            if mode == "naruto" and self.log_anchor is None:
                self.log_anchor = substitution_anchor(mask)

        if mode == "naruto" and state.phase == "naruto_vanishing":
            age = max(0.04, now - state.vanish_started)
            display = render_substitution(
                camera_frame, background, mask, self.log_sprite, self.log_anchor, age, self.log_cache
            )
            if age >= EFFECT_SECONDS:
                state.phase = "vanished"
                self.vanished_frame = display
        elif mode == "goku" and state.phase in ("goku_vanishing", "goku_returning"):
            age = now - state.vanish_started
            progress = min(1.0, age / GOKU_SECONDS)
            if state.phase == "goku_returning":
                progress = 1.0 - progress
            display = render_teleport(camera_frame, background, mask, self.teleport_field, progress)
            if age >= GOKU_SECONDS:
                state.phase = "vanished" if state.phase == "goku_vanishing" else "ready"
                self.vanished_frame = None
        elif mode == "peace" and state.phase == "vanishing":
            if self.frozen is None or self.frozen[0] != state.vanish_started:
                self.frozen = (state.vanish_started, camera_frame.copy())
            progress = min(1.0, (now - state.vanish_started) / PEACE_SECONDS)
            display = cv2.addWeighted(self.frozen[1], 1.0 - progress, background, progress, 0)
            if progress >= 1.0:
                state.phase = "vanished"
        elif state.phase == "ready":
            display = camera_frame
        else:
            progress = 0.0
            if state.phase == "vanishing" and state.vanish_started is not None:
                progress = min(1.0, (now - state.vanish_started) / VANISH_SECONDS)
                if progress >= 1.0:
                    state.phase = "vanished"
            display = apply_dissolve(
                camera_frame, background, mask, progress, self.dissolve_field, self.sparkle_noise
            )
            if state.phase == "vanished":
                self.vanished_frame = display
        return display


class AppWindow:
    """Show the video and the controls in one window. The webcam and the effect can change at any time."""

    ACCENT = "#2f7ddc"
    ACTIVE = "#e8833a"
    PANEL = "#1d2026"
    CONTROL = "#30363e"
    FOREGROUND = "#eceff2"
    MUTED = "#b3c3c9"

    def __init__(self, args: argparse.Namespace, resources: ExitStack):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk

        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
        self.args = args
        self.language = "en"
        self.cameras = list_cameras() or [(0, "Webcam 0")]
        self.capture: cv2.VideoCapture | None = None
        self.virtual_camera = None
        self.virtual_camera_initialized = False
        self.photo = None
        self.status_override: str | None = None
        self.engine = VanishEngine(args)
        resources.callback(self.engine.close)
        resources.callback(self.release_camera)
        resources.callback(self.close_virtual_camera)
        self.meeting_audio: MeetingAudio | None = None
        resources.callback(self.stop_meeting_audio)

        self.root = tk.Tk()
        resources.callback(self.destroy)
        self.root.title("VanishCam")
        self.root.configure(bg=self.PANEL)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=self.CONTROL, background=self.CONTROL,
                        foreground=self.FOREGROUND, arrowcolor=self.FOREGROUND)
        style.map("TCombobox", fieldbackground=[("readonly", self.CONTROL)],
                  foreground=[("readonly", self.FOREGROUND)])
        self.root.option_add("*TCombobox*Listbox.background", self.CONTROL)
        self.root.option_add("*TCombobox*Listbox.foreground", self.FOREGROUND)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.root, width=args.width, height=args.height,
                                bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.image_item = self.canvas.create_image(0, 0, anchor="center")

        panel = tk.Frame(self.root, bg=self.PANEL, padx=16, pady=16)
        panel.grid(row=0, column=1, sticky="ns")
        self.panel = panel
        label_options = dict(bg=self.PANEL, fg=self.MUTED, anchor="w")

        def heading(row: int) -> tk.Label:
            label = tk.Label(panel, **label_options)
            label.grid(row=row, column=0, sticky="ew", pady=(10, 2))
            return label

        def choice(row: int, values, command) -> ttk.Combobox:
            box = ttk.Combobox(panel, state="readonly", values=values, width=30, takefocus=0)
            box.grid(row=row, column=0, sticky="ew")
            box.bind("<<ComboboxSelected>>", lambda event: (command(box.current()), self.root.focus_set()))
            return box

        def button(row: int, command, **options) -> tk.Button:
            widget = tk.Button(panel, command=command, takefocus=0, relief="flat", bd=0,
                               bg=self.CONTROL, fg=self.FOREGROUND, activebackground=self.ACCENT,
                               activeforeground="white", padx=10, pady=8, **options)
            widget.grid(row=row, column=0, sticky="ew", pady=(10, 0))
            return widget

        self.webcam_label = heading(0)
        self.webcam_box = choice(1, [f"{index}: {name}" for index, name in self.cameras], self.select_camera)
        self.video_mode_label = heading(2)
        self.video_mode_box = choice(3, [], self.select_video_mode)
        self.effect_label = heading(4)
        self.effect_box = choice(5, [MODE_NAMES[mode] for mode in MODES], self.select_mode)
        self.detail_label = tk.Label(panel, bg=self.PANEL, fg=self.MUTED, anchor="w", justify="left")
        self.detail_label.grid(row=6, column=0, sticky="ew")
        self.language_label = heading(7)
        self.language_box = choice(8, LANGUAGE_NAMES, self.select_language)
        self.background_button = button(9, self.capture_background, font=("TkDefaultFont", 11, "bold"))
        self.vanish_button = button(10, lambda: self.engine.vanish(time.perf_counter()))
        self.reappear_button = button(11, lambda: self.engine.reappear(time.perf_counter()))
        self.fullscreen_button = button(12, self.toggle_fullscreen)
        self.meeting_enabled = tk.BooleanVar(self.root, value=False)
        self.meeting_check = tk.Checkbutton(
            panel, variable=self.meeting_enabled, command=self.toggle_meeting_audio, takefocus=0,
            anchor="w", bg=self.PANEL, fg=self.FOREGROUND, selectcolor=self.CONTROL,
            activebackground=self.PANEL, activeforeground=self.FOREGROUND, highlightthickness=0,
        )
        self.meeting_check.grid(row=13, column=0, sticky="ew", pady=(12, 0))
        self.status_label = tk.Label(panel, bg=self.PANEL, fg=self.FOREGROUND, anchor="w",
                                     justify="left", wraplength=240)
        self.status_label.grid(row=14, column=0, sticky="ew", pady=(16, 0))
        panel.rowconfigure(15, weight=1)
        self.keys_label = tk.Label(panel, bg=self.PANEL, fg=self.MUTED, anchor="w",
                                   justify="left", wraplength=240, font=("TkDefaultFont", 8))
        self.keys_label.grid(row=16, column=0, sticky="sew")

        self.root.bind("<KeyPress>", self.on_key)
        self.fullscreen = False
        self.webcam_box.current(0)
        self.language_box.current(0)
        mode = args.mode or "normal"
        self.effect_box.current(MODES.index(mode))
        self.update_texts()
        self.camera_index = self.cameras[0][0]
        self.video_modes: list[tuple[str, int, int, float]] = []
        self.video_mode: tuple[str, int, int, float] | None = None
        self.load_video_modes()
        if getattr(args, "meeting_audio", False):
            # Make the microphone before the sound opens, so that the sound goes into the mix.
            self.meeting_enabled.set(True)
            self.start_meeting_audio()
        self.select_mode(MODES.index(mode))
        if args.fullscreen:
            self.toggle_fullscreen()

    def text(self, key: str) -> str:
        return TEXT[self.language][key]

    def run(self) -> None:
        self.root.after(1, self.tick)
        self.root.mainloop()

    def quit(self) -> None:
        self.root.quit()

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def release_camera(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def close_virtual_camera(self) -> None:
        if self.virtual_camera is not None:
            self.virtual_camera.close()
            self.virtual_camera = None

    def open_camera(self, mode: tuple[str, int, int, float]) -> None:
        self.release_camera()
        fourcc, width, height, fps = mode
        try:
            self.capture = open_camera(self.camera_index, width, height, fourcc, fps)
            self.video_mode = mode
            self.status_override = None
            print(f"[Webcam] Webcam {self.camera_index}: {video_mode_name(mode)}.")
        except RuntimeError as error:
            print(f"[Webcam] {error}")
            self.status_override = self.text("status_camera_error")

    def load_video_modes(self) -> None:
        """List the modes of the selected webcam and open the webcam in the default mode."""
        self.video_modes = list_video_modes(self.camera_index)
        mode = default_video_mode(self.video_modes, self.args.width, self.args.height, self.args.fps)
        self.video_mode_box.configure(values=[video_mode_name(item) for item in self.video_modes])
        self.video_mode_box.current(self.video_modes.index(mode))
        self.open_camera(mode)

    def select_camera(self, position: int) -> None:
        self.camera_index = self.cameras[position][0]
        self.load_video_modes()
        # The old background is from a different view.
        self.engine.forget_background()

    def select_video_mode(self, position: int) -> None:
        old = self.video_mode
        self.open_camera(self.video_modes[position])
        if old is None or old[1:3] != self.video_modes[position][1:3]:
            # The background must have the size of the new frames.
            self.engine.forget_background()

    def select_mode(self, position: int) -> None:
        mode = MODES[position]
        self.status_label.configure(text=self.text("status_loading"))
        self.root.update_idletasks()
        try:
            self.engine.set_mode(mode)
            self.status_override = None
        except Exception as error:
            print(f"[Webcam] Cannot load the mode {mode}: {error}")
            self.status_override = self.text("status_mode_error")
            if self.engine.mode in MODES:
                self.effect_box.current(MODES.index(self.engine.mode))
        self.update_texts()

    def start_meeting_audio(self) -> None:
        audio = MeetingAudio()
        try:
            audio.open()
        except RuntimeError as error:
            print(f"[Webcam] {error}")
            self.meeting_enabled.set(False)
            self.status_override = self.text("status_meeting_error")
            return
        self.meeting_audio = audio
        self.status_override = None
        print(f"[Webcam] Select the microphone 'VanishMic' ({MIC_SOURCE}) in the meeting app.")

    def stop_meeting_audio(self) -> None:
        if self.meeting_audio is not None:
            self.meeting_audio.close()
            self.meeting_audio = None
            print("[Webcam] VanishMic is removed.")

    def toggle_meeting_audio(self) -> None:
        """Start or stop VanishMic. Then open the sound again, so that it uses the new output."""
        if self.meeting_enabled.get():
            self.start_meeting_audio()
        else:
            self.stop_meeting_audio()
        self.engine.reload_sound()
        self.root.focus_set()

    def select_language(self, position: int) -> None:
        self.language = LANGUAGES[position]
        self.update_texts()

    def capture_background(self) -> None:
        self.engine.capture_background(time.perf_counter())

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)
        if self.fullscreen:
            self.panel.grid_remove()
        else:
            self.panel.grid()

    def on_key(self, event) -> None:
        key = event.keysym.lower()
        now = time.perf_counter()
        if key == "q":
            self.quit()
        elif key == "escape" and self.fullscreen:
            self.toggle_fullscreen()
        elif key == "space":
            self.capture_background()
        elif key == "v":
            self.engine.vanish(now)
        elif key == "r":
            self.engine.reappear(now)
        elif key == "f":
            self.toggle_fullscreen()

    def update_texts(self) -> None:
        self.webcam_label.configure(text=self.text("webcam"))
        self.video_mode_label.configure(text=self.text("video_mode"))
        self.effect_label.configure(text=self.text("effect"))
        self.language_label.configure(text=self.text("language"))
        mode = self.engine.mode or MODES[self.effect_box.current()]
        self.detail_label.configure(text=self.text(f"{mode}_detail"))
        self.vanish_button.configure(text=self.text("vanish"))
        self.reappear_button.configure(text=self.text("reappear"))
        self.fullscreen_button.configure(text=self.text("fullscreen"))
        self.keys_label.configure(text=self.text("keys"))
        self.meeting_check.configure(text=self.text("meeting"))
        self.update_controls(time.perf_counter())

    def update_controls(self, now: float) -> None:
        phase = self.engine.state.phase
        if phase == "calibrating":
            text = self.text("capturing").format(self.engine.seconds_left(now))
            color = self.ACTIVE
        elif phase == "no_background":
            text, color = self.text("capture"), self.ACCENT
        else:
            text, color = self.text("recapture"), self.CONTROL
        self.background_button.configure(text=text, bg=color)
        if self.status_override:
            status = self.status_override
        elif phase in ("no_background", "calibrating", "ready", "vanished"):
            status = self.text(f"status_{phase}")
        else:
            status = self.text("status_effect")
        self.status_label.configure(text=status)
        usable = "normal" if self.engine.background is not None and phase != "calibrating" else "disabled"
        self.vanish_button.configure(state=usable)
        self.reappear_button.configure(state=usable)

    def show(self, frame: np.ndarray) -> None:
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = min(width / frame.shape[1], height / frame.shape[0])
        if scale > 0 and abs(scale - 1.0) > 0.01:
            size = (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale)))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        image = self.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.photo is not None and (self.photo.width(), self.photo.height()) == image.size:
            self.photo.paste(image)
        else:
            self.photo = self.ImageTk.PhotoImage(image)
            self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.canvas.coords(self.image_item, width // 2, height // 2)

    def tick(self) -> None:
        """Process one webcam frame. Tk calls this function again after each frame."""
        if self.capture is None:
            self.update_controls(time.perf_counter())
            self.root.after(200, self.tick)
            return
        ok, camera_frame = self.capture.read()
        if not ok:
            self.status_override = self.text("status_camera_error")
            self.update_controls(time.perf_counter())
            self.root.after(200, self.tick)
            return
        if not self.args.no_mirror:
            camera_frame = cv2.flip(camera_frame, 1)
        now = time.perf_counter()
        display = self.engine.process(camera_frame, now)
        if not self.virtual_camera_initialized:
            self.virtual_camera = open_virtual_camera(self.args, display.shape[1], display.shape[0])
            self.virtual_camera_initialized = True
        if self.virtual_camera is not None:
            # Only the local preview is a mirror. Video call apps mirror the self-view themselves.
            send_virtual_frame(self.virtual_camera, display if self.args.no_mirror else cv2.flip(display, 1))
        self.show(display)
        self.update_controls(now)
        if self.virtual_camera is not None:
            self.virtual_camera.sleep_until_next_frame()
        self.root.after(1, self.tick)


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    with ExitStack() as resources:
        window = AppWindow(args, resources)
        print("[Webcam] Click 'Capture background', then move out of the camera view for 4 seconds.")
        print("[Webcam] Keys in the window: SPACE new background, R reappear, V vanish, F full screen, Q quit.")
        window.run()
    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[Vanish Lab] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
