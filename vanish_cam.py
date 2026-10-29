"""Local webcam: a V sign with the index and middle finger starts the dissolve."""

from __future__ import annotations

import argparse
import os
import platform
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

# The OpenCV wheel sets QT_QPA_FONTDIR to a folder that does not exist, and Qt
# shows a warning. Qt reads the variable when the first window opens, so use the system fonts.
_font_dir = os.environ.get("QT_QPA_FONTDIR")
if sys.platform.startswith("linux") and not (_font_dir and os.path.isdir(_font_dir)) \
        and os.path.isdir("/usr/share/fonts"):
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "preview"
MENU_WINDOW = "menu"

# Text in the windows. The terminal output is always in English.
# The Hershey fonts of OpenCV show only ASCII characters.
LANGUAGES = ("en", "it")
TEXT = {
    "en": {
        "menu_title": "Choose effect",
        "menu_heading": "CHOOSE YOUR VANISH",
        "language": "L  LANGUAGE: ENGLISH",
        "language_detail": "Click or press L to change the language",
        "camera_detail": "Click or press C to change the webcam",
        "normal_detail": "V sign  /  dissolve",
        "naruto_detail": "Joined hands  /  log and smoke",
        "goku_detail": "Two fingers on the forehead for 1 s  /  teleport",
        "peace_detail": "V sign  /  freeze frame and fade",
        "preview_title": "Webcam - Preview",
    },
    "it": {
        "menu_title": "Scegli effetto",
        "menu_heading": "SCEGLI IL TUO VANISH",
        "language": "L  LINGUA: ITALIANO",
        "language_detail": "Clic o L per cambiare lingua",
        "camera_detail": "Clic o C per cambiare webcam",
        "normal_detail": "Segno a V  /  dissolvenza",
        "naruto_detail": "Mani unite  /  tronco e fumo",
        "goku_detail": "Due dita sulla fronte per 1 s  /  teletrasporto",
        "peace_detail": "Segno a V  /  fermo immagine e dissolvenza",
        "preview_title": "Webcam - Anteprima",
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
        description="Show a vanish effect on the webcam: Normal Vanish, Naruto Vanish or Goku Vanish."
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
        "--mode", choices=("normal", "naruto", "goku", "peace"),
        help="Skip the start menu and start the selected mode",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Show the gesture recognition status in the terminal. The video shows no text.",
    )
    parser.add_argument(
        "--meeting-audio", action="store_true",
        help="Make the virtual microphone VanishMic (Linux). It mixes your microphone and the effect sound.",
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


def choose_mode() -> tuple[str, int, str] | None:
    """Show a simple keyboard and mouse menu before the camera video starts.

    Return the mode, the webcam index and the language code.
    """
    cameras = list_cameras() or [(0, "Webcam 0")]
    selected = [None]
    camera = [0]
    language = [0]
    # Each row has an action, a key and a top position in pixels.
    rows = (("language", "l", 110), ("camera", "c", 210), ("normal", "1", 310),
            ("naruto", "2", 410), ("goku", "3", 510), ("peace", "4", 610))

    def text(key: str) -> str:
        return TEXT[LANGUAGES[language[0]]][key]

    def draw() -> np.ndarray:
        window = np.zeros((720, 700, 3), dtype=np.uint8)
        window[:] = (26, 29, 34)
        cv2.putText(window, text("menu_heading"), (68, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (236, 239, 242), 2, cv2.LINE_AA)
        index, name = cameras[camera[0]]
        headings = {
            "language": text("language"),
            "camera": f"C  WEBCAM {index}: {name[:30]}",
            "normal": "1  NORMAL VANISH",
            "naruto": "2  NARUTO VANISH",
            "goku": "3  GOKU VANISH",
            "peace": "4  PEACE OUT",
        }
        for action, _, top in rows:
            cv2.rectangle(window, (65, top), (635, top + 80), (48, 54, 62), -1, cv2.LINE_AA)
            cv2.rectangle(window, (65, top), (635, top + 80), (111, 142, 153), 2, cv2.LINE_AA)
            cv2.putText(window, headings[action], (90, top + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.77, (240, 245, 247), 2, cv2.LINE_AA)
            cv2.putText(window, text(f"{action}_detail"), (90, top + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.53, (179, 195, 201), 1, cv2.LINE_AA)
        return window

    def activate(action: str) -> None:
        if action == "language":
            language[0] = (language[0] + 1) % len(LANGUAGES)
            cv2.setWindowTitle(MENU_WINDOW, text("menu_title"))
        elif action == "camera":
            camera[0] = (camera[0] + 1) % len(cameras)
        else:
            selected[0] = action

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and 65 <= x <= 635:
            for action, _, top in rows:
                if top <= y <= top + 80:
                    activate(action)

    cv2.namedWindow(MENU_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowTitle(MENU_WINDOW, text("menu_title"))
    cv2.setMouseCallback(MENU_WINDOW, on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(MENU_WINDOW, draw())
            key = cv2.waitKey(30) & 0xFF
            for action, row_key, _ in rows:
                if key == ord(row_key):
                    activate(action)
            if key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(MENU_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0], cameras[camera[0]][0], LANGUAGES[language[0]]
    finally:
        cv2.destroyWindow(MENU_WINDOW)


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


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} is not available. Close the other apps that use it, or select a different webcam in the menu."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Make sure that the bundled resources and the models are available. Do not open a webcam."""
    required = (SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL, SOUND_PATH)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"These resources are missing: {', '.join(missing)}")
    load_log()
    import pygame.mixer  # The distributed build must contain its MP3 decoder.
    for mode in MODE_MODELS:
        segmenter, recognizer = create_tasks(mode)
        try:
            recognizer.close()
        finally:
            segmenter.close()
    print("[Webcam] The self-test is complete: the models, assets and audio are available.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    with ExitStack() as resources:
        if getattr(args, "meeting_audio", False):
            # Make the microphone before the menu, so that the meeting app can select it early.
            meeting_audio = MeetingAudio()
            meeting_audio.open()
            resources.callback(meeting_audio.close)
            print(f"[Webcam] Select the microphone 'VanishMic' ({MIC_SOURCE}) in the meeting app.")
        if args.mode:
            # Without the menu, use the first webcam that the system reports and English text.
            cameras = list_cameras(limit=1)
            choice = args.mode, cameras[0][0] if cameras else 0, "en"
        else:
            choice = choose_mode()
        if choice is None:
            return 0
        mode, camera, language = choice
        return run_camera(args, resources, mode, camera, language)


def run_camera(args: argparse.Namespace, resources: ExitStack, mode: str, camera: int,
               language: str = "en") -> int:
    download_models(mode)
    capture = open_camera(camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks(mode)
    resources.callback(segmenter.close)
    resources.callback(recognizer.close)
    log_sprite = load_log() if mode == "naruto" else None
    vanish_sound = None
    if mode in ("naruto", "goku", "peace"):
        # The effect works without audio, so a missing audio device must not stop the app.
        try:
            vanish_sound = VanishSound(
                {"naruto": SOUND_PATH, "goku": TELEPORT_SOUND, "peace": PEACE_SOUND}[mode]
            )
        except RuntimeError as error:
            print(f"[Webcam] The audio is off: {error}")
    if vanish_sound is not None:
        resources.callback(vanish_sound.close)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(WINDOW_NAME, TEXT[language]["preview_title"])
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    state = AppState(
        calibration_started=time.perf_counter(), fullscreen=args.fullscreen
    )
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    background_samples: list[np.ndarray] = []
    last_mask: np.ndarray | None = None
    last_mask_at = float("-inf")
    dissolve_field: np.ndarray | None = None
    sparkle_noise: np.ndarray | None = None
    teleport_field: np.ndarray | None = None
    goku_hold = HoldTimer()
    log_anchor: tuple[int, int, int] | None = None
    log_cache: dict[int, np.ndarray] = {}
    vanished_frame: np.ndarray | None = None
    # The Peace mode keeps the first frame of each vanish, with its start time.
    frozen: tuple[float, np.ndarray] | None = None
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    last_gesture_report = 0.0
    started = time.perf_counter()
    print("[Webcam] Move out of the camera view for 4 seconds. The app records the background.")
    print("[Webcam] Keys in the window: SPACE new background, R reappear, V vanish, F full screen, Q quit.")

    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped the supply of images.")

            if not args.no_mirror:
                camera_frame = cv2.flip(camera_frame, 1)

            height, width = camera_frame.shape[:2]
            if not virtual_camera_initialized:
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
            if mode == "normal" and (dissolve_field is None or dissolve_field.shape != (height, width)):
                dissolve_field, sparkle_noise = make_dissolve_fields(height, width)
            if mode == "goku" and (teleport_field is None or teleport_field.shape != (height, width)):
                teleport_field = make_teleport_field(height, width)
            now = time.perf_counter()

            elapsed_ms = int((now - started) * 1000)
            mp_image = None

            if (
                mode == "goku" and background is not None
                and state.phase in ("ready", "vanished")
                and now - state.last_gesture_check >= 0.08
            ):
                state.last_gesture_check = now
                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose = recognizer.detect(mp_image, elapsed_ms)
                if args.gesture_debug and now - last_gesture_report >= 1.0:
                    print(f"[Goku] face={recognizer.face_found}, fingers on forehead={pose}")
                    last_gesture_report = now
                if goku_hold.update(pose, now):
                    state.phase = "goku_vanishing" if state.phase == "ready" else "goku_returning"
                    state.vanish_started = now
                    vanished_frame = None
                    if vanish_sound:
                        vanish_sound.play()

            if state.phase == "calibrating":
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Background capture: {remaining} seconds.")
                    calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    background_samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    if background_samples:
                        stack = np.stack(background_samples[-18:], axis=0)
                        background = np.median(stack, axis=0).astype(np.uint8)
                    else:
                        background = camera_frame.copy()
                    background_samples.clear()
                    state.phase = "ready"
                    print("[Webcam] The background is ready. You can move back into the camera view.")
                display = camera_frame

            elif background is None:
                state.phase = "calibrating"
                state.calibration_started = now
                display = camera_frame

            elif state.phase == "vanished":
                if vanished_frame is None:
                    vanished_frame = background.copy()
                    if mode == "naruto" and log_anchor is not None:
                        overlay_log(vanished_frame, log_sprite, landed_log_anchor(log_anchor, height),
                                    cache=log_cache)
                display = vanished_frame

            else:
                mask = last_mask
                if (
                    state.phase == "ready" and mode != "goku"
                    and now - state.last_gesture_check >=
                    (0.08 if mode == "naruto" else GESTURE_INTERVAL_SECONDS)
                ):
                    state.last_gesture_check = now
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if mode in ("normal", "peace"):
                        name, score = recognize_gesture(recognizer, mp_image, elapsed_ms)
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(
                                f"[Gesture] {name}, score={score:.2f}, "
                                f"crop={recognizer.preferred_crop or 'full'}, "
                                f"rotation={recognizer.preferred_rotation or 0}"
                            )
                            last_gesture_report = now
                        if start_vanish_on_gesture(state, name, score, now) and vanish_sound:
                            vanish_sound.play()
                    else:
                        joined = recognizer.recognize_joined_hands(mp_image, elapsed_ms)
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(f"[Hands] found={recognizer.last_hand_count}, "
                                  f"joined={joined}, view={recognizer.preferred_view or 'full'}")
                            last_gesture_report = now
                        if start_naruto_on_join(state, joined, time.perf_counter()) and vanish_sound:
                            vanish_sound.play()

                if state.phase in ("vanishing", "naruto_vanishing", "goku_vanishing", "goku_returning"):
                    # The normal dissolve updates the silhouette at 20 Hz. The Naruto
                    # effect needs it only when the person is still visible.
                    needs_mask = (
                        (mode in ("normal", "goku") and (mask is None or now - last_mask_at >= 0.05))
                        or (mode == "naruto" and (log_anchor is None or now - state.vanish_started < 0.13))
                    )
                    if needs_mask:
                        if mp_image is None:
                            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                        last_mask = mask
                        last_mask_at = now
                    if mode == "naruto" and log_anchor is None:
                        log_anchor = substitution_anchor(mask)

                if mode == "naruto" and state.phase == "naruto_vanishing":
                    age = max(0.04, time.perf_counter() - state.vanish_started)
                    display = render_substitution(
                        camera_frame, background, mask, log_sprite, log_anchor, age, log_cache
                    )
                    if age >= EFFECT_SECONDS:
                        state.phase = "vanished"
                        vanished_frame = display
                elif mode == "goku" and state.phase in ("goku_vanishing", "goku_returning"):
                    age = now - state.vanish_started
                    progress = min(1.0, age / GOKU_SECONDS)
                    if state.phase == "goku_returning":
                        progress = 1.0 - progress
                    display = render_teleport(camera_frame, background, mask, teleport_field, progress)
                    if age >= GOKU_SECONDS:
                        state.phase = "vanished" if state.phase == "goku_vanishing" else "ready"
                        vanished_frame = None
                elif mode == "peace" and state.phase == "vanishing":
                    if frozen is None or frozen[0] != state.vanish_started:
                        frozen = (state.vanish_started, camera_frame.copy())
                    progress = min(1.0, (time.perf_counter() - state.vanish_started) / PEACE_SECONDS)
                    display = cv2.addWeighted(frozen[1], 1.0 - progress, background, progress, 0)
                    if progress >= 1.0:
                        state.phase = "vanished"
                elif state.phase == "ready":
                    display = camera_frame
                else:
                    progress = 0.0
                    if state.phase == "vanishing" and state.vanish_started is not None:
                        progress = min(
                            1.0, (time.perf_counter() - state.vanish_started) / VANISH_SECONDS
                        )
                        if progress >= 1.0:
                            state.phase = "vanished"
                    display = apply_dissolve(
                        camera_frame, background, mask, progress, dissolve_field, sparkle_noise
                    )
                    if state.phase == "vanished":
                        vanished_frame = display

            if virtual_camera is not None:
                # Only the local preview is a mirror. Video call apps mirror the self-view themselves.
                send_virtual_frame(virtual_camera, display if args.no_mirror else cv2.flip(display, 1))
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state.phase = "calibrating"
                state.calibration_started = now
                state.vanish_started = None
                log_anchor = None
                last_mask = None
                last_mask_at = float("-inf")
                vanished_frame = None
                background_samples.clear()
                calibration_remaining = None
                print("[Webcam] Move out of the camera view for 4 seconds. The app records the new background.")
            elif key == ord("r") and mode == "goku" and state.phase == "vanished":
                # In the Goku mode, R plays the teleport in reverse.
                state.phase = "goku_returning"
                state.vanish_started = time.perf_counter()
                last_mask = None
                if vanish_sound:
                    vanish_sound.play()
            elif key == ord("r"):
                if background is not None:
                    state.phase = "ready"
                    state.vanish_started = None
                    vanished_frame = None
                    log_anchor = None
                    last_mask = None
                    last_mask_at = float("-inf")
            elif key == ord("v") and background is not None:
                state.phase = {"naruto": "naruto_vanishing", "goku": "goku_vanishing"}.get(mode, "vanishing")
                state.vanish_started = time.perf_counter()
                if vanish_sound:
                    vanish_sound.play()
                if mode == "naruto":
                    log_anchor = None
                vanished_frame = None
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
    finally:
        resources.close()

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
