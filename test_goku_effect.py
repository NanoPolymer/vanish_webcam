import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from local_app import vanish_cam as app
from local_app.goku_effect import (
    HoldTimer, fingers_on_forehead, make_teleport_field, render_teleport,
)
from local_app.test_video_output import arguments


WIDTH, HEIGHT = 640, 480
# Face box (left, top, width, height) and eye height, in pixels.
FACE = (270, 60, 100, 120, 110)


def hand(index_tip, middle_tip, base_y=0.30, bent=False):
    """Make 21 normalized landmarks with straight index and middle fingers."""
    points = [(0.5, base_y + 0.12)] * 21
    for base, start_x, tip in ((5, 0.49, index_tip), (9, 0.51, middle_tip)):
        for step in range(4):
            share = step / 3
            x = start_x + (tip[0] - start_x) * share
            y = base_y + (tip[1] - base_y) * share
            if bent and step == 2:
                x += 0.06
            points[base + step] = (x, y)
    return [SimpleNamespace(x=x, y=y) for x, y in points]


class GestureTests(unittest.TestCase):
    def test_two_fingers_together_on_the_forehead(self):
        together = hand((0.495, 0.18), (0.505, 0.18))
        self.assertTrue(fingers_on_forehead([together], FACE, WIDTH, HEIGHT))

    def test_other_poses_do_not_start_the_effect(self):
        cases = {
            "v sign": hand((0.44, 0.18), (0.56, 0.18)),
            "below the eyes": hand((0.495, 0.30), (0.505, 0.30), base_y=0.42),
            "bent finger": hand((0.495, 0.18), (0.505, 0.18), bent=True),
        }
        for name, landmarks in cases.items():
            with self.subTest(name):
                self.assertFalse(fingers_on_forehead([landmarks], FACE, WIDTH, HEIGHT))

    def test_no_face_means_no_forehead(self):
        together = hand((0.495, 0.18), (0.505, 0.18))
        self.assertFalse(fingers_on_forehead([together], None, WIDTH, HEIGHT))

    def test_detector_uses_face_box_and_eye_keypoints(self):
        together = hand((0.495, 0.18), (0.505, 0.18))
        face = SimpleNamespace(
            bounding_box=SimpleNamespace(origin_x=270, origin_y=60, width=100, height=120),
            keypoints=[SimpleNamespace(x=0.47, y=110 / HEIGHT), SimpleNamespace(x=0.53, y=110 / HEIGHT)],
        )
        detector = app.GokuGestureDetector(
            SimpleNamespace(detect_for_video=lambda image, ts: SimpleNamespace(hand_landmarks=[together])),
            SimpleNamespace(detect_for_video=lambda image, ts: SimpleNamespace(detections=[face])),
        )
        image = SimpleNamespace(width=WIDTH, height=HEIGHT)
        self.assertTrue(detector.detect(image, 1000))
        self.assertTrue(detector.face_found)


class HoldTimerTests(unittest.TestCase):
    def test_hold_for_one_second_then_release_before_next_trigger(self):
        timer = HoldTimer()
        steps = [(True, 0.0, False), (True, 0.5, False), (True, 1.0, True),
                 (True, 2.5, False),  # The pose did not stop.
                 (False, 2.6, False), (True, 2.7, False), (True, 3.7, True)]
        for pose, now, expected in steps:
            self.assertEqual(timer.update(pose, now), expected, (pose, now))

    def test_break_starts_the_countdown_again(self):
        timer = HoldTimer()
        for pose, now in ((True, 0.0), (False, 0.5), (True, 0.6)):
            timer.update(pose, now)
        self.assertFalse(timer.update(True, 1.5))
        self.assertTrue(timer.update(True, 1.6))


class TeleportRenderTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((60, 80, 3), 200, np.uint8)
        self.background = np.full((60, 80, 3), 10, np.uint8)
        self.mask = np.zeros((60, 80), np.float32)
        self.mask[10:50, 20:60] = 1.0
        self.field = make_teleport_field(60, 80)

    def render(self, progress):
        return render_teleport(self.frame, self.background, self.mask, self.field, progress)

    def test_start_and_end(self):
        start = self.render(0.0)
        np.testing.assert_array_equal(start[10:50, 20:60], self.frame[10:50, 20:60])
        np.testing.assert_array_equal(self.render(1.0), self.background)

    def test_slits_show_the_background_only_inside_the_body(self):
        middle = self.render(0.5)
        body = middle[10:50, 20:60]
        self.assertTrue((body == 10).any())  # Background through the slits.
        self.assertTrue((body > 10).any())  # Parts of the body stay.
        outside = middle.copy()
        outside[10:50, 20:60] = 10
        np.testing.assert_array_equal(outside, self.background)

    def test_field_is_fixed_and_complete_before_the_end(self):
        np.testing.assert_array_equal(self.field, make_teleport_field(60, 80))
        self.assertLess(float(self.field.max()), 1.0)


class GokuVideoLoopTests(unittest.TestCase):
    def test_gesture_vanishes_and_returns(self):
        # Frame n arrives at n * 0.25 seconds. Calibration ends at frame 16.
        clock = [0]

        def read():
            clock[0] += 1
            value = 10 if clock[0] <= 16 else 200
            return True, np.full((24, 40, 3), value, np.uint8)

        def pose(image, timestamp):
            return 17 <= clock[0] <= 21 or 27 <= clock[0] <= 31

        capture = Mock()
        capture.read.side_effect = read
        segmenter, detector = Mock(), Mock(face_found=True)
        detector.detect.side_effect = pose
        sound = Mock()
        shown = []
        keys = iter([0] * 34 + [ord("q")])
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(mode="goku")))
            mocks.enter_context(patch.object(app, "list_cameras", return_value=[(0, "A")]))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, detector)))
            mocks.enter_context(patch.object(app, "VanishSound", return_value=sound))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=None))
            mocks.enter_context(patch.object(app, "person_mask", return_value=np.ones((24, 40), np.float32)))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=lambda: clock[0] * 0.25))
            for name in ("namedWindow", "setWindowTitle", "resizeWindow", "destroyAllWindows"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: shown.append(frame.copy())))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=lambda delay: next(keys)))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)

        shown = {n: frame for n, frame in enumerate(shown, start=1)}
        self.assertTrue((shown[20] == 200).all())  # The countdown is not complete.
        self.assertTrue((shown[22] == 10).any() and (shown[22] == 200).any())  # Slits.
        self.assertTrue((shown[26] == 10).all())  # Vanished.
        self.assertTrue((shown[32] == 10).any() and (shown[32] == 200).any())  # Return.
        self.assertTrue((shown[35] == 200).all())  # Visible again.
        self.assertEqual(sound.play.call_count, 2)
        sound.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
