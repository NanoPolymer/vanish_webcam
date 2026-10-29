import argparse
import struct
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from local_app import vanish_cam as app


def arguments(**overrides):
    values = dict(
        width=960, height=540, fps=30.0, fullscreen=False,
        no_mirror=True, virtual_camera=None, virtual_backend=None, virtual_device=None,
        gesture_debug=False, mode="normal",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class VirtualCameraTests(unittest.TestCase):
    def test_bgr_format_and_actual_frame_dimensions(self):
        camera = Mock(device="OBS Virtual Camera")
        module = SimpleNamespace(Camera=Mock(return_value=camera), PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIs(app.open_virtual_camera(arguments(), 640, 480), camera)
        module.Camera.assert_called_once_with(
            width=640, height=480, fps=30.0, fmt="BGR", backend=None, device=None,
        )

    def test_unavailable_driver_auto_falls_back_but_explicit_mode_fails(self):
        module = SimpleNamespace(Camera=Mock(side_effect=RuntimeError("driver missing")),
                                 PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIsNone(app.open_virtual_camera(arguments(), 640, 480))
            for options in (dict(virtual_camera=True), dict(virtual_backend="obs")):
                with self.assertRaisesRegex(RuntimeError, "driver missing"):
                    app.open_virtual_camera(arguments(**options), 640, 480)

    def test_disabled_camera_does_not_open_device(self):
        module = SimpleNamespace(Camera=Mock())
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIsNone(app.open_virtual_camera(arguments(virtual_camera=False), 640, 480))
        module.Camera.assert_not_called()

    def test_send_preserves_colors_and_handles_noncontiguous_frames(self):
        camera = Mock(width=40, height=24)
        frame = np.zeros((24, 40, 3), np.uint8)
        frame[:] = (10, 70, 200)  # BGR: do not swap red and blue.
        app.send_virtual_frame(camera, frame[:, ::-1])
        sent = camera.send.call_args.args[0]
        self.assertTrue(sent.flags.c_contiguous)
        np.testing.assert_array_equal(sent, frame)

    def test_send_resizes_if_input_resolution_changes(self):
        camera = Mock(width=40, height=24)
        app.send_virtual_frame(camera, np.full((48, 80, 3), 42, np.uint8))
        sent = camera.send.call_args.args[0]
        self.assertEqual(sent.shape, (24, 40, 3))
        self.assertTrue((sent == 42).all())


def make_engine(mocks, mode="normal", **options):
    """Make an engine with mocked models. Return the engine, the segmenter and the recognizer."""
    segmenter, recognizer = Mock(), Mock()
    mocks.enter_context(patch.object(app, "download_models"))
    mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
    engine = app.VanishEngine(arguments(mode=mode, **options))
    engine.set_mode(mode)
    return engine, segmenter, recognizer


class EngineTests(unittest.TestCase):
    def run_frames(self, gesture=("No_Hand", 0.0), mode="normal"):
        frames = [np.full((24, 40, 3), color, np.uint8) for color in (10, 40, 80, 120, 160)]
        times = (0.1, 4.1, 4.3, 5.0, 8.2)
        with ExitStack() as mocks:
            mocks.enter_context(patch("builtins.print"))
            mocks.enter_context(patch.object(app, "person_mask", return_value=np.ones((24, 40), np.float32)))
            mocks.enter_context(patch.object(app, "recognize_gesture", return_value=gesture))
            mocks.enter_context(patch.object(app, "VanishSound"))
            engine, segmenter, recognizer = make_engine(mocks, mode)
            engine.capture_background(0.0)
            shown = []
            for frame, now in zip(frames, times):
                shown.append(engine.process(frame, now).copy())
                if now == 4.3 and gesture[0] != "Victory":
                    engine.vanish(4.35)
            engine.close()
        segmenter.close.assert_called_once()
        recognizer.close.assert_called_once()
        return frames, shown

    def test_frames_in_all_phases(self):
        frames, shown = self.run_frames()
        np.testing.assert_array_equal(shown[0], frames[0])  # Calibration.
        np.testing.assert_array_equal(shown[2], frames[2])  # Ready.
        np.testing.assert_array_equal(shown[4], frames[1])  # Vanished: background only.

    def test_first_gesture_starts_dissolve(self):
        frames, shown = self.run_frames(gesture=("Victory", 0.9))
        self.assertTrue(np.any(shown[3] != frames[3]))  # The animation is not complete.
        self.assertTrue(np.any(shown[3] != frames[1]))  # The person is still visible.
        np.testing.assert_array_equal(shown[4], frames[1])  # The animation is complete.

    def test_peace_freezes_the_first_frame_and_fades_to_the_background(self):
        frames, shown = self.run_frames(gesture=("Victory", 0.9), mode="peace")
        np.testing.assert_array_equal(shown[2], frames[2])  # The frozen frame.
        # 0.7 s of 1.4 s: half of the frozen frame (80) and half of the background (40).
        self.assertTrue((np.abs(shown[3].astype(int) - 60) <= 1).all())
        np.testing.assert_array_equal(shown[4], frames[1])

    def test_no_background_shows_the_camera(self):
        with ExitStack() as mocks:
            engine, _, recognizer = make_engine(mocks)
            frame = np.full((24, 40, 3), 7, np.uint8)
            engine.vanish(1.0)  # No background: no effect.
            np.testing.assert_array_equal(engine.process(frame, 1.0), frame)
            self.assertEqual(engine.state.phase, "no_background")
            recognizer.recognize_gesture.assert_not_called()

    def test_mode_change_keeps_background_and_closes_old_models(self):
        with ExitStack() as mocks:
            mocks.enter_context(patch("builtins.print"))
            mocks.enter_context(patch.object(app, "VanishSound"))
            engine, segmenter, recognizer = make_engine(mocks)
            engine.capture_background(0.0)
            engine.process(np.full((24, 40, 3), 5, np.uint8), 4.1)
            engine.state.phase = "vanished"
            engine.set_mode("peace")
            segmenter.close.assert_called_once()
            recognizer.close.assert_called_once()
            self.assertEqual(engine.mode, "peace")
            self.assertEqual(engine.state.phase, "ready")
            self.assertTrue((engine.background == 5).all())
            engine.forget_background()
            self.assertIsNone(engine.background)
            self.assertEqual(engine.state.phase, "no_background")


class VideoModeTests(unittest.TestCase):
    def fake_driver(self, formats):
        """formats: {fourcc: {(width, height): [fps, ...]}}. Answer the V4L2 list requests like a driver."""
        codes = {name: int.from_bytes(name.encode(), "little") for name in formats}

        def ioctl(device, request, buffer):
            index, code = struct.unpack_from("<II", buffer, 0)
            if request == app.VIDIOC_ENUM_FMT:
                names = list(formats)
                if index >= len(names):
                    raise OSError("EINVAL")
                struct.pack_into("<I", buffer, 44, codes[names[index]])
                return
            name = next(key for key, value in codes.items() if value == code)
            sizes = list(formats[name])
            if request == app.VIDIOC_ENUM_FRAMESIZES:
                if index >= len(sizes):
                    raise OSError("EINVAL")
                struct.pack_into("<III", buffer, 8, app.V4L2_DISCRETE, *sizes[index])
                return
            width, height = struct.unpack_from("<II", buffer, 8)
            rates = formats[name][(width, height)]
            if index >= len(rates):
                raise OSError("EINVAL")
            struct.pack_into("<III", buffer, 16, app.V4L2_DISCRETE, 1, rates[index])

        return ioctl

    def test_modes_prefer_mjpg_and_usual_rates(self):
        ioctl = self.fake_driver({
            "YUYV": {(640, 480): [30, 15], (320, 240): [30]},
            "MJPG": {(1280, 720): [60, 30, 24], (640, 480): [30], (800, 600): [20, 10]},
            "H264": {(1920, 1080): [30]},
        })
        modes = app.read_video_modes(ioctl, -1)
        self.assertEqual(modes, [
            ("MJPG", 1280, 720, 60.0), ("MJPG", 1280, 720, 30.0),
            ("MJPG", 800, 600, 20.0), ("MJPG", 640, 480, 30.0),
        ])
        self.assertEqual(app.default_video_mode(modes, 1280, 720, 30.0), ("MJPG", 1280, 720, 30.0))
        self.assertEqual(app.default_video_mode(modes, 960, 540, 30.0), ("MJPG", 640, 480, 30.0))  # 30 fps before size.
        self.assertEqual(app.video_mode_name(modes[0]), "1280x720  60 fps  MJPG")


def has_display():
    try:
        import tkinter
        tkinter.Tk().destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(has_display(), "Tk needs a display")
class WindowTests(unittest.TestCase):
    def test_window_sends_clean_unmirrored_frames_and_closes_devices(self):
        frame = np.full((24, 40, 3), 30, np.uint8)
        frame[:, 0] = 255  # The frame is not symmetrical, so a mirror image is different.
        capture = Mock()
        capture.read.return_value = (True, frame)
        virtual = Mock(width=40, height=24)
        with ExitStack() as mocks:
            mocks.enter_context(patch("builtins.print"))
            mocks.enter_context(patch.object(app, "list_cameras", return_value=[(3, "A"), (5, "B")]))
            mocks.enter_context(patch.object(app, "list_video_modes", return_value=[
                ("MJPG", 1280, 720, 30.0), ("MJPG", 40, 24, 30.0), ("YUYV", 40, 24, 60.0)]))
            opening = mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=virtual))
            mocks.enter_context(patch.object(app, "recognize_gesture", return_value=("No_Hand", 0.0)))
            engine_mocks = ExitStack()
            mocks.enter_context(engine_mocks)
            engine_mocks.enter_context(patch.object(app, "download_models"))
            tasks = (Mock(), Mock())
            engine_mocks.enter_context(patch.object(app, "create_tasks", return_value=tasks))
            with ExitStack() as resources:
                window = app.AppWindow(arguments(no_mirror=False), resources)
                window.root.withdraw()
                window.tick()
                window.tick()
                sent = virtual.send.call_args.args[0]
                np.testing.assert_array_equal(sent, frame)  # The preview is a mirror, the output is not.
                self.assertEqual(window.background_button.cget("text"), "Capture background")
                window.capture_background()
                window.tick()
                self.assertTrue(window.background_button.cget("text").startswith("Leave the view: "))
                window.select_language(1)
                self.assertEqual(window.effect_label.cget("text"), "Effetto")
                with patch.object(app, "MeetingAudio") as meeting, \
                        patch.object(window.engine, "reload_sound") as reload_sound:
                    window.meeting_check.invoke()  # On.
                    meeting.return_value.open.assert_called_once()
                    window.meeting_check.invoke()  # Off.
                    meeting.return_value.close.assert_called_once()
                    self.assertEqual(reload_sound.call_count, 2)
                    meeting.return_value.open.side_effect = RuntimeError("no pactl")
                    window.meeting_check.invoke()  # A failure leaves the box off.
                    self.assertFalse(window.meeting_enabled.get())
                    self.assertIsNone(window.meeting_audio)
                self.assertEqual(opening.call_args.args, (3, 1280, 720, "MJPG", 30.0))
                window.capture_background()
                window.select_video_mode(2)  # Other size: the background is removed.
                self.assertEqual(opening.call_args.args, (3, 40, 24, "YUYV", 60.0))
                self.assertEqual(window.engine.state.phase, "no_background")
                window.capture_background()
                window.select_video_mode(1)  # Same size: the capture continues.
                self.assertEqual(window.engine.state.phase, "calibrating")
                window.select_camera(1)
                self.assertEqual(opening.call_args.args[0], 5)
                self.assertEqual(window.engine.state.phase, "no_background")
            self.assertEqual(capture.release.call_count, 4)
            virtual.close.assert_called_once()
            for task in tasks:
                task.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
