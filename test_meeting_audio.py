import os
import unittest
from unittest.mock import patch

from local_app import meeting_audio
from local_app.meeting_audio import MeetingAudio


class FakePactl:
    def __init__(self, source="alsa_input.usb-mic", modules="", fail_on=None):
        self.source = source
        self.modules = modules
        self.fail_on = fail_on
        self.calls = []
        self.next_id = 100

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "get-default-source":
            return self.source
        if args[:3] == ("list", "short", "modules"):
            return self.modules
        if args[0] == "load-module":
            if args[1] == self.fail_on:
                raise RuntimeError("load failed")
            self.next_id += 1
            return str(self.next_id)
        return ""


class MeetingAudioTests(unittest.TestCase):
    def setUp(self):
        mocks = [
            patch.object(meeting_audio.platform, "system", return_value="Linux"),
            patch.object(meeting_audio.shutil, "which", return_value="/usr/bin/pactl"),
            patch.dict(os.environ, {}, clear=False),
        ]
        for mock in mocks:
            mock.start()
            self.addCleanup(mock.stop)
        os.environ.pop("PULSE_SINK", None)

    def test_open_routes_microphone_and_effect_then_close_unloads_only_own_modules(self):
        fake = FakePactl()
        audio = MeetingAudio(fake)
        audio.open()
        loaded = [call for call in fake.calls if call[0] == "load-module"]
        self.assertEqual(len(loaded), 6)
        self.assertIn(("load-module", "module-loopback", "source=alsa_input.usb-mic",
                       "sink=vanish_mix", "latency_msec=20"), loaded)
        self.assertEqual(os.environ["PULSE_SINK"], "vanish_fx")
        audio.close()
        unloaded = [call[1] for call in fake.calls if call[0] == "unload-module"]
        self.assertEqual(unloaded, ["106", "105", "104", "103", "102", "101"])

    def test_leftovers_of_an_earlier_run_are_removed(self):
        modules = ("7\tmodule-null-sink\tsink_name=vanish_mix\n"
                   "8\tmodule-alsa-card\tdevice_id=0\n"
                   "9\tmodule-loopback\tsource=vanish_fx.monitor")
        fake = FakePactl(modules=modules)
        MeetingAudio(fake).open()
        unloaded = [call[1] for call in fake.calls if call[0] == "unload-module"]
        self.assertEqual(unloaded, ["7", "9"])

    def test_monitor_or_own_mic_as_default_input_is_refused(self):
        for source in ("alsa_output.speakers.monitor", "vanish_mic"):
            with self.subTest(source), self.assertRaisesRegex(RuntimeError, "not a real microphone"):
                MeetingAudio(FakePactl(source=source)).open()

    def test_failure_unloads_the_modules_that_were_loaded(self):
        fake = FakePactl(fail_on="module-remap-source")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            MeetingAudio(fake).open()
        self.assertIn(("unload-module", "101"), fake.calls)
        self.assertNotIn("PULSE_SINK", os.environ)

    def test_other_systems_get_a_clear_error(self):
        with patch.object(meeting_audio.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(RuntimeError, "VB-Audio"):
                MeetingAudio(FakePactl()).open()


if __name__ == "__main__":
    unittest.main()
