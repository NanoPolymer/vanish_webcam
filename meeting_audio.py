"""Virtual microphone for meetings: the real microphone plus the effect sound.

The app plays the effect into the "vanish_fx" sink. Loopbacks send it to the
speakers and into the "vanish_mix" sink, together with the real microphone.
Port links (pw-link) connect "vanish_mix" to "VanishMic", which the meeting
app records. VanishMic is a null sink with the class Audio/Source/Virtual.
PipeWire gives it the HARDWARE flag, so KDE Plasma shows it as a normal input
also when "Show virtual devices" is off. A remap-source or a loopback source
does not have this flag. This needs PipeWire (pipewire-pulse and pw-link).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time


MIX_SINK = "vanish_mix"
EFFECT_SINK = "vanish_fx"
MIC_SOURCE = "vanish_mic"
NAMES = (MIX_SINK, EFFECT_SINK, MIC_SOURCE)


def pactl(*args: str) -> str:
    try:
        result = subprocess.run(["pactl", *args], capture_output=True, text=True,
                                check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"pactl {' '.join(args)} failed: {error}") from error
    return result.stdout.strip()


def pw_link(output: str, input_: str) -> bool:
    try:
        return subprocess.run(["pw-link", output, input_], capture_output=True,
                              timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class MeetingAudio:
    def __init__(self, run=pactl, link=pw_link, wait=time.sleep):
        self.run = run
        self.link = link
        self.wait = wait
        self.modules: list[str] = []

    def open(self) -> None:
        if platform.system() != "Linux" or not (shutil.which("pactl") and shutil.which("pw-link")):
            raise RuntimeError(
                "--meeting-audio needs Linux with PipeWire, pactl and pw-link. "
                "On Windows, use VB-Audio Virtual Cable or Voicemeeter."
            )
        self.remove_leftovers()
        microphone = self.run("get-default-source")
        if microphone.endswith(".monitor") or microphone.startswith("vanish_"):
            raise RuntimeError(
                f"The default input is '{microphone}', not a real microphone. "
                "Select your microphone as the default input in the sound settings."
            )
        try:
            self.load("module-null-sink", "media.class=Audio/Source/Virtual", f"sink_name={MIC_SOURCE}",
                      "sink_properties=device.description=VanishMic", "channel_map=front-left,front-right")
            self.load("module-null-sink", f"sink_name={MIX_SINK}",
                      "sink_properties=device.description=VanishMix")
            self.load("module-loopback", f"source={microphone}", f"sink={MIX_SINK}", "latency_msec=20")
            self.load("module-null-sink", f"sink_name={EFFECT_SINK}",
                      "sink_properties=device.description=VanishFX")
            self.load("module-loopback", f"source={EFFECT_SINK}.monitor", f"sink={MIX_SINK}",
                      "latency_msec=20")
            # Without a sink argument, the loopback uses the speakers or the headset.
            self.load("module-loopback", f"source={EFFECT_SINK}.monitor", "latency_msec=20")
            self.connect_mix_to_mic()
        except RuntimeError:
            self.close()
            raise
        # SDL (pygame) and ffplay read these variables when the audio opens.
        os.environ["PULSE_SINK"] = EFFECT_SINK
        os.environ.setdefault("SDL_AUDIODRIVER", "pulseaudio")

    def connect_mix_to_mic(self) -> None:
        """Link the mix to the microphone. The links go away with the modules."""
        for channel in ("FL", "FR"):
            # The ports can appear a short time after the module loads.
            for _ in range(20):
                if self.link(f"{MIX_SINK}:monitor_{channel}", f"{MIC_SOURCE}:input_{channel}"):
                    break
                self.wait(0.1)
            else:
                raise RuntimeError(f"Cannot link {MIX_SINK} to {MIC_SOURCE} ({channel}).")

    def load(self, module: str, *args: str) -> None:
        self.modules.append(self.run("load-module", module, *args))

    def remove_leftovers(self) -> None:
        """Unload the modules of an earlier run that did not stop correctly."""
        for line in self.run("list", "short", "modules").splitlines():
            fields = line.split("\t")
            if len(fields) >= 3 and any(name in fields[2] for name in NAMES):
                self.run("unload-module", fields[0])

    def close(self) -> None:
        # Send the effect sound to the default output again when the audio opens next time.
        if os.environ.get("PULSE_SINK") == EFFECT_SINK:
            del os.environ["PULSE_SINK"]
        while self.modules:
            try:
                self.run("unload-module", self.modules.pop())
            except RuntimeError:
                pass  # The next start removes the leftovers.
