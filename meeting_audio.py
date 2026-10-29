"""Virtual microphone for meetings: the real microphone plus the effect sound.

The app plays the effect into the "vanish_fx" sink. Loopbacks send it to the
speakers and into the "vanish_mix" sink, together with the real microphone.
A loopback sends "vanish_mix" to "VanishMic", which the meeting app records.
The playback side of that loopback is an Audio/Source node, so desktop sound
settings show VanishMic as a normal input. This needs PipeWire (pipewire-pulse).
A remap-source is not used: it has the "filter" device class, and KDE Plasma
does not show it as an input.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess


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


class MeetingAudio:
    def __init__(self, run=pactl):
        self.run = run
        self.modules: list[str] = []

    def open(self) -> None:
        if platform.system() != "Linux" or shutil.which("pactl") is None:
            raise RuntimeError(
                "--meeting-audio needs Linux with PipeWire and pactl. "
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
            self.load("module-null-sink", f"sink_name={MIX_SINK}",
                      "sink_properties=device.description=VanishMix")
            self.load("module-loopback", f"source={MIX_SINK}.monitor", "latency_msec=20",
                      f'sink_input_properties="media.class=Audio/Source node.name={MIC_SOURCE} '
                      'node.description=VanishMic"')
            self.load("module-loopback", f"source={microphone}", f"sink={MIX_SINK}", "latency_msec=20")
            self.load("module-null-sink", f"sink_name={EFFECT_SINK}",
                      "sink_properties=device.description=VanishFX")
            self.load("module-loopback", f"source={EFFECT_SINK}.monitor", f"sink={MIX_SINK}",
                      "latency_msec=20")
            # Without a sink argument, the loopback uses the speakers or the headset.
            self.load("module-loopback", f"source={EFFECT_SINK}.monitor", "latency_msec=20")
        except RuntimeError:
            self.close()
            raise
        # SDL (pygame) and ffplay read these variables when the audio opens.
        os.environ["PULSE_SINK"] = EFFECT_SINK
        os.environ.setdefault("SDL_AUDIODRIVER", "pulseaudio")

    def load(self, module: str, *args: str) -> None:
        self.modules.append(self.run("load-module", module, *args))

    def remove_leftovers(self) -> None:
        """Unload the modules of an earlier run that did not stop correctly."""
        for line in self.run("list", "short", "modules").splitlines():
            fields = line.split("\t")
            if len(fields) >= 3 and any(name in fields[2] for name in NAMES):
                self.run("unload-module", fields[0])

    def close(self) -> None:
        while self.modules:
            try:
                self.run("unload-module", self.modules.pop())
            except RuntimeError:
                pass  # The next start removes the leftovers.
