"""Load an effect sound before the video starts. Play it while the video frames continue."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


SOUND_PATH = Path(__file__).resolve().parent / "assets" / "vanish.MP3"


def trim_leading_silence(samples: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Remove the silence at the start, so that the sound starts with the effect.

    The threshold is a fraction of the maximum sample value.
    """
    limit = threshold * (np.iinfo(samples.dtype).max if samples.dtype.kind in "iu" else 1.0)
    level = np.abs(samples.astype(np.float32))
    if level.ndim > 1:
        level = level.max(axis=1)
    loud = np.flatnonzero(level > limit)
    return samples[loud[0]:].copy() if loud.size else samples


class VanishSound:
    def __init__(self, path: Path = SOUND_PATH):
        if not path.is_file():
            raise RuntimeError(f"Cannot find the audio file: {path}")
        self.path = path

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        self.process = None
        self.ffplay = None
        try:
            from pygame import mixer
        except ImportError:
            self.ffplay = shutil.which("ffplay")
            if self.ffplay is None:
                raise RuntimeError(f"Install pygame or ffplay to play {path.name}.")
            return

        self.mixer = mixer
        self.owns_mixer = mixer.get_init() is None
        try:
            if self.owns_mixer:
                mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
            # Decode the file one time, before the webcam opens.
            from pygame import sndarray

            self.sound = sndarray.make_sound(trim_leading_silence(sndarray.array(mixer.Sound(str(path)))))
        except Exception as error:
            if self.owns_mixer:
                mixer.quit()
            raise RuntimeError(f"Cannot open the audio file {path.name}: {error}") from error

    def play(self) -> None:
        if self.ffplay is not None:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            try:
                self.process = subprocess.Popen(
                    [self.ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
                     "-af", "silenceremove=start_periods=1:start_threshold=-40dB",
                     "-i", str(self.path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as error:
                raise RuntimeError(f"Cannot start the audio: {error}") from error
            return
        self.sound.stop()
        if self.sound.play() is None:
            raise RuntimeError("Cannot start the audio: all audio channels are in use.")

    def close(self) -> None:
        if self.ffplay is not None:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            return
        if self.owns_mixer:
            self.mixer.quit()
            self.owns_mixer = False
