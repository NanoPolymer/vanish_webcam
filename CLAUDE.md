# VanishCam (fork)

This repository is a fork of `Nicovic17/vanish_webcam`. The `origin` remote is the fork `NanoPolymer/vanish_webcam`. The `upstream` remote is the original repository.

## Setup and tests

- Use Python 3.12. MediaPipe 1.0.1 has no wheels for newer Python versions.
- Make the environment with `uv venv -p 3.12 .venv-local` and `uv pip install -p .venv-local -r requirements.txt`.
- The tests import the package `local_app`. Make a symlink with the name `local_app` that points to this folder, in a different folder.
- Run the tests from the folder that contains the symlink:
  `.venv-local/bin/python -m unittest local_app.test_meeting_audio local_app.test_goku_effect local_app.test_vanish_cam local_app.test_naruto_effect local_app.test_video_output`
- Run `.venv-local/bin/python vanish_cam.py --self-test` to load all models without a webcam.

## Language rules

- Write code comments, docstrings and terminal output (stdout, stderr, exceptions, `--help`, `.bat` echo lines) in ASD-STE100 Simplified Technical English. The terminal output is always English.
- The window text has English and Italian versions. Put it in the `TEXT` dictionary in `vanish_cam.py`. English is the default.
- Use only ASCII characters in window text. The rule keeps the text the same on all systems.

## Window

- `AppWindow` is the Tkinter window. It shows the video and a side panel with the webcam, the effect, the language and the buttons.
- `VanishEngine` keeps the background, the models and the effect state. The tests use it without a window.
- The window uses Pillow (`ImageTk`) to show the frames. The `PhotoImage` PPM path is 4 times slower.
- On Linux, `list_video_modes` reads the webcam modes with V4L2 ioctls. Other systems get `FALLBACK_VIDEO_MODES`.
- The virtual camera opens again when the video mode changes. While an app reads a v4l2loopback device, the driver keeps the old size without an error. `virtual_output_size` reads the real size, and the app then uses the old settings again.

## README

- The README has a "Fork" section. Add each new user-visible feature of the fork to it.

## Assets

- `assets/teleport.MP3` and `assets/peace.mp3` are third-party sounds. They are in `.gitignore`. Do not commit them. Without them, the Goku mode and the Peace mode have no sound.
- To make `assets/peace.mp3` again, download https://www.youtube.com/watch?v=MTO6ZKheD4A with `uvx yt-dlp -x --audio-format mp3`. Then cut it with `ffmpeg -ss 1.70 -to 3.95 -af "afade=t=out:st=1.95:d=0.3"`. The full file has a click at the start and a loud crash at the end.
- `MODE_SOUNDS` in `vanish_cam.py` gives the sound for each mode.

## Known problems

- When no hand is visible, a gesture check takes approximately 32 ms (1 video view and 3 image retries). The check occurs each 140 ms, and the video stutters. Possible fixes: 1 retry view for each check, or a worker thread.
- The app downloads the MediaPipe models to `models/` on the first start. `MODE_MODELS` in `vanish_cam.py` gives the models for each mode.

## Meeting audio (checkbox or `--meeting-audio`)

- `meeting_audio.py` needs PipeWire, `pactl` and `pw-link`.
- VanishMic is a null sink with `media.class=Audio/Source/Virtual`. A `pw-link` connects `vanish_mix` to it.
- Do not use `module-remap-source` or a loopback as the source. They do not get the `HARDWARE` flag, and KDE Plasma hides them when "Show virtual devices" is off.
- The app unloads only the module IDs that it loaded. On start, it removes the modules of an earlier run that crashed.
- The effect sound reads `PULSE_SINK` only when the audio opens. After the checkbox changes, `VanishEngine.reload_sound` opens the sound again.
