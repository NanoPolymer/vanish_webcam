# VanishCam: webcam disappearance effects

Make yourself vanish on your webcam with a hand gesture. The video is processed on your computer. It is neither recorded nor sent to a server. The video in the preview and the virtual camera shows only the video and the effects, without overlaid menus, instructions, or text. The controls sit in a side panel next to the video.

## Fork

This is a fork of [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam). It adds:

- **Control window:** one window with the video and a side panel, instead of the start menu. Webcam, video mode, effect, and language are dropdowns, and you can change all of them while the app runs.
- **Background capture button:** it turns orange and counts down, so you no longer need to watch the terminal.
- **Video mode dropdown:** the resolutions and frame rates your webcam supports, like in OBS. The virtual camera follows the selected mode.
- **Faster webcam input:** the app asks the webcam for MJPG, which many webcams deliver at full frame rate only in this format.
- **Goku Vanish:** two fingers on the forehead, then a teleport through vertical slits.
- **Peace Out:** a V sign freezes the frame, then you fade away like in the meme.
- **Meeting microphone (Linux):** a virtual microphone, VanishMic, that sends the effect sound to the other meeting participants. A checkbox switches it on and off.
- **Fixes:** the virtual camera is no longer mirrored, Naruto mode runs without an audio device, effect sounds start without their leading silence, and the V sign dissolve is twice as fast.

The upstream Windows EXE does not contain these changes. To use them, run the app from the sources or build the EXE yourself.

## Credits

The original app is by [Nicovic17](https://github.com/Nicovic17). If you like it, you can buy the author a coffee through [PayPal](https://paypal.me/nicovic17) or [Revolut](https://revolut.me/nicoauriemma). The upstream Windows build is available at [nicoaauriemma.vercel.app/shop](https://nicoaauriemma.vercel.app/shop).

## Getting started

### Linux

Use Python 3.12 (MediaPipe has no wheels for newer versions yet):

```sh
uv venv -p 3.12 .venv-local
uv pip install -p .venv-local -r requirements.txt
.venv-local/bin/python vanish_cam.py
```

### Windows

Put this folder in a parent folder and name it `local_app`, then double-click `AVVIA_LOCALE.bat`. The script creates the Python environment and installs the components on the first start.

### First steps

1. In the side panel, choose the **Webcam**, the **Video mode**, and the **Effect**.
2. Click **Capture background** and step out of the camera frame. The button turns orange and counts down 4 seconds. When it shows **New background**, step back in.
3. Do the gesture of the effect, or press `V`.

The app downloads the MediaPipe models to `models/` on the first start.

## Effects

**Normal Vanish:** Raise your index and middle fingers in a V sign. The dissolve starts as soon as the gesture is recognized and lasts about 1.8 seconds.

**Naruto Vanish:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. You vanish in a cloud of smoke, and a wooden log appears, falls, and remains visible until you press `R`. `assets/vanish.MP3` plays when the disappearance starts.

**Goku Vanish:** Hold your index and middle fingers straight and together against your forehead for 1 second. Vertical slits open in your silhouette, the background shows through them, and you teleport away in about 0.3 seconds. To reappear, lower your hand, then repeat the gesture for another second, or press `R`. If you break the pose, the countdown starts over.

**Peace Out:** Show the V sign. The image freezes, then you fade away in about 1.4 seconds.

The Goku and Peace Out sounds are third-party sounds and not part of the repository. To hear them, put your own MP3 files at `assets/teleport.MP3` and `assets/peace.mp3` (both are in `.gitignore`). Without them, these effects run without sound.

To start with a different effect selected, pass `--mode normal`, `--mode naruto`, `--mode goku`, or `--mode peace`.

## Controls

- `SPACE` or **Capture background**: capture a new background.
- `V` or **Vanish**: start the selected effect manually.
- `R` or **Reappear**: make the person reappear. In Goku mode, it plays the reverse teleport.
- `F` or **Full screen**: toggle full screen. Full screen hides the side panel. `ESC` leaves it.
- `Q` or closing the window: quit and release the cameras.

When you change the webcam or the video size, the old background no longer fits, so capture a new one. When you change the effect or only the frame rate, the background stays.

The side panel is in English by default. Choose **Italiano** under **Language** to switch to Italian. Terminal messages are always in English.

## Video mode

On Linux, the **Video mode** list shows the modes your webcam supports: 30 and 60 fps for each size, in MJPG where the webcam offers it. On other systems, it shows usual modes.

The virtual camera follows the video mode: it gets the same size and frame rate. While an app reads the virtual camera, the driver cannot change its size. Then the virtual camera keeps its old size and frame rate, the video is scaled to fit, and the terminal tells you. Close the apps that use the virtual camera, then select the video mode again.

To pick the video mode at start, pass the size and the highest frame rate:

```sh
python vanish_cam.py --width 1280 --height 720 --fps 60
```

## Virtual camera and OBS

If a virtual camera is available (OBS Virtual Camera or v4l2loopback), the app uses it automatically. The terminal shows the device name. Select that camera in the app that will receive the video. OBS's own virtual camera must be stopped while this app is producing its output. To require a virtual camera, pass `--virtual-camera` (or use `AVVIA_VIRTUAL.bat`).

To use the effect in an OBS scene instead, run the app with `--no-virtual-camera` (or `AVVIA_OBS.bat`), add a **Window Capture** source, select **VanishCam**, and then start the virtual camera from OBS. Press `F` for full screen to hide the side panel, or crop it away in OBS. Alternatively, use [Unity Capture](https://github.com/schellingb/UnityCapture) as a separate virtual video source with `--virtual-backend unitycapture`.

OBS has only one virtual camera. Do not use it as both an OBS input and output at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Effect sound in meetings

The virtual camera carries only video. To send the effect sound to the other participants on Linux, tick **Meeting microphone (VanishMic)** in the side panel. You can switch it on and off while the app runs. To start with it on, pass `--meeting-audio`.

The app creates a virtual microphone, **VanishMic**, that mixes your default microphone and the effect sound. You still hear the sound through your speakers or headset. Select **VanishMic** as the microphone in the meeting app, and set its noise suppression to low (in Zoom, use "Original sound"), or it may filter out the effect. The app removes the virtual devices when you untick the box or close the app. This needs PipeWire with `pactl` and `pw-link`. On Windows, use [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) or Voicemeeter instead.

## Troubleshooting

To see the gesture recognition status, pass `--gesture-debug`. The information appears only in the terminal, not in the video.

## Build the Windows EXE

Run `BUILD_WINDOWS.bat`. It installs the build components, downloads the models, and makes `dist\VanishCam.exe` with PyInstaller. The optional sounds go into the EXE only if they are in `assets/`.

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using a supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py`, so it follows the person's position and scales with the video. No anime frames or assets are included in the project.

Goku mode uses the [MediaPipe BlazeFace short-range model](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md) to estimate the forehead position and `hand_landmarker.task` to track the fingers. The teleport's visual trace is drawn in code.
