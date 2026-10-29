# VanishCam — webcam disappearance effects

Video is processed on your computer. It is neither recorded nor sent to a server. The video in the preview window and the virtual camera shows only the video and effects, without overlaid menus, instructions, or text. The controls sit in a side panel next to the video.

For step-by-step instructions in English and Italian, see the [user guide](USER_GUIDE_EN_IT.md).

## ☕ Buy me a coffee

If you like this project and want to support it, you can buy me a coffee through [PayPal](https://paypal.me/nicovic17) or [Revolut](https://revolut.me/nicoauriemma).

## Download for Windows

[Download VanishCam.exe]( https://nicoaauriemma.vercel.app/shop ) and double-click it to run the app. You do not need to install Python.

The published EXE may predate the Goku mode. To use them now, run the local scripts or rebuild the EXE from these sources.

## Getting started

1. Double-click `AVVIA_LOCALE.bat`.
2. In the side panel, choose the **Webcam** and the **Effect** (**Normal Vanish**, **Naruto Vanish**, **Goku Vanish**, or **Peace Out**). You can change both at any time while the app runs.
3. Click **Capture background** and step out of the camera frame. The button turns orange and counts down 4 seconds. When it shows **New background**, step back in.

**Normal Vanish:** Raise your index and middle fingers in a V sign. The dissolve starts as soon as the gesture is recognized and lasts about 1.8 seconds.

**Goku:** Hold your index and middle fingers straight and together against your forehead for 1 second. Vertical slits open in your silhouette, the background shows through them, and you teleport away in about 0.3 seconds. To reappear, lower your hand, then repeat the gesture for another 1 second. The return uses the reverse effect. If you break the pose, the countdown starts over. `V` and `R` also work: `R` plays the return effect. To troubleshoot the gesture, pass `--gesture-debug`. No text is overlaid on the video.

The teleport sound is not part of the repository. To hear it, put your own MP3 at `assets/teleport.MP3` (the file is in `.gitignore`). Without it, Goku mode runs without sound.

**Peace Out:** Show the V sign. The image freezes, then you fade away in about 1.4 seconds, like the "peace out" meme. The sound is not part of the repository either. To hear it, put your own MP3 at `assets/peace.mp3` (the file is in `.gitignore`).

**Naruto Vanish:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. As soon as the gesture is recognized, you vanish in a cloud of smoke and a wooden log appears, falls, and remains visible until you press `R`. No microphone is needed. Naruto mode uses `hand_landmarker.task` to track both hands. `assets/vanish.MP3` plays through your computer speakers when the disappearance starts, including when triggered with `V`. You can trigger it again after pressing `R`.

To start with a different effect selected:

```powershell
local_app\AVVIA_LOCALE.bat --mode naruto
local_app\AVVIA_LOCALE.bat --mode normal
local_app\AVVIA_LOCALE.bat --mode goku
local_app\AVVIA_LOCALE.bat --mode peace
```

## Virtual camera and OBS

If [OBS Studio](https://obsproject.com/) is installed, the app automatically tries to open **OBS Virtual Camera**. The terminal shows the device name. Use `AVVIA_VIRTUAL.bat` to require a virtual camera. Select **OBS Virtual Camera** in the app that will receive the video. OBS's virtual camera must be stopped while this app is producing its output.

To use the effect in an OBS scene, run `AVVIA_OBS.bat`, add a **Window Capture** source, select **VanishCam**, and then start the virtual camera from OBS. Press `F` for full screen to hide the side panel, or crop it away in OBS. Alternatively, use [Unity Capture](https://github.com/schellingb/UnityCapture) as a separate virtual video source:

```powershell
local_app\AVVIA_VIRTUAL.bat --virtual-backend unitycapture
```

OBS has only one virtual camera. Do not use it as both an OBS input and output at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Effect sound in meetings

The virtual camera carries only video. To send the effect sound to the other participants on Linux, tick **Meeting microphone (VanishMic)** in the side panel. You can switch it on and off while the app runs. To start with it on, pass `--meeting-audio`:

```sh
python vanish_cam.py --meeting-audio
```

The app creates a virtual microphone, **VanishMic**, that mixes your default microphone and the effect sound. You still hear the sound through your speakers or headset. Select **VanishMic** as the microphone in the meeting app, and set its noise suppression to low (in Zoom, use "Original sound"), or it may filter out the effect. The app removes the virtual devices when you untick the box or close the app. This needs PipeWire with `pactl` and `pw-link`. On Windows, use [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) or Voicemeeter instead.

## Controls

- `SPACE` or **Capture background**: capture a new background.
- `R` or **Reappear**: make the person reappear.
- `V` or **Vanish**: manually start the selected effect.
- `F` or **Full screen**: toggle fullscreen. Full screen hides the side panel. `ESC` leaves it.
- `Q` or closing the window: quit and release the cameras.

In Goku mode, `R` plays the reverse teleport.

When you change the webcam, the old background no longer fits, so capture a new one. When you change the effect, the background stays.

The side panel is in English by default. Choose **Italiano** under **Language** to switch to Italian. Terminal messages are always in English.

You can change the resolution and maximum virtual-output FPS:

```powershell
local_app\AVVIA_LOCALE.bat --width 1280 --height 720 --fps 30
```

To troubleshoot gesture recognition without adding text to the video, pass `--gesture-debug`. Diagnostic information appears only in the terminal.

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using the supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py` so it follows the person's position and scales with the video. We also considered a [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet), but the procedural animation adapts better to dynamic positioning without an additional asset. No anime frames or assets are included in the project.

Goku mode uses the [MediaPipe BlazeFace short-range model](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md) to estimate the forehead position and `hand_landmarker.task` to track the fingers. The teleport's visual trace is drawn in code, without anime frames.
