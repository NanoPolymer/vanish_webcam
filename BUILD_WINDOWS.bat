@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Install Python 3 for Windows before you build the executable.
  goto :error
)

py -3 -c "import PyInstaller, cv2, mediapipe, numpy, pyvirtualcam, pygame, PIL" >nul 2>nul
if errorlevel 1 (
  echo The script installs the build dependencies.
  py -3 -m pip install -r requirements.txt "pyinstaller>=6,<7"
  if errorlevel 1 goto :error
)

if not exist "models\gesture_recognizer.task" goto :models
if not exist "models\selfie_segmenter_landscape.tflite" goto :models
if not exist "models\blaze_face_short_range.tflite" goto :models
goto :build

:models
echo The script downloads the necessary MediaPipe models.
py -3 -c "from vanish_cam import MODE_MODELS, download_models; [download_models(m) for m in MODE_MODELS]"
if errorlevel 1 goto :error

:build
py -3 -m PyInstaller --noconfirm --distpath dist --workpath build VanishCam.spec
if errorlevel 1 goto :error

"dist\VanishCam.exe" --self-test
if errorlevel 1 goto :error

echo.
echo Ready: dist\VanishCam.exe
exit /b 0

:error
echo.
echo The build failed. Read the messages above.
pause
exit /b 1
