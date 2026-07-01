#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="Otoscope Exam AI"
MAIN_EXE="otoscope_exam_ai"
WORKER_NAME="otoscope_ai_worker"
RELEASE_ROOT="dist_macos/otoscope_exam_ai_mac"
ZIP_PATH="dist_macos/otoscope_exam_ai_mac.zip"

rm -rf build dist dist_macos

python -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --copy-metadata imageio \
  --copy-metadata imageio-ffmpeg \
  --collect-data imageio_ffmpeg \
  --collect-binaries imageio_ffmpeg \
  --hidden-import imageio.plugins.ffmpeg \
  --hidden-import imageio_ffmpeg \
  app.py

python -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name "$WORKER_NAME" \
  --add-data "config.yaml:." \
  --add-data "fixed_questions_100.json:." \
  --add-data "best_resnet50_eardrum.pth:." \
  --add-data "logistic_regression_model.pkl:." \
  --add-data "checkpoints:checkpoints" \
  --add-data "detection:detection" \
  --add-data "net:net" \
  --add-data "misc:misc" \
  --copy-metadata pytorch-lightning \
  --copy-metadata lightning-utilities \
  --copy-metadata torchmetrics \
  --copy-metadata imageio \
  --copy-metadata imageio-ffmpeg \
  --collect-data imageio_ffmpeg \
  --collect-binaries imageio_ffmpeg \
  --hidden-import detection.model \
  --hidden-import detection.infer \
  --hidden-import net.resnet50 \
  --hidden-import misc.torchutils \
  --hidden-import imageio.plugins.ffmpeg \
  --hidden-import imageio_ffmpeg \
  --hidden-import pytorch_lightning.__version__ \
  --hidden-import pytorch_lightning.__about__ \
  ai_worker_process.py

mkdir -p "$RELEASE_ROOT/result" "$RELEASE_ROOT/ai_output"
cp -R "dist/$APP_NAME.app" "$RELEASE_ROOT/$APP_NAME.app"
cp -R "dist/$WORKER_NAME" "$RELEASE_ROOT/ai_worker"
mkdir -p "$RELEASE_ROOT/ffmpeg"
FFMPEG_EXE="$(
python - <<'PY'
import shutil
try:
    import imageio_ffmpeg
    print(imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    print(shutil.which("ffmpeg") or "")
PY
)"
if [ -z "$FFMPEG_EXE" ] || [ ! -f "$FFMPEG_EXE" ]; then
  echo "Could not locate ffmpeg executable for bundling" >&2
  exit 1
fi
cp "$FFMPEG_EXE" "$RELEASE_ROOT/ffmpeg/ffmpeg"
if [ -d videos ]; then
  cp -R videos "$RELEASE_ROOT/videos"
else
  mkdir -p "$RELEASE_ROOT/videos"
  cat > "$RELEASE_ROOT/videos/PUT_VIDEOS_HERE.txt" <<'EOF'
Place the seven category folders here before running the app:

AOM
Effusion
Normal
Perforation
Retraction
Tubes
Tympanosclerosis
EOF
fi
cp README.md "$RELEASE_ROOT/README.md"
cp READ_ME_FIRST_MAC.txt "$RELEASE_ROOT/READ_ME_FIRST_MAC.txt"
cp fixed_questions_100.json "$RELEASE_ROOT/fixed_questions_100.json"
if [ -d ai_precomputed ]; then
  cp -R ai_precomputed "$RELEASE_ROOT/ai_precomputed"
fi

chmod +x "$RELEASE_ROOT/$APP_NAME.app/Contents/MacOS/$APP_NAME" || true
chmod +x "$RELEASE_ROOT/ai_worker/$WORKER_NAME" || true
chmod +x "$RELEASE_ROOT/ffmpeg/ffmpeg" || true

mkdir -p dist_macos
(
  cd dist_macos
  ditto -c -k --sequesterRsrc --keepParent otoscope_exam_ai_mac otoscope_exam_ai_mac.zip
)

echo "Built: $ZIP_PATH"
