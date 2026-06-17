$ErrorActionPreference = "Stop"

$Python = "C:\Users\haolu\AppData\Local\anaconda3\envs\otoscope_exam\python.exe"

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name otoscope_exam_ai `
    --copy-metadata imageio `
    --copy-metadata imageio-ffmpeg `
    --collect-data imageio_ffmpeg `
    --collect-binaries imageio_ffmpeg `
    --hidden-import imageio.plugins.ffmpeg `
    --hidden-import imageio_ffmpeg `
    app.py

if ($LASTEXITCODE -ne 0) {
    throw "Main application PyInstaller build failed with exit code $LASTEXITCODE"
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name otoscope_ai_worker `
    --add-data "config.yaml;." `
    --add-data "best_resnet50_eardrum.pth;." `
    --add-data "logistic_regression_model.pkl;." `
    --add-data "checkpoints;checkpoints" `
    --add-data "detection;detection" `
    --add-data "net;net" `
    --add-data "misc;misc" `
    --copy-metadata pytorch-lightning `
    --copy-metadata lightning-utilities `
    --copy-metadata torchmetrics `
    --copy-metadata imageio `
    --copy-metadata imageio-ffmpeg `
    --collect-data imageio_ffmpeg `
    --collect-binaries imageio_ffmpeg `
    --hidden-import detection.model `
    --hidden-import detection.infer `
    --hidden-import net.resnet50 `
    --hidden-import misc.torchutils `
    --hidden-import imageio.plugins.ffmpeg `
    --hidden-import imageio_ffmpeg `
    --hidden-import pytorch_lightning.__version__ `
    --hidden-import pytorch_lightning.__about__ `
    ai_worker_process.py

if ($LASTEXITCODE -ne 0) {
    throw "AI worker PyInstaller build failed with exit code $LASTEXITCODE"
}

$CurrentDirectory = [System.Environment]::CurrentDirectory
[System.IO.Directory]::CreateDirectory([System.IO.Path]::Combine($CurrentDirectory, "result")) | Out-Null
[System.IO.Directory]::CreateDirectory([System.IO.Path]::Combine($CurrentDirectory, "ai_output")) | Out-Null

[Console]::WriteLine("Build complete: .\dist\otoscope_exam_ai.exe and .\dist\otoscope_ai_worker\")
