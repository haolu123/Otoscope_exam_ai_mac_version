# Otoscope Exam AI

Windows desktop exam application for otoscope videos with integrated AI analysis.

## Run

Double-click:

```text
otoscope_exam_ai.exe
```

The expected directory structure is:

```text
otoscope_exam_ai/
├─ otoscope_exam_ai.exe
├─ ai_worker/
│  ├─ otoscope_ai_worker.exe
│  └─ _internal/
├─ videos/
│  ├─ AOM/
│  ├─ Effusion/
│  ├─ Normal/
│  ├─ Perforation/
│  ├─ Retraction/
│  ├─ Tubes/
│  └─ Tympanosclerosis/
├─ result/
└─ ai_output/
```

The exam result CSV is saved to `result/` with three columns:

```text
Video ID, Correct Answer, Participant Answer
```

AI probabilities are saved separately to `result/ai_result_*.csv`.
Selected key frames and heatmaps are cached in `ai_output/`.

Diagnostic logs are saved automatically:

```text
result/application.log
result/native_crash.log
```

If the application closes unexpectedly, include these two files when reporting
the issue.

## AI Analysis

Video playback and AI analysis read frames through imageio and its bundled
FFmpeg binary. OpenCV video decoding is used only as an AI fallback if imageio
cannot open a video.

AI analysis runs in the separate `ai_worker/otoscope_ai_worker.exe` process.
If native AI libraries fail, the exam application remains open and reports the
AI error. Keep the complete `ai_worker/` folder beside `otoscope_exam_ai.exe`.

For each video, the application selects up to five key frames using:

```text
combined score = eardrum probability * image quality probability
```

The selected frames are then sent to the disease classifier. The interface shows:

- AI prediction
- class probabilities
- selected key frames
- Grad-CAM heatmaps
- eardrum and image quality probabilities for each key frame

## Build

The conda environment is named `otoscope_exam`.
It uses Python 3.11, PySide6 6.7.3, OpenCV, PyTorch, torchvision, PyTorch Lightning,
scikit-learn, pandas, joblib, PyYAML, and pytorch-grad-cam.

```powershell
.\build_exe.ps1
```

## Question Count

By default, the exam uses 100 videos with category-balanced sampling.

```python
QUESTION_LIMIT = 100
BALANCE_CATEGORIES = True
```

To use every video instead:

```python
QUESTION_LIMIT = None
BALANCE_CATEGORIES = False
```
