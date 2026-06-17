import json
import os
import sys
import traceback
from pathlib import Path


class NullWriter:
    def write(self, message):
        return 0

    def flush(self):
        return None

    def isatty(self):
        return False


def configure_native_runtime():
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


def write_json_atomic(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=True)
    temporary_path.replace(path)


def main() -> int:
    if len(sys.argv) != 4:
        return 2

    configure_native_runtime()
    if sys.stdout is None:
        sys.stdout = NullWriter()
    if sys.stderr is None:
        sys.stderr = NullWriter()
    video_path = Path(sys.argv[1]).resolve()
    output_root = Path(sys.argv[2]).resolve()
    result_path = Path(sys.argv[3]).resolve()
    matplotlib_cache = output_root / "_matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    try:
        from PySide6 import QtCore
        import torch
        import cv2

        _ = QtCore.qVersion()
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        from ai_analysis import AIAnalyzer

        result = AIAnalyzer().analyze_video(video_path, output_root)
        write_json_atomic(result_path, {"ok": True, "result": result})
        return 0
    except BaseException as exc:
        write_json_atomic(
            result_path,
            {
                "ok": False,
                "error": str(exc) or type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
