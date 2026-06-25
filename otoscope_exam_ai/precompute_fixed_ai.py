import json
import os
import sys
import traceback
from pathlib import Path


def configure_runtime():
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


def make_relative_result(result: dict, root: Path) -> dict:
    converted = dict(result)
    for key in ("video_path", "output_folder"):
        value = converted.get(key)
        if value:
            converted[key] = str(Path(value).resolve().relative_to(root)).replace("\\", "/")
    for key in ("frame_files", "heatmap_files"):
        converted[key] = [
            str(Path(value).resolve().relative_to(root)).replace("\\", "/") if value else ""
            for value in converted.get(key, [])
        ]
    return converted


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def main() -> int:
    configure_runtime()
    root = Path(__file__).resolve().parent
    manifest_path = root / "fixed_questions_100.json"
    output_root = root / "ai_precomputed"
    results_path = output_root / "precomputed_ai_results.json"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or len(manifest) != 100:
        raise ValueError("fixed_questions_100.json must contain exactly 100 questions.")

    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        results = {}

    from PySide6 import QtCore
    import cv2
    import torch

    _ = QtCore.qVersion()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from ai_analysis import AIAnalyzer

    analyzer = AIAnalyzer()
    total = len(manifest)
    for index, item in enumerate(manifest, start=1):
        relative_path = item["relative_path"]
        video_path = root / relative_path
        existing = results.get(relative_path)
        if existing and "error" not in existing:
            print(f"[{index}/{total}] SKIP {relative_path}", flush=True)
            continue

        print(f"[{index}/{total}] RUN  {relative_path}", flush=True)
        try:
            result = analyzer.analyze_video(video_path, output_root)
            results[relative_path] = make_relative_result(result, root)
            print(
                f"[{index}/{total}] OK   {relative_path} -> "
                f"{results[relative_path].get('predicted_label')}",
                flush=True,
            )
        except BaseException as exc:
            results[relative_path] = {
                "video_path": relative_path,
                "error": str(exc) or type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
            print(f"[{index}/{total}] FAIL {relative_path}: {exc}", flush=True)
        finally:
            write_json(results_path, results)

    ok_count = sum(1 for value in results.values() if "error" not in value)
    error_count = sum(1 for value in results.values() if "error" in value)
    print(f"Done. successful={ok_count} failed={error_count} output={results_path}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
