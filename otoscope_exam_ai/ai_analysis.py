import gc
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import cv2
import imageio.v2 as imageio
import joblib
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from torchvision import transforms

try:
    from skimage import exposure

    HAS_SKI_EXPOSURE = True
except Exception:
    HAS_SKI_EXPOSURE = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    HAS_GRADCAM = True
except Exception:
    HAS_GRADCAM = False

from detection.infer import load_inference_model, run_inference_on_loaded_model
from app_diagnostics import get_logger
from net.resnet50_cam import Net as ResNet50CAM


CATEGORIES = [
    "AOM",
    "Effusion",
    "Normal",
    "Perforation",
    "Retraction",
    "Tube",
    "Tympanosclerosis",
]

EARD_NUM_THRESH = 0.7
QUALITY_THRESH = 0.5
INFER_EVERY_N_FRAMES = 2
MASK_UPDATE_EVERY_N_FRAMES = 5
MAX_CONSECUTIVE_READ_FAILURES = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


@dataclass
class KeyFrame:
    score: float
    eardrum_score: float
    quality_score: float
    frame_bgr: np.ndarray


def resource_path(relative_path: str) -> Path:
    try:
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).resolve().parent
    return base_path / relative_path


def output_folder_for_video(video_path: Path, output_root: Path) -> Path:
    digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in video_path.stem)
    folder = output_root / f"{stem}_{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _clahe_equalize(gray_uint8: np.ndarray) -> np.ndarray:
    if HAS_SKI_EXPOSURE:
        eq = exposure.equalize_adapthist(gray_uint8 / 255.0) * 255.0
        return eq.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_uint8)


def generate_mask_from_frame(img_bgr: np.ndarray, hough_scale_target: float = 224.0):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    scale = hough_scale_target / max(h, w)
    resized_h = max(1, int(h * scale))
    resized_w = max(1, int(w * scale))
    gray_resized = cv2.resize(gray, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    gray_eq = _clahe_equalize(gray_resized)
    blurred = cv2.medianBlur(gray_eq, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=20,
        minRadius=20,
        maxRadius=int(max(resized_h, resized_w) * 0.5),
    )

    best_score = -1e9
    best_circle = None
    y_grid, x_grid = np.ogrid[:resized_h, :resized_w]
    if circles is not None:
        for circle in np.uint16(np.around(circles[0])):
            cx, cy, radius = (int(value) for value in circle)
            if abs(cx - resized_w // 2) > 40 or abs(cy - resized_h // 2) > 20:
                continue
            mask_in = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius**2
            mask_out = (
                ((x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= (radius + 12) ** 2)
                & (~mask_in)
            )
            if mask_in.sum() < 100:
                continue
            mean_in = gray_resized[mask_in].mean()
            mean_out = gray_resized[mask_out].mean() if mask_out.sum() > 0 else 0.0
            score = mean_in - mean_out
            if score > best_score:
                best_score = score
                best_circle = (cx, cy, radius)

    if best_circle is not None:
        cx, cy, radius = best_circle
    else:
        cx, cy = resized_w // 2, resized_h // 2
        radius = int(min(resized_h, resized_w) * 0.45)

    mask_circle = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius**2
    mask_full = cv2.resize(
        mask_circle.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    gray_eq_fullres = _clahe_equalize(gray)
    _, bright_mask = cv2.threshold(gray_eq_fullres, 200, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(gray_eq_fullres, 30, 255, cv2.THRESH_BINARY_INV)
    remove_mask = (bright_mask > 0) | (dark_mask > 0)
    final_mask = mask_full & (~remove_mask)

    kernel = np.ones((19, 19), np.uint8)
    final_mask = cv2.erode(final_mask.astype(np.uint8), kernel, iterations=1)
    return final_mask, gray


def compute_image_quality_with_mask(
    img_bgr: np.ndarray, mask_uint8: np.ndarray, gray_uint8: np.ndarray
) -> dict[str, float]:
    mask = mask_uint8.astype(bool)
    if mask.sum() == 0:
        return {
            "brightness": -1.0,
            "local_std": -1.0,
            "sharpness": -1.0,
            "colorfulness": -1.0,
        }

    gray_masked = gray_uint8[mask]
    brightness = float(np.mean(gray_masked))

    lap = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    sharpness = float(np.var(lap[mask]))

    gray_f = gray_uint8.astype(np.float32)
    g1 = cv2.GaussianBlur(gray_f, (3, 3), 0)
    g2 = cv2.GaussianBlur(gray_f**2, (3, 3), 0)
    local_std = np.sqrt(np.maximum(g2 - g1**2, 0))
    local_std_mean = float(np.mean(local_std[mask]))

    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colorfulness = float(
        np.sqrt(np.var(rg[mask]) + np.var(yb[mask]))
        + 0.3 * np.sqrt(np.mean(rg[mask]) ** 2 + np.mean(yb[mask]) ** 2)
    )

    return {
        "brightness": brightness,
        "sharpness": sharpness,
        "local_std": local_std_mean,
        "colorfulness": colorfulness,
    }


class AIAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._models_loaded = False
        self.resnet50 = None
        self.logreg_model = None
        self.disease_model = None
        self.cfg = None

    def load_models(self):
        if self._models_loaded:
            return

        logger = get_logger()
        logger.info("Loading AI models on device=%s", self.device)
        self.ensure_lightning_version()

        self.resnet50 = ResNet50CAM(out_dim=2)
        state = torch.load(resource_path("best_resnet50_eardrum.pth"), map_location=self.device)
        self.resnet50.load_state_dict(state)
        self.resnet50.to(self.device).eval()

        self.logreg_model = joblib.load(resource_path("logistic_regression_model.pkl"))

        with resource_path("config.yaml").open("r", encoding="utf-8") as file:
            self.cfg = yaml.safe_load(file)
        self.cfg.setdefault("inference", {})["device"] = str(self.device)
        self.disease_model = load_inference_model(self.cfg)
        self._models_loaded = True
        logger.info("AI models loaded")

    @staticmethod
    def ensure_lightning_version():
        import pytorch_lightning as pl

        if hasattr(pl, "__version__"):
            return
        try:
            from importlib.metadata import version

            pl.__version__ = version("pytorch-lightning")
        except Exception:
            pl.__version__ = "2.6.5"

    def analyze_video(self, video_path: Path, output_root: Path) -> dict:
        started = time.time()
        logger = get_logger()
        logger.info("AI analysis started for %s", video_path)
        keyframes = None
        try:
            self.load_models()
            output_dir = output_folder_for_video(video_path, output_root)

            keyframes = self.select_keyframes(video_path)
            if not keyframes:
                raise RuntimeError("No qualifying key frames were found for this video.")
            logger.info("Selected %s key frames for %s", len(keyframes), video_path)

            frame_files = []
            eardrum_probs = []
            quality_probs = []

            for index, keyframe in enumerate(keyframes, start=1):
                frame_name = f"frame_{index}_score_{keyframe.score:.3f}.png"
                frame_path = output_dir / frame_name
                cv2.imwrite(str(frame_path), keyframe.frame_bgr)
                frame_files.append(str(frame_path))
                eardrum_probs.append(keyframe.eardrum_score)
                quality_probs.append(keyframe.quality_score)

            inference_results = run_inference_on_loaded_model(
                self.disease_model, self.cfg, str(output_dir)
            )
            result = inference_results[0]
            labels = [self.display_label(label) for label in result["labels"]]
            probabilities = [float(value) for value in result["prediction_probs"]]
            predicted_index = int(np.argmax(probabilities))
            predicted_label = labels[predicted_index]
            logger.info("Disease prediction for %s: %s", video_path, predicted_label)

            heatmap_files = self.generate_disease_heatmaps(
                keyframes, output_dir, predicted_label, predicted_index
            )
            logger.info("Generated %s heatmaps for %s", len(heatmap_files), video_path)

            return {
                "video_path": str(video_path),
                "output_folder": str(output_dir),
                "frame_files": frame_files,
                "heatmap_files": heatmap_files,
                "eardrum_probabilities": eardrum_probs,
                "quality_probabilities": quality_probs,
                "labels": labels,
                "probabilities": probabilities,
                "predicted_label": predicted_label,
                "elapsed_seconds": round(time.time() - started, 2),
            }
        finally:
            keyframes = None
            self.cleanup_runtime_memory()
            logger.info("AI analysis memory cleanup complete for %s", video_path)

    def select_keyframes(self, video_path: Path) -> list[KeyFrame]:
        try:
            return self.select_keyframes_with_imageio(video_path)
        except Exception:
            get_logger().exception("imageio reader failed for %s; falling back to OpenCV", video_path)
            return self.select_keyframes_with_opencv(video_path)

    def select_keyframes_with_imageio(self, video_path: Path) -> list[KeyFrame]:
        frame_count = 0
        last_mask = None
        last_gray = None
        selected: list[KeyFrame] = []
        fallback_by_eardrum: list[KeyFrame] = []

        reader = imageio.get_reader(str(video_path), format="ffmpeg")
        try:
            for frame_rgb in reader:
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                frame_count += 1
                if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

                if frame_count % INFER_EVERY_N_FRAMES != 0:
                    continue

                if last_mask is None or frame_count % MASK_UPDATE_EVERY_N_FRAMES == 0:
                    try:
                        last_mask, last_gray = generate_mask_from_frame(frame)
                    except Exception:
                        last_mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                eardrum_score = self.predict_eardrum_probability(frame)
                quality_score = self.predict_quality_probability(frame, last_mask, last_gray)
                if eardrum_score is None:
                    continue
                if quality_score is None:
                    quality_score = 0.0

                fallback_by_eardrum.append(
                    KeyFrame(
                        score=eardrum_score,
                        eardrum_score=eardrum_score,
                        quality_score=quality_score,
                        frame_bgr=frame.copy(),
                    )
                )
                fallback_by_eardrum.sort(key=lambda item: item.eardrum_score, reverse=True)
                fallback_by_eardrum = fallback_by_eardrum[:5]

                if eardrum_score <= EARD_NUM_THRESH or quality_score <= QUALITY_THRESH:
                    continue

                combined_score = eardrum_score * quality_score
                selected.append(
                    KeyFrame(
                        score=combined_score,
                        eardrum_score=eardrum_score,
                        quality_score=quality_score,
                        frame_bgr=frame.copy(),
                    )
                )
                selected.sort(key=lambda item: item.score, reverse=True)
                selected = selected[:5]
        finally:
            reader.close()

        return selected if selected else fallback_by_eardrum

    def select_keyframes_with_opencv(self, video_path: Path) -> list[KeyFrame]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path.name}")

        frame_count = 0
        last_mask = None
        last_gray = None
        selected: list[KeyFrame] = []
        fallback_by_eardrum: list[KeyFrame] = []
        consecutive_failures = 0

        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                        break
                    continue
                consecutive_failures = 0

                frame_count += 1
                if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

                if frame_count % INFER_EVERY_N_FRAMES != 0:
                    continue

                if last_mask is None or frame_count % MASK_UPDATE_EVERY_N_FRAMES == 0:
                    try:
                        last_mask, last_gray = generate_mask_from_frame(frame)
                    except Exception:
                        last_mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        last_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                eardrum_score = self.predict_eardrum_probability(frame)
                quality_score = self.predict_quality_probability(frame, last_mask, last_gray)
                if eardrum_score is None:
                    continue
                if quality_score is None:
                    quality_score = 0.0

                fallback_by_eardrum.append(
                    KeyFrame(
                        score=eardrum_score,
                        eardrum_score=eardrum_score,
                        quality_score=quality_score,
                        frame_bgr=frame.copy(),
                    )
                )
                fallback_by_eardrum.sort(key=lambda item: item.eardrum_score, reverse=True)
                fallback_by_eardrum = fallback_by_eardrum[:5]

                if eardrum_score <= EARD_NUM_THRESH or quality_score <= QUALITY_THRESH:
                    continue

                combined_score = eardrum_score * quality_score
                selected.append(
                    KeyFrame(
                        score=combined_score,
                        eardrum_score=eardrum_score,
                        quality_score=quality_score,
                        frame_bgr=frame.copy(),
                    )
                )
                selected.sort(key=lambda item: item.score, reverse=True)
                selected = selected[:5]
        finally:
            capture.release()

        return selected if selected else fallback_by_eardrum

    def predict_eardrum_probability(self, frame_bgr: np.ndarray) -> float | None:
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            tensor = self.img_transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.resnet50(tensor)
                probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].item())
        except Exception:
            return None

    def predict_quality_probability(
        self, frame_bgr: np.ndarray, mask_uint8: np.ndarray, gray_uint8: np.ndarray
    ) -> float | None:
        try:
            features = compute_image_quality_with_mask(frame_bgr, mask_uint8, gray_uint8)
            if features["brightness"] <= 0:
                return None
            data = pd.DataFrame([features])[
                ["colorfulness", "local_std", "sharpness", "brightness"]
            ]
            return float(self.logreg_model.predict_proba(data)[:, 1][0])
        except Exception:
            return None

    def generate_disease_heatmaps(
        self,
        keyframes: list[KeyFrame],
        output_dir: Path,
        predicted_label: str,
        target_class_index: int,
    ) -> list[str]:
        if not HAS_GRADCAM:
            return [""] * len(keyframes)

        target_layers = [self.disease_model.model.layer4[-1]]
        cam_extractor = GradCAM(model=self.disease_model, target_layers=target_layers)
        heatmap_files = []
        try:
            for index, keyframe in enumerate(keyframes, start=1):
                heatmap_path = output_dir / (
                    f"cam_frame_{index}_{predicted_label}_score_{keyframe.score:.3f}.png"
                )
                saved_path = self.save_disease_heatmap(
                    cam_extractor,
                    keyframe.frame_bgr,
                    heatmap_path,
                    target_class_index,
                )
                heatmap_files.append(str(saved_path) if saved_path else "")
        finally:
            self.release_cam_extractor(cam_extractor)
            del cam_extractor
        return heatmap_files

    @staticmethod
    def release_cam_extractor(cam_extractor):
        release = getattr(cam_extractor, "release", None)
        if callable(release):
            release()
            return
        activations_and_grads = getattr(cam_extractor, "activations_and_grads", None)
        release = getattr(activations_and_grads, "release", None)
        if callable(release):
            release()

    def save_disease_heatmap(
        self,
        cam_extractor,
        frame_bgr: np.ndarray,
        heatmap_path: Path,
        target_class_index: int,
    ) -> Path | None:
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            tensor = self.img_transform(pil_img).unsqueeze(0).to(self.device)
            targets = [ClassifierOutputTarget(target_class_index)]
            grayscale_cam = cam_extractor(input_tensor=tensor, targets=targets)[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (frame_bgr.shape[1], frame_bgr.shape[0]))
            rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cv2.imwrite(str(heatmap_path), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            return heatmap_path
        except Exception:
            get_logger().exception("Grad-CAM generation failed for %s", heatmap_path)
            return None
        finally:
            for name in ("tensor", "grayscale_cam", "rgb_img", "cam_image"):
                if name in locals():
                    del locals()[name]

    def cleanup_runtime_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def display_label(label: str) -> str:
        normalized = label.strip().lower()
        mapping = {
            "aom": "AOM",
            "effusion": "Effusion",
            "normal": "Normal",
            "perforation": "Perforation",
            "retraction": "Retraction",
            "tube": "Tubes",
            "tubes": "Tubes",
            "tympanosclerosis": "Tympanosclerosis",
        }
        return mapping.get(normalized, label.strip().title())


ANALYZER = AIAnalyzer()
