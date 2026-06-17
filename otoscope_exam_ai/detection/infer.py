import os
import sys
import glob
import yaml
import torch
from typing import List, Dict
from torchvision import transforms
from PIL import Image

# Re-use your utility for dynamic loading
from detection.utils import get_obj_from_str

def resource_path(relative_path):
    """ Get absolute path to resource, handles PyInstaller's _internal folder """
    try:
        # In PyInstaller 6+, sys._MEIPASS points directly to the '_internal' folder
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
# ============================================================
# Utility: Frame Loading (Set 2 Template)
# ============================================================
def load_video_frames(video_dir: str, img_size: int) -> torch.Tensor:
    frame_paths = sorted(
        [
            p for p in glob.glob(os.path.join(video_dir, "*.jpg"))
            + glob.glob(os.path.join(video_dir, "*.png"))
            if not os.path.basename(p).startswith("cam_")
        ]
    )
    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in: {video_dir}")

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    imgs = [tfm(Image.open(p).convert("RGB")) for p in frame_paths]
    return torch.stack(imgs, dim=0), frame_paths

# ============================================================
# Run inference on a single video (Multiclass Math)
# ============================================================
def infer_single_video(model, video_dir: str, device: str, img_size: int):
    model.eval()
    model.to(device)

    # 1. Load frames: (T, C, H, W)
    imgs, _ = load_video_frames(video_dir, img_size)
    
    with torch.no_grad():
        # 2. Get frame-level logits
        flat_logits = model(imgs.to(device))
        # print(f"DEBUG LOGITS [Frame 0]: {flat_logits[0][:5]}")
        # 3. Softmax: Forces probabilities across classes to sum to 1.0 (Mutually exclusive)
        # 
        probs = torch.softmax(flat_logits, dim=1)
        # print(probs)

    # 4. Aggregation: Mean of probabilities over all frames (Set 1 logic)
    video_probs = probs.mean(dim=0)
    # print(video_probs)
    
    # Renormalize for numerical safety
    if video_probs.sum() > 0:
        video_probs /= video_probs.sum()
    
    # print(video_probs)
    # 5. Argmax: Select the single winner (Multiclass behavior)
    pred_idx = int(torch.argmax(video_probs).item())
    
    # 6. Map to label name
    label_name = model.label_list[pred_idx] if (hasattr(model, 'label_list') and model.label_list) else str(pred_idx)

    # 7. Standardized Return: Matches Set 2 structure
    return {
        "video_dir": video_dir,
        "predicted_labels": [label_name], # Returned as a list of strings
        "prediction_probs": video_probs.cpu().numpy().tolist(),
        "labels": model.label_list
    }

# ============================================================
# Run inference on many videos (The Missing Template Function)
# ============================================================
def infer_many_videos(model, video_list: List[str], device: str, img_size: int):
    """
    Iterates through a list of directories, each representing a video.
    """
    results = []
    for vd in video_list:
        res = infer_single_video(model, vd, device, img_size)
        results.append(res)
    return results

# ============================================================
# Deployment Helpers (Set 2 Template)
# ============================================================
def load_inference_model(cfg):
    """Initializes model once for the application lifetime."""
    ckpt_path = resource_path(cfg["model"]["checkpoint_path"])
    ModelClass = get_obj_from_str(cfg["model"]["target"])
    
    model = ModelClass.load_from_checkpoint(
        ckpt_path,
        strict=False,
        **cfg["model"].get("params", {})
    )
    
    device = cfg["inference"].get("device", "cuda")
    model.to(device).eval()
    return model

def run_inference_on_loaded_model(model, cfg, video_input):
    """API-friendly execution for Autoscope or Flask."""
    video_list = [video_input] if isinstance(video_input, str) else video_input
    img_size = cfg["inference"].get("img_size", 224)
    device = cfg["inference"].get("device", "cuda")

    results = infer_many_videos(model, video_list, device, img_size)
    
    return [
        {
            "predicted_labels": r['predicted_labels'], 
            "prediction_probs": r['prediction_probs'],
            "labels": r['labels']
        } for r in results
    ]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    loaded_model = load_inference_model(config)
    # Example: Run on a test directory if provided in config
    if "video_dir" in config.get("data", {}):
        out = run_inference_on_loaded_model(loaded_model, config, config["data"]["video_dir"])
        print(out)
