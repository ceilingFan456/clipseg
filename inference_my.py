# bridge_batch_infer.py
import os, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch
# import requests

# ! wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
# ! unzip -d weights -j weights.zip
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
model.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('/home/t-qimhuang/weights/clipseg/rd64-uni-refined.pth'), strict=False)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])

# ---------------------------
# Helpers (unchanged behavior)
# ---------------------------
def _ensure_dir(d: Path):
    os.makedirs(d, exist_ok=True)

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    # keep exactly as in your original
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def iter_dataset_images(IN_ROOT: Path):
    for image_path in sorted((IN_ROOT / "image").rglob("*.jpg")):
        try:
            parts = image_path.resolve().relative_to(IN_ROOT.resolve()).parts
        except Exception:
            print(f"[WARN] skip (not under IN_ROOT): {image_path}")
            continue
        if len(parts) < 2 or parts[0] != "image":
            print(f"[WARN] skip (unexpected path): {image_path}")
            continue
        case_name = parts[1]
        frame_name = image_path.name
        yield image_path, case_name, frame_name

def build_output_paths(OUT_BASE: Path, case_name: str, frame_name: str, code: str):
    img_dir    = OUT_BASE / "image" / case_name
    mask_dir   = OUT_BASE / "mask" / case_name / code
    masked_dir = OUT_BASE / "masked" / case_name / code
    mask_name  = Path(frame_name).with_suffix(".png").name
    return {
        "img_out":    img_dir / frame_name,
        "mask_out":   mask_dir / mask_name,
        "masked_out": masked_dir / frame_name,
        "dirs":       [img_dir, mask_dir, masked_dir],
    }

def main():
    # ---- configure roots (same semantics as your previous script) ----
    # IN_ROOT  = Path("/home/t-qimhuang/disk/robot_dataset/final_test_set/roboengine_test_video")
    # OUT_BASE = Path("/home/t-qimhuang/disk/robot_dataset/final_test_set/roboengine_test_video_clipseg")
    IN_ROOT  = Path("/home/t-qimhuang/disk/robot_dataset/final_test_set/run5_test105")
    OUT_BASE = Path("/home/t-qimhuang/disk/robot_dataset/final_test_set/run5_test105_clipseg")

    # keep your code mapping; we’ll pass user_text into your infer function
    PROMPT_TO_CODE = {"robot arm": "000", "gripper": "001", "robot": "002"}

    print("Starting batch inference (linked to quickstart_infer.infer_one)...")

    for image_path, case_name, frame_name in tqdm(iter_dataset_images(IN_ROOT), desc="Processing images"):
        # Load image
        input_image = Image.open(image_path)
        img = transform(input_image).unsqueeze(0)
        H, W = input_image.size[1], input_image.size[0]

        # Save ORIGINAL image once (same as before)
        img_dir = OUT_BASE / "image" / case_name
        _ensure_dir(img_dir)
        # img_path_out = img_dir / frame_name
        
        # Run your model per prompt and save masks the same way
        for user_text, code in PROMPT_TO_CODE.items():
            # ---------- HOOK into your Quickstart file ----------
            # Expect infer_one to return mask in HxW (bool / {0,1} / float probs / logits)
            with torch.no_grad():
                preds = model(img.repeat(1, 1, 1, 1), [user_text])[0]
                mask = torch.sigmoid(preds[0][0])

            # Normalize mask to boolean like your previous post-processing
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().squeeze()
                # if shape is (1,H,W) or (H,W)
                mask = mask[0] if mask.ndim == 3 and mask.shape[0] == 1 else mask
                if mask.dtype.is_floating_point:
                    # If looks like logits or probabilities → sigmoid then threshold at 0.5
                    # Heuristic: if max>1 or min<0, treat as logits
                    mx, mn = float(mask.max()), float(mask.min())
                    if mx > 1.0 or mn < 0.0:
                        mask = torch.sigmoid(mask)
                    mask = (mask >= 0.1).to(torch.uint8).numpy()
                else:
                    mask = mask.to(torch.uint8).numpy()
            else:
                mask = np.asarray(mask)
                if mask.dtype.kind == "f":
                    # same logits/prob handling for numpy
                    mx, mn = float(mask.max()), float(mask.min())
                    if mx > 1.0 or mn < 0.0:
                        mask = 1 / (1 + np.exp(-mask))
                    mask = (mask >= 0.1).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)

            # Ensure shape matches original for overlay; if needed, resize
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            combined = mask.astype(bool)

            # ---- save binary mask (PNG) ----
            outs = build_output_paths(OUT_BASE, case_name, frame_name, code)
            for d in outs["dirs"]:
                _ensure_dir(d)

            ok = cv2.imwrite(str(outs["mask_out"]), (combined.astype(np.uint8) * 255))
            if not ok:
                print(f"[ERROR] Failed to save mask: {outs['mask_out']}")

            # ---- save masked visualization (transparent red overlay) ----
            alpha = 0.5
            ori = cv2.imread(str(image_path))
            overlay = ori.copy()
            overlay[combined] = (255, 0, 0)  # RGB red
            vis_bgr = cv2.addWeighted(overlay, alpha, ori, 1 - alpha, 0)

            ok = cv2.imwrite(str(outs["masked_out"]), vis_bgr)
            if not ok:
                print(f"[ERROR] Failed to save masked image: {outs['masked_out']}")

            ok = cv2.imwrite(str(outs["img_out"]), ori)
            if not ok:
                print(f"[ERROR] Failed to save original image: {outs['img_out']}")

if __name__ == "__main__":
    main()
