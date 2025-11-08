import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from models.clipseg import CLIPDensePredT  # keep unchanged

def parse_args():
    parser = argparse.ArgumentParser(description="CLIPSeg batch inference")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Root folder containing image/ subfolder with JPGs")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output base folder where image/, mask/, masked/ will be saved")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- model loading (unchanged) ---
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.eval()
    model.load_state_dict(torch.load('/home/t-qimhuang/weights/clipseg/rd64-uni-refined.pth'), strict=False)

    # --- rest of setup ---
    OUT_BASE = Path(args.output_folder)
    IN_ROOT = Path(args.input_folder)

    PROMPT_TO_CODE = {"robot": "000", "gripper": "001", "robot arm": "002"}

    def _ensure_dir(d: Path):
        os.makedirs(d, exist_ok=True)

    all_imgs = sorted((IN_ROOT / "image").rglob("*.jpg"))
    if not all_imgs:
        print(f"[WARN] No .jpg files under {IN_ROOT / 'image'}")
        return

    print(f"[INFO] Processing {len(all_imgs)} images …")
    for img_path in tqdm(all_imgs[:3], desc="Images", unit="img"):
        try:
            parts = img_path.resolve().relative_to(IN_ROOT.resolve()).parts
        except Exception:
            print(f"[WARN] skip (not under IN_ROOT): {img_path}")
            continue
        if len(parts) < 2 or parts[0] != "image":
            print(f"[WARN] skip (unexpected path): {img_path}")
            continue

        case_name = parts[1]
        frame_name = img_path.name

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] cannot read: {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # convert to PIL and apply transforms like in notebook
        pil_img = Image.fromarray(image_rgb)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ])
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        for user_text, code in PROMPT_TO_CODE.items():
            with torch.no_grad():
                preds = model(img_tensor.repeat(1,1,1,1), user_text)[0]  # keep use of your model API
                mask_pred = torch.sigmoid(preds[0][0]).cpu().numpy() > 0.5

            # binary mask save
            mask_dir = OUT_BASE / "mask" / case_name / code
            _ensure_dir(mask_dir)
            cv2.imwrite(str(mask_dir / frame_name), (mask_pred.astype(np.uint8) * 255))

            # save original
            img_dir = OUT_BASE / "image" / case_name
            _ensure_dir(img_dir)
            cv2.imwrite(str(img_dir / frame_name), image_bgr)

            # save masked overlay
            masked_dir = OUT_BASE / "masked" / case_name / code
            _ensure_dir(masked_dir)
            overlay = image_bgr.copy()
            overlay[mask_pred] = (0, 0, 255)
            alpha = 0.5
            vis = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
            cv2.imwrite(str(masked_dir / frame_name), vis)

    print(f"[DONE] Results saved to: {OUT_BASE}")

if __name__ == "__main__":
    main()
