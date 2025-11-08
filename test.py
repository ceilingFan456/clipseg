import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm

# your model import (unchanged)
from models.clipseg import CLIPDensePredT

def main():
    # set paths here
    IN_ROOT = Path("/robot_dataset/final_test_set/run5_test105_clipseg")  # input folder
    OUT_BASE = Path("/path/to/output_folder")                            # output folder

    PROMPT_TO_CODE = {"robot": "000", "gripper": "001", "robot arm": "002"}

    # load model (unchanged)
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(
        torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')),
        strict=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).half()  # assuming fp16

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])

    def _ensure_dir(p: Path):
        os.makedirs(p, exist_ok=True)

    all_imgs = sorted((IN_ROOT / "image").rglob("*.jpg"))
    if not all_imgs:
        print(f"[WARN] No .jpg files under {IN_ROOT/'image'}")
        return

    print(f"[INFO] Processing {len(all_imgs)} images …")
    for img_path in tqdm(all_imgs, desc="Images", unit="img"):
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

        pil_img = Image.fromarray(image_rgb)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        for user_text, code in PROMPT_TO_CODE.items():
            with torch.no_grad():
                preds = model(img_tensor.repeat(1,1,1,1), user_text)[0]
                mask_pred = torch.sigmoid(preds[0][0]).cpu().numpy() > 0.5

            # binary mask
            mask_dir = OUT_BASE / "mask" / case_name / code
            _ensure_dir(mask_dir)
            cv2.imwrite(str(mask_dir / frame_name), (mask_pred.astype(np.uint8) * 255))

            # save original image
            img_dir = OUT_BASE / "image" / case_name
            _ensure_dir(img_dir)
            cv2.imwrite(str(img_dir / frame_name), image_bgr)

            # save masked overlay
            masked_dir = OUT_BASE / "masked" / case_name / code
            _ensure_dir(masked_dir)
            overlay = image_bgr.copy()
            overlay[mask_pred] = (0, 0, 255)  # red overlay
            alpha = 0.5
            vis = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
            cv2.imwrite(str(masked_dir / frame_name), vis)

    print(f"[DONE] Results saved to: {OUT_BASE}")

if __name__ == "__main__":
    main()
