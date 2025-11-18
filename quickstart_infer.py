# quickstart_infer.py
# CLIPSeg inference wrapper (for your local model & weights)
# Returns raw logits (H, W) for the bridge to post-process

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.clipseg import CLIPDensePredT

# ----------------------
# Lazy load global model
# ----------------------
_model = None
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

def _lazy_init():
    global _model
    if _model is not None:
        return
    # Instantiate CLIPSeg model
    _model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    _model.eval()
    # Load only decoder weights (non-strict)
    _model.load_state_dict(
        torch.load('/home/t-qimhuang/weights/clipseg/rd64-uni-refined.pth', map_location='cpu'),
        strict=False
    )
    _model = _model.cuda() if torch.cuda.is_available() else _model

@torch.inference_mode()
def infer_one(image_rgb: np.ndarray, user_text: str):
    """
    Args:
        image_rgb: np.uint8 (H, W, 3), RGB order.
        user_text: e.g. "robot", "gripper", "robot arm"
    Returns:
        mask_logits: float32 numpy array (H, W) of raw logits
    """
    _lazy_init()
    device = next(_model.parameters()).device

    # Convert numpy to PIL
    pil_image = Image.fromarray(image_rgb)
    img_tensor = _transform(pil_image).unsqueeze(0).to(device)

    # Forward
    preds = _model(img_tensor, [user_text])  # returns list of tensors, one per text
    logits = preds[0][0]                     # shape (1, H, W) → take first

    # Move to CPU numpy float32
    logits_np = logits.detach().cpu().numpy().astype(np.float32)

    return logits_np
