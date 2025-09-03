import os
import csv
import time
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image


# -----------------------------
# Generic utilities
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_utc():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Prompt IO
# -----------------------------
def read_prompts_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# -----------------------------
# CLIP preprocessing (differentiable)
# -----------------------------
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def clip_preprocess_tensor(img_bchw: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    """
    Differentiable resize + center-crop + CLIP normalization.
    img_bchw expected in [0,1]. Returns normalized tensor for CLIP.
    """
    b, c, h, w = img_bchw.shape
    # Resize so shorter side = out_size
    scale = out_size / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = F.interpolate(img_bchw, size=(nh, nw), mode="bilinear", align_corners=False)
    # Center crop
    top = max(0, (nh - out_size) // 2)
    left = max(0, (nw - out_size) // 2)
    img = img[:, :, top:top + out_size, left:left + out_size]
    # Normalize
    mean = CLIP_MEAN.to(img.device, img.dtype)
    std = CLIP_STD.to(img.device, img.dtype)
    img = (img - mean) / std
    return img


# -----------------------------
# Sampling to disk (train/val/final)
# -----------------------------
def save_samples(
    sd,
    out_root: Path,
    split: str,
    step: int,
    prompts: List[str],
    num_images_per_prompt: int,
    cfg,
    seed: int,
    width: int,
    height: int,
):
    if not prompts:
        return
    folder = out_root / split / f"step_{step:07d}"
    ensure_dir(folder)
    csv_path = folder / "index.csv"
    write_header = not csv_path.exists()

    device = sd.unet.device
    gen = torch.Generator(device=device).manual_seed(seed)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "split", "prompt_idx", "sample_idx", "filename", "prompt"])
        for p_idx, prompt in enumerate(prompts):
            for s_idx in range(num_images_per_prompt):
                pixels, _, _ = sd_forward_unroll_for_utils(sd, [prompt], cfg=cfg, generator=gen)
                img = (pixels[0].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ts = now_utc()
                fn = f"{ts}_p{p_idx:05d}_{s_idx:02d}.png"
                fp = folder / fn
                Image.fromarray(img).save(fp)
                writer.writerow([step, split, p_idx, s_idx, str(fp), prompt])


# We import lazily to avoid circular imports (train -> utils -> train).
def sd_forward_unroll_for_utils(sd, prompts, cfg, generator):
    from models import sd_forward_unroll
    return sd_forward_unroll(sd, prompts, cfg=cfg, generator=generator, enable_grad=False)
