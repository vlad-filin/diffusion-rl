#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image

# Reuse your project code
from models import SD15, DraftConfig, sd_forward_unroll, PromptDataset, PickScore
from utils import read_prompts_file, ensure_dir


def parse_args():
    ap = argparse.ArgumentParser("SD1.5 DRaFT inference (local) reusing repo code")
    ap.add_argument("--sd15_dir", required=True,
                    help="Local SD1.5 root (must contain subfolders: tokenizer, text_encoder, vae, unet, scheduler)")
    ap.add_argument("--prompts_file", required=True, help="Text file with one prompt per line")
    ap.add_argument("--out_dir", default="infer_outputs", help="Where to save images & CSV")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--samples_per_prompt", type=int, default=1,
                    help="How many images to draw per prompt (re-seeds deterministically)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--lora_dir", default=None, help="Optional LoRA weights folder saved via unet.save_attn_procs")
    ap.add_argument("--with_pickscore", action="store_true",
                    help="Compute PickScore with current models.PickScore implementation")
    return ap.parse_args()


@torch.no_grad()
def save_batch(folder: Path, start_idx: int, pixels: torch.Tensor, prompts: list[str],
               per_prompt_counts: dict[int, int], pick_scores: list[float] | None = None):
    """
    Save images with deterministic names: p{prompt_idx:05d}_{sample_idx:02d}.png
    No timestamps.
    """
    ensure_dir(folder)
    csv_path = folder / "index.csv"
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            header = ["filename", "prompt_idx", "sample_idx", "prompt"]
            if pick_scores is not None:
                header.append("pickscore")
            w.writerow(header)

        for i, (img, prompt) in enumerate(zip(pixels, prompts)):
            p_idx = start_idx + i
            # sample index increments per prompt
            s_idx = per_prompt_counts.get(p_idx, 0)
            per_prompt_counts[p_idx] = s_idx + 1

            fn = f"p{p_idx:05d}_{s_idx:02d}.png"
            arr = (img.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(arr).save(folder / fn)

            row = [fn, p_idx, s_idx, prompt]
            if pick_scores is not None:
                row.append(float(pick_scores[i]))
            w.writerow(row)


def main():
    args = parse_args()

    # Device & dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "fp16" and torch.cuda.is_available():
        dtype = torch.float16
    elif args.precision == "bf16" and torch.cuda.is_available():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Build SD1.5 using your wrapper
    sd = SD15(args.sd15_dir, device=device, dtype=dtype)

    # Optionally load LoRA weights saved by your training script
    if args.lora_dir:
        print(f"[info] Loading LoRA from: {args.lora_dir}")
        sd.unet.load_attn_procs(args.lora_dir)

    # Draft config (reuse your dataclass)
    draft_cfg = DraftConfig(
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance,
        method="draft",  # pure inference; gradient path irrelevant
        K=1,
        n_lv=1,
        gradient_checkpointing=False,
    )

    # Prompts & DataLoader (reuse your dataset)
    prompts = read_prompts_file(args.prompts_file)
    if not prompts:
        print("[warn] No prompts found.")
        return
    ds = PromptDataset(prompts)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Optional PickScore using your current implementation
    pick = PickScore().to(device) if args.with_pickscore else None

    # Output structure: out_dir/inference/steps{N}_cfg{X}_seed{S}
    out_dir = Path(args.out_dir)
    tag = f"steps{args.ddim_steps}_cfg{args.guidance}_seed{args.seed}"
    folder = out_dir / "inference" / tag
    ensure_dir(folder)

    # Deterministic generator; for multiple samples per prompt we offset the seed
    per_prompt_counts: dict[int, int] = {}
    start_idx = 0

    # Each pass over loader = one sample per prompt; repeat samples_per_prompt times
    for rep in range(args.samples_per_prompt):
        gen = torch.Generator(device=device).manual_seed(args.seed + rep)
        for batch_prompts in loader:
            batch_prompts = list(batch_prompts)

            # autocast only when on CUDA and using lower precision
            use_amp = (device.type == "cuda" and dtype != torch.float32)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                pixels, _, _ = sd_forward_unroll(sd, batch_prompts, cfg=draft_cfg, generator=gen, enable_grad=False)

            scores = None
            if pick is not None:
                with torch.no_grad():
                    scores = pick.score(pixels, batch_prompts).tolist()

            save_batch(folder, start_idx, pixels, batch_prompts, per_prompt_counts, scores)

            start_idx += len(batch_prompts)

        # reset start_idx for next repetition so prompt indices stay consistent
        start_idx = 0

    print(f"[done] Images & index saved at: {folder}")


if __name__ == "__main__":
    main()
    main()