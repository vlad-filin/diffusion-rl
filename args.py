from typing import List, Tuple
import argparse
from utils import read_prompts_file


def load_train_val_prompts(args) -> Tuple[List[str], List[str]]:
    if args.train_prompts_file or args.val_prompts_file:
        train_prompts = read_prompts_file(args.train_prompts_file)
        val_prompts = read_prompts_file(args.val_prompts_file)
    elif args.prompts_file:
        all_prompts = read_prompts_file(args.prompts_file)
        n_val = min(args.num_val_prompts, len(all_prompts))
        val_prompts = all_prompts[:n_val]
        train_prompts = all_prompts[n_val:]
    else:
        train_prompts, val_prompts = [], []
    return train_prompts, val_prompts


def parse_args():
    p = argparse.ArgumentParser(
        description="DRaFT / DRaFT-K / DRaFT-LV finetuning for SD1.5 with PickScore"
    )

    p.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--output_dir", type=str, default="outputs")

    p.add_argument("--ddim_steps", type=int, default=50, help="Sampling steps (50 per paper)")
    p.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance (7.5 per paper)")

    p.add_argument("--method", type=str, choices=["draft", "draft-k", "draft-lv"], default="draft-k")
    p.add_argument("--K", type=int, default=1, help="Backprop through last-K steps (draft-k)")
    p.add_argument("--n_lv", type=int, default=2, help="DRaFT-LV inner loops (paper uses 2)")

    p.add_argument("--train_steps", type=int, default=2000, help="2k small-scale; 10k large-scale")
    p.add_argument("--batch_size", type=int, default=4, help="4 small-scale; 16 large-scale")
    p.add_argument("--lr", type=float, default=4e-4, help="4e-4 small-scale; 2e-4 large-scale")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)

    p.add_argument("--lora_rank", type=int, default=8, help="LoRA inner dim (8 small-scale; 32 large-scale)")

    p.add_argument("--train_prompts_file", type=str, default=None)
    p.add_argument("--val_prompts_file", type=str, default=None)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--num_val_prompts", type=int, default=100)

    p.add_argument("--sample_every", type=int, default=200)
    p.add_argument("--samples_per_prompt", type=int, default=1)
    p.add_argument("--save_samples_dir", type=str, default="samples")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")

    return p.parse_args()
