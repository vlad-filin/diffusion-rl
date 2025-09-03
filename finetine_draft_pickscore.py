from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from args import parse_args, load_train_val_prompts
from utils import seed_everything, save_samples, ensure_dir
from models import (
    PromptDataset,
    PickScore,
    add_lora_to_unet,
    SD15,
    DraftConfig,
    sd_forward_unroll,
    draft_lv_last_step,
)


def main():
    args = parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mixed precision
    dtype = torch.float32
    if args.mixed_precision == "bf16" and torch.cuda.is_available():
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16" and torch.cuda.is_available():
        dtype = torch.float16

    # Build SD1.5
    sd = SD15(args.pretrained_model_name_or_path, device=device, dtype=dtype)

    # LoRA
    #lora = add_lora_to_unet(sd.unet, rank=args.lora_rank)
    # Ensure LoRA params live on same device/dtype as UNet
    #lora.to(device=device, dtype=sd.unet.dtype)
    trainable_params = add_lora_to_unet(sd.unet, rank=args.lora_rank)
    opt= torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.adam_beta1, args.adam_beta2))
    # PickScore
    pick = PickScore().to(device)



    # Draft config
    draft_cfg = DraftConfig(
        ddim_steps=args.ddim_steps,
        guidance_scale=args.guidance_scale,
        method=args.method,
        K=args.K,
        n_lv=args.n_lv,
        gradient_checkpointing=(args.gradient_checkpointing and not args.no_gradient_checkpointing),
    )

    # Prompts
    train_prompts, val_prompts = load_train_val_prompts(args)
    train_ds = PromptDataset(train_prompts) if train_prompts else PromptDataset(["a photo of a dog"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    samples_dir = Path(args.save_samples_dir)
    ensure_dir(samples_dir)

    # Training loop
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    progress = tqdm(total=args.train_steps, desc="Training", dynamic_ncols=True)
    lora_dir = out_dir / f"lora_sd15_draft_rank{args.lora_rank}_steps{args.train_steps}"
    sd.unet.save_attn_procs(lora_dir)
    print(f"Saved LoRA to {lora_dir}")
    while global_step < args.train_steps:
        for batch_prompts in train_loader:
            if global_step >= args.train_steps:
                break

            # Forward unroll (with gradients in the last-K steps according to method)
            gen = torch.Generator(device=device).manual_seed(args.seed + global_step)
            autocast_ctx = (
                torch.cuda.amp.autocast if args.mixed_precision in {"fp16", "bf16"} else torch.cpu.amp.autocast
            )
            with autocast_ctx(enabled=(args.mixed_precision != "no"), dtype=dtype):
                pixels, z0, timesteps = sd_forward_unroll(
                    sd, list(batch_prompts), cfg=draft_cfg, generator=gen, enable_grad=True
                )

                if args.method == "draft-lv":
                    if draft_cfg.K != 1:
                        raise ValueError("DRaFT-LV requires K=1 (last-step).")
                    # Average rewards over n_lv re-noisings of x0
                    inner_imgs = draft_lv_last_step(
                        sd, pixels.detach(), list(batch_prompts), timesteps, draft_cfg.n_lv, draft_cfg.guidance_scale
                    )
                    rewards = [pick.score(x, list(batch_prompts)) for x in inner_imgs]
                    reward = torch.stack(rewards, dim=0).mean(dim=0)
                else:
                    reward = pick.score(pixels, list(batch_prompts))

                loss = -reward.mean()

            # Backward + step
            opt.zero_grad(set_to_none=True)
            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            global_step += 1
            progress.update(1)
            # Sampling hook
            if (global_step % args.sample_every == 0) or (global_step == 1):
                try:
                    save_samples(
                        sd,
                        out_root=samples_dir,
                        split="train",
                        step=global_step,
                        prompts=train_prompts[: min(16, len(train_prompts))],
                        num_images_per_prompt=args.samples_per_prompt,
                        cfg=draft_cfg,
                        seed=args.seed,
                        width=512,
                        height=512,
                    )
                    save_samples(
                        sd,
                        out_root=samples_dir,
                        split="val",
                        step=global_step,
                        prompts=val_prompts[: min(16, len(val_prompts))],
                        num_images_per_prompt=args.samples_per_prompt,
                        cfg=draft_cfg,
                        seed=args.seed + 1,
                        width=512,
                        height=512,
                    )
                except Exception as e:
                    print(f"[WARN] Sampling failed at step {global_step}: {e}")

            if global_step % 50 == 0:
                print(f"step={global_step} loss={loss.item():.4f}")

    # Final samples
    save_samples(
        sd,
        out_root=samples_dir,
        split="final",
        step=global_step,
        prompts=val_prompts[: min(16, len(val_prompts))] or train_prompts[: min(16, len(train_prompts))],
        num_images_per_prompt=args.samples_per_prompt,
        cfg=draft_cfg,
        seed=args.seed + 999,
        width=512,
        height=512,
    )

    # Save LoRA weights
    lora_dir = out_dir / f"lora_sd15_draft_rank{args.lora_rank}_steps{args.train_steps}"
    sd.unet.save_attn_procs(lora_dir)
    print(f"Saved LoRA to {lora_dir}")


if __name__ == "__main__":
    main()
