from dataclasses import dataclass
from typing import List, Optional, Tuple
import inspect
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import AutoTokenizer, CLIPTextModel, AutoModel
from utils import clip_preprocess_tensor
from peft import LoraConfig

# -----------------------------
# Dataset
# -----------------------------
class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# -----------------------------
# PickScore (CLIP-like reward)
# -----------------------------
@dataclass
class PickScore:
    processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_name: str = "yuvalkirstain/PickScore_v1"
    #processor_name = "/workspace/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c"
    #model_name  = "/workspace/.cache/huggingface/hub/models--yuvalkirstain--PickScore_v1/snapshots/a4e4367c6dfa7288a00c550414478f865b875800"
    def __post_init__(self):
        #kw = dict(local_files_only=True, cache_dir="/workspace/.cache/huggingface")
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.processor_name,)# **kw)
        self.model = AutoModel.from_pretrained(self.model_name, )#**kw)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.model.to(device)
        return self

    def score(self, pixel_imgs: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        Compute PickScore for a batch of images (B,3,H,W) in [0,1] and list of texts.
        Returns tensor of shape (B,) with differentiable path through the image.
        """
        device = pixel_imgs.device
        # Preprocess images differentiably
        px = clip_preprocess_tensor(pixel_imgs)
        # Get image features (no grad in model params, but keep grad wrt inputs)
        image_embs = self.model.get_image_features(pixel_values=px)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        # Text features
        tok = self.text_tokenizer(
            texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            text_embs = self.model.get_text_features(**tok)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        # Scale
        logit_scale = self.model.logit_scale.exp()
        # Cosine similarity
        scores = logit_scale *(text_embs * image_embs).sum(dim=-1)
        return scores


# -----------------------------
# LoRA utilities
# -----------------------------
def add_lora_to_unet(unet, rank: int = 16):
    """
    0.35.1-compatible LoRA install for SD 1.5 UNet.
    Uses PEFT LoraConfig + unet.add_adapter; returns the trainable params list.
    """
    unet_lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # installs LoRA layers internally (no adapter_name needed here)
    unet.add_adapter(unet_lora_cfg)

    # return only the LoRA params (what the official guide optimizes)
    return [p for p in unet.parameters() if p.requires_grad]

# -----------------------------
# SD1.5 wrapper
# -----------------------------
@dataclass
class SD15:
    model_name: str
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self):
        #kw = dict(local_files_only=True, cache_dir="/workspace/.cache/huggingface")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, subfolder="tokenizer",)# **kw)
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_name, subfolder="text_encoder",)# **kw)
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae",)# **kw)
        self.unet = UNet2DConditionModel.from_pretrained(self.model_name, subfolder="unet",)# **kw)
        self.scheduler = DDIMScheduler.from_pretrained(self.model_name, subfolder="scheduler",)# **kw)
        self.text_encoder.to(self.device, dtype=self.dtype)
        self.vae.to(self.device, dtype=self.dtype)
        self.unet.to(self.device, dtype=self.dtype)

        self.vae_scale_factor = 0.18215

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
    def encode_text(self, prompts: List[str]):
        tok = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            enc = self.text_encoder(**tok)
        return enc.last_hidden_state

    def get_uncond_emb(self, batch_size: int):
        return self.encode_text([""] * batch_size)

    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.vae_scale_factor
        imgs = self.vae.decode(latents).sample
        imgs = (imgs.clamp(-1, 1) + 1) / 2  # to [0,1]
        return imgs

    def vae_encode(self, pixel_imgs: torch.Tensor) -> torch.Tensor:
        imgs = (pixel_imgs * 2 - 1).clamp(-1, 1)
        latents = self.vae.encode(imgs).latent_dist.mode()
        latents = latents * self.vae_scale_factor
        return latents


# -----------------------------
# DRaFT sampling & training core
# -----------------------------
@dataclass
class DraftConfig:
    ddim_steps: int = 50
    guidance_scale: float = 7.5
    method: str = "draft-k"  # {draft, draft-k, draft-lv}
    K: int = 1
    n_lv: int = 2
    gradient_checkpointing: bool = True


def ddim_timesteps_for(scheduler: DDIMScheduler, steps: int) -> torch.LongTensor:
    scheduler.set_timesteps(steps)
    return scheduler.timesteps


def sd_forward_unroll(
    sd: SD15,
    prompts: List[str],
    cfg: DraftConfig,
    generator: Optional[torch.Generator] = None,
    enable_grad: bool = True,
):
    device = sd.unet.device
    dtype = sd.unet.dtype
    bsz = len(prompts)

    # Text embeddings (cond & uncond)
    text_emb = sd.encode_text(prompts)
    uncond_emb = sd.get_uncond_emb(bsz)
    context = torch.cat([uncond_emb, text_emb], dim=0)

    timesteps = ddim_timesteps_for(sd.scheduler, cfg.ddim_steps)

    # Init latents
    shape = (bsz, sd.unet.in_channels, sd.vae.config.sample_size, sd.vae.config.sample_size)
    shape = (shape[0], shape[1], shape[2] // 8, shape[3] // 8)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # Truncation boundary for DRaFT-K
    if cfg.method == "draft":
        k_boundary = 0
    else:
        k = max(1, min(cfg.K, cfg.ddim_steps))
        k_boundary = len(timesteps) - k

    # Enable gradient checkpointing at the module level
    if cfg.gradient_checkpointing:
        sd.unet.enable_gradient_checkpointing()
    else:
        sd.unet.disable_gradient_checkpointing()

    # Unroll sampling
    for i, t in enumerate(timesteps):
        do_grad = enable_grad and (i >= k_boundary)
        ctx = torch.enable_grad() if do_grad else torch.no_grad()
        with ctx:
            latent_in = torch.cat([latents] * 2, dim=0)
            latent_in = sd.scheduler.scale_model_input(latent_in, t)
            noise_pred = sd.unet(latent_in, t, encoder_hidden_states=context).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + cfg.guidance_scale * (noise_text - noise_uncond)
            latents = sd.scheduler.step(noise_pred, t, latents).prev_sample

    pixels = sd.vae_decode(latents)
    return pixels, latents, timesteps


def draft_lv_last_step(
    sd: SD15,
    x0_pixels: torch.Tensor,
    prompts: List[str],
    timesteps: torch.LongTensor,
    n_lv: int,
    guidance_scale: float,
):
    """
    Low-variance estimator for K=1: re-noise x0 to t_1 (the last used timestep),
    do one denoising step multiple times with different noises and sum rewards.
    Returns list of pixel tensors (one per LV inner loop).
    """
    assert n_lv >= 1
    device = sd.unet.device
    bsz = x0_pixels.shape[0]
    z0 = sd.vae_encode(x0_pixels)
    t_last = timesteps[-1]

    outs = []
    for _ in range(n_lv):
        noise = torch.randn_like(z0)
        zt = sd.scheduler.add_noise(z0, noise, t_last)
        latent_in = torch.cat([zt] * 2, dim=0)
        latent_in = sd.scheduler.scale_model_input(latent_in, t_last)
        text_emb = sd.encode_text(prompts)
        uncond_emb = sd.get_uncond_emb(bsz)
        context = torch.cat([uncond_emb, text_emb], dim=0)
        noise_pred = sd.unet(latent_in, t_last, encoder_hidden_states=context).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        z_prev = sd.scheduler.step(noise_pred, t_last, zt).prev_sample
        x_pixels = sd.vae_decode(z_prev)
        outs.append(x_pixels)
    return outs
