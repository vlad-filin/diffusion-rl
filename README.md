# Diffusion-RL: DRaFT with PickScore on Stable Diffusion 1.5

This project implements **Directly Finetuning Diffusion Models on Differentiable Rewards (DRaFT)** on top of **Stable Diffusion 1.5**, using **PickScore** as the reward function.  
It supports:
- LoRA-based finetuning (`finetine_draft_pickscore.py`)
- Prompt-based sampling during training
- Offline inference with or without LoRA (`inference.py`)
- Utility scripts for reproducibility (`launch.sh`, `download.sh`)

---
![Evolution of generated images during training](draft2_vis.png)
## Setup

Install dependencies and create a virtual environment:

`bash launch.sh`

Download Stable Diffusion 1.5 and other models using:

`bash download.sh`


---

## Training

Prepare prompts filelists.txt (one prompt per line ) and launch training using following:

`python finetine_draft_pickscore.py --pretrained_model_name_or_path /workspace/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14 --output_dir /workspace/checkpoints/sd15_draft2_pickscore --train_prompts_file /workspace/data/train.txt --train_steps 2000 --batch_size 6 --ddim_steps 50 --K 2 --guidance_scale 7.5 --lr 4e-4 --weight_decay 0.1 --lora_rank 8 --sample_every 200 --save_samples_dir /workspace/samples/sd15_draft2_pickscore --mixed_precision bf16 --gradient_checkpointing `

## Inference

Launch inference using following:

`python inference.py \
  --sd15_dir /workspace/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/<snapshot_id> \
  --prompts_file data/test_prompts.txt \
  --out_dir infer_outputs \
  --ddim_steps 50 \
  --guidance 7.5 \
  --batch_size 4 \
  --samples_per_prompt 1 \
  --seed 1234 \
  --precision fp16 \
  --lora_dir outputs/lora_sd15_draft_rank8_steps2000 \
  --with_pickscore`
---
Model checkpoints and more generated samples are available [here](https://drive.google.com/drive/folders/1lWGWkSWsDtsD-lb2RybOiwIrQLOq_8Ik?usp=sharing)