#!/usr/bin/env bash
set -euxo pipefail

# Basic system deps
apt-get update
apt-get install -y git git-lfs wget tmux libgl1 libglib2.0-0
git lfs install

# Python deps
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir \
  accelerate==0.34.2 \
  transformers==4.43.3 \
  diffusers==0.30.3 \
  datasets==2.20.0 \
  peft==0.12.0 \
  xformers==0.0.27.post2 \
  safetensors==0.4.3 \
  torchvision==0.18.1 \
  timm==1.0.9 \
  sentencepiece \
  bitsandbytes==0.43.2 \
  wandb \
  huggingface_hub[cli]==0.24.6
pip install pickscore==0.1.5 || true


# Workspace layout
mkdir -p /workspace/{data,checkpoints,logs,code}



