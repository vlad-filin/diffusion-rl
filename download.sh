#!/usr/bin/env bash
set -euxo pipefail

apt-get update
apt-get install -y git git-lfs aria2 tmux
git lfs install

python -V
pip install --upgrade pip
pip install 'huggingface_hub[cli]==0.24.6' hf_transfer==0.1.6

mkdir -p /workspace/.cache/huggingface /workspace/code /workspace/logs
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN= #your token
python /workspace/code/cache_models.py

