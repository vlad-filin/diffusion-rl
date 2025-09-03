import os
from huggingface_hub import login, snapshot_download

tok = os.environ.get("HF_TOKEN")
if tok: login(token=tok, add_to_git_credential=True)


MODELS = [
  # Base model for training
  ("runwayml/stable-diffusion-v1-5",
   ["*.json","*.bin","*.safetensors","*config*","*.txt"]),

  # Optional extras (uncomment if you need them)
    ("stabilityai/sd-vae-ft-mse", ["*.safetensors","*config*"]),
    ("openai/clip-vit-large-patch14", ["*.json","*.txt","pytorch_model*.bin","*.safetensors","*merges.txt","*vocab.json"]),

  # Reward model(s) for DRAFT — fill in your exact repo(s) when decided
   ("yuvalkirstain/PickScore_v1", ["*.safetensors","*.bin","*config*", "*.json","*.txt"]),
     ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", ["*.safetensors","*.bin","*config*","*.json","*.txt"])
  # ("<your-pickscore-repo-id>", ["*"]),
]

for repo, patterns in MODELS:
    print(f"==> Caching {repo} ...")
    snapshot_download(
        repo_id=repo,
        allow_patterns=patterns,
        resume_download=True
    )
print("✅ All requested repos cached.")