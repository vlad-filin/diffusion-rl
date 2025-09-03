import pandas as pd
import csv
from pathlib import Path

# Input and output paths
src_path = "PartiPrompts.tsv"
train_out = "train.txt"
val_out = "val.txt"

def detect_prompt_column(df):
    """Try to guess the column that contains prompts."""
    candidates = []
    for col in df.columns:
        lower = str(col).strip().lower()
        if any(k in lower for k in ["prompt", "text", "caption", "description", "instruction"]):
            candidates.append(col)
    if candidates:
        priority = ["prompt", "text", "caption", "description", "instruction"]
        def score(c):
            l = str(c).strip().lower()
            for i, k in enumerate(priority):
                if k in l:
                    return i
            return 999
        candidates.sort(key=score)
        return candidates[0]

    # fallback: longest average string length
    def avg_len(series):
        s = series.dropna().astype(str).str.len()
        return s.mean() if len(s) else 0
    return max(df.columns, key=lambda c: avg_len(df[c]))

def main():
    # Load TSV
    try:
        df = pd.read_csv(src_path, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(src_path, sep="\t", quoting=csv.QUOTE_NONE, engine="python", on_bad_lines="skip")

    # Pick prompt column
    prompt_col = detect_prompt_column(df)

    # Extract and clean
    prompts = (
        df[prompt_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    prompts = prompts[prompts.str.len() > 0].drop_duplicates().tolist()

    # Split into val/train
    val_prompts = prompts[-100:]
    train_prompts = prompts[:-100]

    Path(val_out).write_text("\n".join(val_prompts), encoding="utf-8")
    Path(train_out).write_text("\n".join(train_prompts), encoding="utf-8")

    print(f"Saved {len(val_prompts)} prompts to {val_out}")
    print(f"Saved {len(train_prompts)} prompts to {train_out}")

if __name__ == "__main__":
    main()
