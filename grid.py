#!/usr/bin/env python3
import argparse, csv, re
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Example file: 20250902T193512Z_p00007_00.png
FNAME_RE = re.compile(
    r'(?P<ts>\d{8}T\d{6}Z)[_\-]p(?P<pid>\d{5})[_\-](?P<idx>\d{2})',
    re.IGNORECASE
)

def parse_name(stem: str):
    m = FNAME_RE.search(stem)
    if not m: return None
    ts = datetime.strptime(m.group("ts"), "%Y%m%dT%H%M%SZ")
    return ts, f"p{m.group('pid')}", int(m.group("idx"))

def to_pid_str(n: int) -> str:
    return f"p{n:05d}"

def to_step_dirname(n: int) -> str:
    return f"step_{n:07d}"

def normalize_prompt_text(s: str) -> str:
    # Preserve content verbatim but normalize whitespace for single-line caption rendering
    # (no truncation; all characters kept)
    return " ".join(s.replace("\r", " ").replace("\n", " ").split())

def find_latest_for_pid_in_step(step_dir: Path, pid_str: str) -> Optional[Path]:
    if not step_dir.exists(): return None
    best_ts, best_path = None, None
    # Fast glob by pid pattern
    for p in step_dir.glob(f"*{pid_str}_*.png"):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS): continue
        parsed = parse_name(p.stem)
        if not parsed: continue
        ts, pid, _ = parsed
        if pid != pid_str: continue
        if best_ts is None or ts > best_ts:
            best_ts, best_path = ts, p
    return best_path

def load_csv_prompts(step_dir: Path) -> Dict[str, str]:
    """
    Load pXXXXX -> prompt from step/index.csv.
    We keep the *last* occurrence per pid (latest sample_idx), but content is identical anyway.
    """
    m: Dict[str, str] = {}
    csv_path = step_dir / "index.csv"
    if not csv_path.exists(): return m
    # csv module handles quoted commas etc. We don't strip/trim content.
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = f"p{int(row['prompt_idx']):05d}"
                prompt = row.get("prompt", "")
                m[pid] = prompt  # last wins
            except Exception:
                # ignore malformed lines; keep going
                continue
    return m

def load_and_pad(img: Image.Image, target_size, pad, bg=(245, 245, 245)):
    tw, th = target_size
    w, h = img.size
    scale = min((tw - 2*pad) / max(1, w), (th - 2*pad) / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), bg)
    canvas.paste(img, ((tw - nw) // 2, (th - nh) // 2))
    return canvas

def draw_caption(
    tile: Image.Image,
    text: str,
    base_font: Optional[ImageFont.FreeTypeFont],
    debug_label: str = "",
    echo: bool = True
):
    if not text:
        return
    draw = ImageDraw.Draw(tile)
    W, H = tile.size
    band_h = max(20, H // 12)
    draw.rectangle([(0, H - band_h), (W, H)], fill=(0, 0, 0))

    # Provide a base font if not supplied
    font = base_font
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, band_h - 6))
        except Exception:
            font = ImageFont.load_default()

    # Measurement helper (Pillow 10+ uses textbbox)
    def measure(txt: str, fnt):
        try:
            bbox = draw.textbbox((0, 0), txt, font=fnt)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            return draw.textsize(txt, font=fnt)

    # Fit-to-width by reducing font size, never truncating
    original_size = getattr(font, "size", band_h - 2)
    size = original_size
    tw, th = measure(text, font)
    while tw > W - 4 and size > 8:
        size -= 2
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            font = ImageFont.load_default()
        tw, th = measure(text, font)

    # If still too wide, left align so full text is drawn (may overflow to the right — allowed)
    overflow = tw > W - 2
    x = 2 if overflow else (W - tw) // 2
    y = H - band_h + (band_h - th) // 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    # Echo to stdout exactly what was rendered
    if echo:
        status = []
        if size != original_size: status.append(f"shrunk:{original_size}->{size}")
        if overflow: status.append("overflow-right")
        meta = f" [{', '.join(status)}]" if status else ""
        if debug_label:
            print(f"[caption]{meta} {debug_label}: {text}")
        else:
            print(f"[caption]{meta} {text}")

def parse_int_list(spec: str) -> List[int]:
    return [int(x) for x in spec.replace(",", " ").split() if x.strip()]

def main():
    ap = argparse.ArgumentParser("Grid by selected prompt IDs (rows) × steps (cols) with full prompt captions")
    ap.add_argument("--root", required=True, help=".../samples/<exp>/train")
    ap.add_argument("--pids", required=True, help="e.g. '1,3,7'")
    ap.add_argument("--steps", required=True, help="e.g. '1,400,1200,2000'")
    ap.add_argument("--out", "-o", default="grid_pid_step.png")
    ap.add_argument("--tile_w", type=int, default=512)
    ap.add_argument("--tile_h", type=int, default=512)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--captions", action="store_true", help="Draw captions on tiles")
    ap.add_argument("--blank_label", default="missing")
    args = ap.parse_args()

    root = Path(args.root)
    pid_strs = [to_pid_str(n) for n in parse_int_list(args.pids)]
    step_nums = parse_int_list(args.steps)
    step_dirs = [root / to_step_dirname(s) for s in step_nums]

    # Collect prompt map from all steps; last write wins (prompts should match across steps)
    prompt_map: Dict[str, str] = {}
    for sdir in step_dirs:
        prompt_map.update(load_csv_prompts(sdir))

    # Prepare font once
    try:
        base_font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:
        base_font = None

    grid_W = len(step_dirs) * args.tile_w
    grid_H = len(pid_strs) * args.tile_h
    grid = Image.new("RGB", (grid_W, grid_H), (255, 255, 255))

    for r, pid in enumerate(pid_strs):
        prompt_raw = prompt_map.get(pid, "")
        prompt_text = normalize_prompt_text(prompt_raw)  # keep full content; only normalize whitespace
        for c, step_num in enumerate(step_nums):
            sdir = root / to_step_dirname(step_num)
            path = find_latest_for_pid_in_step(sdir, pid)
            cell_x, cell_y = c * args.tile_w, r * args.tile_h

            if path is None:
                tile = Image.new("RGB", (args.tile_w, args.tile_h), (230, 230, 230))
                caption = f"{pid} · step {step_num} · {args.blank_label}" if args.captions else ""
                if caption:
                    draw_caption(tile, caption, base_font, debug_label=f"{pid} · step {step_num}")
            else:
                img = Image.open(path).convert("RGB")
                tile = load_and_pad(img, (args.tile_w, args.tile_h), args.pad)
                caption = ""
                if args.captions:
                    # First column: prepend FULL prompt text (no truncation)
                    if c == 0 and prompt_text:
                        caption = f"{prompt_text} | {pid} · step {step_num}"
                    else:
                        caption = f"{pid} · step {step_num}"
                    draw_caption(tile, caption, base_font, debug_label=f"{pid} · step {step_num}")

            grid.paste(tile, (cell_x, cell_y))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved {out_path} | rows={len(pid_strs)} × cols={len(step_dirs)}")

if __name__ == "__main__":
    main()
