"""
Download MedSAM2 model checkpoints (Windows-friendly; no bash required).

Run from project root:
  python scripts/download_medsam2_checkpoints.py

Checkpoints are written to MedSAM2/checkpoints/ (or MEDSAM2_ROOT/checkpoints if set).
"""

import os
import sys
from pathlib import Path

# Project root: one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDSAM2_ROOT = Path(os.getenv("MEDSAM2_ROOT", PROJECT_ROOT / "MedSAM2"))
CHECKPOINTS_DIR = MEDSAM2_ROOT / "checkpoints"

# HuggingFace MedSAM2 checkpoints (same as download.sh)
HF_BASE = "https://huggingface.co/wanglab/MedSAM2/resolve/main"
MODELS = [
    "MedSAM2_2411.pt",
    "MedSAM2_US_Heart.pt",
    "MedSAM2_MRI_LiverLesion.pt",
    "MedSAM2_CTLesion.pt",
    "MedSAM2_latest.pt",
]

# Optional: SAM2 base (for training / some configs)
SAM2_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_MODEL = "sam2.1_hiera_tiny.pt"


def download_file(url: str, dest: Path) -> None:
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  urllib failed: {e}")
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                dest.write_bytes(resp.read())
        except Exception as e2:
            raise RuntimeError(f"Download failed: {e2}") from e2


def main() -> None:
    if not MEDSAM2_ROOT.is_dir():
        print(f"MEDSAM2 root not found: {MEDSAM2_ROOT}", file=sys.stderr)
        print("Clone the repo first: git clone https://github.com/bowang-lab/MedSAM2.git", file=sys.stderr)
        sys.exit(1)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MedSAM2 checkpoints to {CHECKPOINTS_DIR}")

    for name in MODELS:
        dest = CHECKPOINTS_DIR / name
        if dest.is_file():
            print(f"  Skip (exists): {name}")
            continue
        url = f"{HF_BASE}/{name}"
        print(f"  Downloading {name}...")
        download_file(url, dest)
        print(f"  Saved: {dest}")

    # SAM2 base model (used by some configs)
    sam2_dest = CHECKPOINTS_DIR / SAM2_MODEL
    if not sam2_dest.is_file():
        url = f"{SAM2_BASE}/{SAM2_MODEL}"
        print(f"  Downloading {SAM2_MODEL}...")
        download_file(url, sam2_dest)
        print(f"  Saved: {sam2_dest}")
    else:
        print(f"  Skip (exists): {SAM2_MODEL}")

    print("Done. Use MedSAM2_latest.pt or MedSAM2_CTLesion.pt for CT lesion segmentation.")


if __name__ == "__main__":
    main()
