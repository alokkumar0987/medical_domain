"""
Run TotalSegmentator on LIDC-IDRI data and show the output.

TotalSegmentator accepts:
  - A folder of DICOM slices (e.g. LIDC-IDRI-0004)
  - A NIfTI file
  - A ZIP of DICOMs

Prerequisites:
  pip install TotalSegmentator
  (Or from repo: cd TotalSegmentator && pip install -e .)

Usage (from project root):
  python scripts/run_totalseg_on_lidc.py
  python scripts/run_totalseg_on_lidc.py "D:\\path\\to\\LIDC-IDRI-0004"
  python scripts/run_totalseg_on_lidc.py "D:\\path\\to\\volume.nii.gz"

Output is written to data/totalseg/LIDC-IDRI-0004 (or a label derived from the input path).
After processing, the script lists all produced mask files and prints a short summary.
"""

import os
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))).resolve()
TOTALSEG_ROOT = Path(os.getenv("TOTALSEG_ROOT", str(DATA_ROOT / "totalseg"))).resolve()
TOTALSEG_BIN = os.getenv("TOTALSEG_BIN", "TotalSegmentator")

# Default: LIDC-IDRI-0004 under manifest
DEFAULT_INPUT = PROJECT_ROOT / "manifest-1585232716547" / "LIDC-IDRI" / "LIDC-IDRI-0004"


def run_totalsegmentator(input_path: Path, output_dir: Path, fast: bool = True) -> bool:
    """Run TotalSegmentator CLI. Returns True on success."""
    base = TOTALSEG_BIN.split() if " " in TOTALSEG_BIN else [TOTALSEG_BIN]
    cmd = base + ["-i", str(input_path), "-o", str(output_dir)]
    if fast:
        cmd.append("--fast")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        return True
    except subprocess.CalledProcessError as e:
        print(f"TotalSegmentator failed with exit code {e.returncode}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(
            f"TotalSegmentator not found. Install with: pip install TotalSegmentator\n"
            f"Or set TOTALSEG_BIN to the full path of the CLI.",
            file=sys.stderr,
        )
        return False


def main() -> int:
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1]).resolve()
    else:
        input_path = DEFAULT_INPUT

    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        print("Create the folder or pass a valid path, e.g.:")
        print('  python scripts/run_totalseg_on_lidc.py "D:\\path\\to\\LIDC-IDRI-0004"')
        return 1

    # Output dir name from input (e.g. LIDC-IDRI-0004 or volume)
    out_name = input_path.name if input_path.is_dir() else input_path.stem.replace(".nii", "")
    output_dir = TOTALSEG_ROOT / out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TotalSegmentator test run")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print()

    if not run_totalsegmentator(input_path, output_dir, fast=True):
        return 1

    # List and summarize output
    nifti_files = sorted(output_dir.glob("*.nii.gz"))
    print()
    print("=" * 60)
    print("Output after TotalSegmentator processing")
    print("=" * 60)
    print(f"Directory: {output_dir}")
    print(f"Number of mask files: {len(nifti_files)}")
    print()
    if nifti_files:
        print("Mask files (organ / structure labels):")
        for f in nifti_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}  ({size_mb:.2f} MB)")
        # LIDC is lung CT – highlight lung-related
        lung_like = [f for f in nifti_files if "lung" in f.name.lower() or "heart" in f.name.lower() or "liver" in f.name.lower()]
        if lung_like:
            print()
            print("Lung/thorax-related structures (relevant for LIDC nodules):")
            for f in lung_like:
                print(f"  {f.name}")
    else:
        print("No .nii.gz files found in output directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
