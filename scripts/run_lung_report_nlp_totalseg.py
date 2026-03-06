"""
Lung CT pipeline: text report -> NLP extraction -> TotalSegmentator.

Flow:
  1. Use a written radiology report (lung CT, e.g. LIDC nodule findings).
  2. Run NLP extraction to get organ / region / finding (clinical intent).
  3. Run TotalSegmentator on the CT scan folder to get organ masks.

LIDC-IDRI: If you pass a patient folder, the pipeline auto-selects the CT series
(skips SEG/RTSTRUCT/1-bit and picks the folder with the most CT DICOMs).
To see which series exist: python scripts/find_lidc_ct_series.py "path/to/LIDC-IDRI-XXXX".
Some LIDC downloads contain only annotations (SEG/SR); the CT series must be present.

Run with venv activated:
  python scripts/run_lung_report_nlp_totalseg.py [path_to_patient_or_ct_series_or_nifti]
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))).resolve()
TOTALSEG_ROOT = Path(os.getenv("TOTALSEG_ROOT", str(DATA_ROOT / "totalseg"))).resolve()
TOTALSEG_BIN = os.getenv("TOTALSEG_BIN", "TotalSegmentator")

# Default LIDC path (patient folder; CT selector will find the CT series)
DEFAULT_CT_PATH = PROJECT_ROOT / "manifest-1585232716547" / "LIDC-IDRI" / "LIDC-IDRI-0004"


def get_best_ct_series(patient_dir: Path) -> Optional[Path]:
    """
    LIDC-IDRI: find the folder that contains the actual CT series (many CT DICOMs),
    not SEG/RTSTRUCT or annotation series (1-bit masks). Returns the path to the
    folder with the most CT DICOM files and valid bit depth, or None if none found.
    """
    try:
        import pydicom
    except ImportError:
        return None
    candidates: List[Tuple[Path, int, bool]] = []  # (folder, n_files, is_ct)
    patient_dir = Path(patient_dir).resolve()
    for root, _dirs, files in os.walk(patient_dir):
        root_path = Path(root)
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue
        sample = root_path / dcm_files[0]
        try:
            ds = pydicom.dcmread(str(sample), stop_before_pixels=True)
        except Exception:
            continue
        modality = (getattr(ds, "Modality", "") or "").upper()
        if modality in ("SEG", "RTSTRUCT", "RTS"):
            continue
        bits_allocated = getattr(ds, "BitsAllocated", 0) or 0
        if bits_allocated < 8:
            continue
        is_ct = modality == "CT"
        candidates.append((root_path, len(dcm_files), is_ct))
    if not candidates:
        return None
    # Prefer CT modality, then by most files
    candidates.sort(key=lambda x: (not x[2], -x[1]))
    return candidates[0][0]


# ----- 1. Lung CT report text (LIDC-style findings) -----
LUNG_CT_REPORT = (
    "CT chest without contrast: Lung windows demonstrate multiple pulmonary nodules. "
    "Right upper lobe solid nodule measures approximately 9 mm. "
    "Left lower lobe subpleural nodule measures 6 mm. "
    "No pleural effusion or pneumothorax. Mediastinum is unremarkable. "
    "Recommend short-term follow-up CT for the dominant right upper lobe nodule."
)


def run_nlp_extraction(report_text: str) -> dict:
    """Step 1: Extract clinical intent (organ, region, finding) from report."""
    from nlp.clinical_intent_extractor import extract_clinical_intent
    return extract_clinical_intent(report_text)


def run_totalsegmentator(input_path: Path, output_dir: Path, fast: bool = True) -> bool:
    """Step 2: Run TotalSegmentator on CT input (folder or NIfTI)."""
    base = TOTALSEG_BIN.split() if " " in TOTALSEG_BIN else [TOTALSEG_BIN]
    cmd = base + ["-i", str(input_path), "-o", str(output_dir)]
    if fast:
        cmd.append("--fast")
    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        return True
    except subprocess.CalledProcessError as e:
        print(f"  TotalSegmentator failed with exit code {e.returncode}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(
            "  TotalSegmentator not found. Install: pip install TotalSegmentator",
            file=sys.stderr,
        )
        return False


def main() -> int:
    ct_path = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_CT_PATH

    print("=" * 60)
    print("Lung CT pipeline: Report -> NLP -> TotalSegmentator")
    print("=" * 60)

    # ----- Step 1: Report text -----
    print("\n--- 1. Report text (lung CT) ---")
    print(LUNG_CT_REPORT)

    # ----- Step 2: NLP extraction -----
    print("\n--- 2. NLP extraction (clinical intent) ---")
    try:
        intent = run_nlp_extraction(LUNG_CT_REPORT)
        print("Extracted intent:")
        print(json.dumps(intent, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"NLP extraction failed: {e}", file=sys.stderr)
        return 1

    # ----- Step 3: TotalSegmentator on CT -----
    print("\n--- 3. TotalSegmentator on CT scan ---")
    if not ct_path.exists():
        print(f"CT path does not exist: {ct_path}", file=sys.stderr)
        print("Pass the patient folder (e.g. LIDC-IDRI-0004) or the CT series folder.")
        return 1

    # LIDC-IDRI: if path is a directory, auto-select CT series (skip SEG/annotation)
    actual_ct_path = ct_path
    if ct_path.is_dir():
        best = get_best_ct_series(ct_path)
        if best is not None and best != ct_path:
            actual_ct_path = best
            n_dcm = len(list(actual_ct_path.glob("*.dcm")))
            print(f"  LIDC: using CT series folder ({n_dcm} DICOMs):")
            print(f"    {actual_ct_path}")
        elif best is None:
            print("  LIDC: no CT series found (only SEG/1-bit?). Pass a folder that contains CT DICOMs, or use find_lidc_ct_series.py to list series.", file=sys.stderr)
            return 1
        else:
            print(f"  Input: {ct_path} (using as single series)")

    out_name = ct_path.name if ct_path.is_dir() else ct_path.stem.replace(".nii", "")
    output_dir = TOTALSEG_ROOT / out_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if actual_ct_path == ct_path and ct_path.is_dir():
        print(f"  Input:  {ct_path}")
    print(f"  Output: {output_dir}")

    if not run_totalsegmentator(actual_ct_path, output_dir, fast=True):
        return 1

    # ----- Output summary -----
    nifti_files = sorted(output_dir.glob("*.nii.gz"))
    print("\n--- Output after TotalSegmentator ---")
    print(f"  Directory: {output_dir}")
    print(f"  Mask files: {len(nifti_files)}")
    lung_related = [f.name for f in nifti_files if "lung" in f.name.lower() or "heart" in f.name.lower()]
    if lung_related:
        print("  Lung/thorax-related:")
        for name in lung_related:
            print(f"    {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
