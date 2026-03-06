"""
Find the CT series folder inside a LIDC-IDRI patient folder.
Run to see which folders have DICOMs and which one is selected as CT.

  python scripts/find_lidc_ct_series.py "D:\\path\\to\\LIDC-IDRI-0004"
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_lung_report_nlp_totalseg import get_best_ct_series


def main() -> int:
    patient_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else PROJECT_ROOT / "manifest-1585232716547" / "LIDC-IDRI" / "LIDC-IDRI-0004"
    if not patient_dir.is_dir():
        print(f"Not a directory: {patient_dir}")
        return 1
    try:
        import pydicom
    except ImportError:
        print("Install pydicom: pip install pydicom")
        return 1
    print(f"Scanning: {patient_dir}\n")
    for root, _dirs, files in os.walk(patient_dir):
        root_path = Path(root)
        dcm = [f for f in files if f.lower().endswith(".dcm")]
        if not dcm:
            continue
        sample = root_path / dcm[0]
        try:
            ds = pydicom.dcmread(str(sample), stop_before_pixels=True)
        except Exception as e:
            print(f"  {root_path.relative_to(patient_dir)}: {len(dcm)} .dcm (read err: {e})")
            continue
        mod = getattr(ds, "Modality", "") or ""
        bits = getattr(ds, "BitsAllocated", 0) or 0
        skip = mod.upper() in ("SEG", "RTSTRUCT", "RTS") or bits < 8
        print(f"  {root_path.relative_to(patient_dir)}: {len(dcm)} .dcm  Modality={mod} BitsAllocated={bits}  {'(skip)' if skip else '(candidate)'}")
    best = get_best_ct_series(patient_dir)
    print(f"\nSelected CT series: {best}")
    if best:
        print(f"  Use: python scripts/run_lung_report_nlp_totalseg.py \"{best}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
