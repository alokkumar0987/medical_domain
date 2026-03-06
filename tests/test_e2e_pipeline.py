"""
End-to-end test: Create case -> Confirm point -> Segment -> Analyze.

Uses a minimal 3D NIfTI (small volume), no TotalSegmentator. Assumes API at
http://127.0.0.1:8000. Run with: python -m tests.test_e2e_pipeline
"""

import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import httpx

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_URL = os.getenv("PIPELINE_API_URL", "http://127.0.0.1:8000")
REPORT = "A 4.5 cm hypo-attenuating mass is noted in the superior pole of the right kidney."


def make_mini_nifti() -> Path:
    """Create a small 3D NIfTI (30x30x30) and return path."""
    shape = (30, 30, 30)
    data = np.zeros(shape, dtype=np.float32)
    # Add a small "lesion" blob in the center for mask/volume
    data[12:18, 12:18, 12:18] = 100.0
    img = nib.Nifti1Image(data, np.eye(4))
    fd, path = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    nib.save(img, path)
    return Path(path)


def run_e2e():
    nifti_path = make_mini_nifti()
    try:
        with httpx.Client(timeout=60.0) as client:
            # 1. Create case
            with open(nifti_path, "rb") as f:
                content = f.read()
            r = client.post(
                f"{API_URL}/cases",
                files={"scan_file": ("mini.nii.gz", content, "application/gzip")},
                data={"modality": "ct", "report_text": REPORT},
            )
            r.raise_for_status()
            case = r.json()
            case_id = case["case_id"]
            print(f"[1] Create case: case_id={case_id}")
            print(f"    shape={case['scan']['shape']}, intent={case['intent'].get('organ')}")

            # 2. Confirm point (human-in-the-loop)
            point = {"x": 15, "y": 15, "z": 15}
            r = client.post(f"{API_URL}/cases/{case_id}/confirm-point", json=point)
            r.raise_for_status()
            print(f"[2] Confirm point: {r.json()['confirmed_point_voxel']}")

            # 3. Verification
            r = client.get(f"{API_URL}/cases/{case_id}/verification")
            r.raise_for_status()
            print(f"[3] Verification: status={r.json().get('verification_status')}")

            # 4. Segment
            r = client.post(f"{API_URL}/cases/{case_id}/segment")
            r.raise_for_status()
            print(f"[4] Segment: {r.json().get('mask_path')}")

            # 5. Analyze
            r = client.post(f"{API_URL}/cases/{case_id}/analyze")
            r.raise_for_status()
            m = r.json()
            print(f"[5] Analyze: mesh={m.get('mesh_path')}, volume_cm3={m.get('volume_cm3')}, max_diameter_cm={m.get('max_diameter_cm')}")

            print("\n[OK] End-to-end data flow passed.")
            return 0
    except httpx.ConnectError as e:
        print(f"[ERROR] Cannot reach API at {API_URL}. Start backend: uvicorn main:app --host 127.0.0.1 --port 8000")
        return 1
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] API error: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 400 and "OPENROUTER_API_KEY" in (e.response.text or ""):
            print("  -> Set OPENROUTER_API_KEY in .env or environment, then restart backend (uvicorn).")
        return 1
    finally:
        nifti_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(run_e2e())
