"""
Test run for LIDC-IDRI-0003: report text + NLP + (optional) API verification.

Patient: LIDC-IDRI-0003 (manifest-1585232716547)
Dataset: 4 annotated nodules (Nodules 1–4) in study 01-01-2000-94866.

This script:
  1. Uses a radiology report written to match LIDC-IDRI-0003 (lung nodules).
  2. Runs Step 1 (NLP) to get organ/region/finding.
  3. Prints example bbox, confirm-point, and verification JSON.
  4. If API is running and CASE_ID is set, calls confirm-point and GET verification.

Run from project root:
  python -m examples.run_lidc_0003
  # With live API (after creating a case from 0003 scan):
  set CASE_ID=<your_case_id>
  set PIPELINE_API_URL=http://127.0.0.1:8000
  python -m examples.run_lidc_0003
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nlp.clinical_intent_extractor import extract_clinical_intent


# ----- Report text corresponding to LIDC-IDRI-0003 -----
# Manifest: Nodules 1–4 with multiple reader annotations (lung CT).
LIDC_0003_REPORT = (
    "CT chest: Multiple pulmonary nodules are noted. "
    "A solid nodule in the right lung measures approximately 8 mm; "
    "a second nodule in the left lower lobe, subpleural, measures 6 mm. "
    "Two additional small nodules are present in the right lung. "
    "No pleural effusion. Recommend short-term follow-up for the dominant right lung nodule."
)


def main():
    api_url = os.getenv("PIPELINE_API_URL", "http://127.0.0.1:8000")
    case_id = os.getenv("CASE_ID", "").strip()

    print("=" * 60)
    print("PATIENT: LIDC-IDRI-0003 (manifest-1585232716547)")
    print("Study: 01-01-2000-94866 | Nodules 1–4 annotated")
    print("=" * 60)

    # ----- Step 1: Clinical intent (NLP) -----
    print("\n--- Step 1: Clinical intent extraction (NLP) ---")
    print("Input report (corresponding to LIDC-IDRI-0003):")
    print(f"  {LIDC_0003_REPORT}")
    intent = extract_clinical_intent(LIDC_0003_REPORT)
    print("Output JSON (organ / region / finding):")
    print(json.dumps(intent, indent=2, ensure_ascii=False))

    # ----- Step 2: Anatomical grounding (example) -----
    print("\n--- Step 2: Anatomical grounding (TotalSegmentator) ---")
    print("For lung nodules, organ may be lung_right / lung_left or lung; TotalSegmentator returns rough bbox.")
    example_bbox = {
        "x": 180,
        "y": 220,
        "z": 95,
        "width": 45,
        "height": 50,
        "depth": 35,
    }
    print("Example bbox JSON (voxel coordinates):")
    print(json.dumps(example_bbox, indent=2))
    print("(Real values from POST /cases/{case_id}/ground after uploading 0003 scan.)")

    # ----- Step 3: Human-in-the-loop (confirm point) -----
    print("\n--- Step 3: Human-in-the-loop (verification) ---")
    confirmed_point_json = {"x": 200, "y": 240, "z": 108}
    print("Doctor clicks lesion center on slice; request:")
    print("  POST /cases/{case_id}/confirm-point")
    print("  Body:", json.dumps(confirmed_point_json, indent=2))
    print("Response example:")
    print(json.dumps({
        "case_id": case_id or "<case_id>",
        "message": "Verified point stored. Safe to call /segment.",
        "confirmed_point_voxel": confirmed_point_json,
    }, indent=2))

    # ----- Output before segmentation (verification JSON) -----
    print("\n--- Output before sending to segmentation model ---")
    print("  GET /cases/{case_id}/verification")
    verification_output = {
        "case_id": case_id or "<case_id>",
        "bbox_voxel": example_bbox,
        "confirmed_point_voxel": confirmed_point_json,
        "verification_status": "verified",
    }
    print("Verification response (coordinate JSON + status):")
    print(json.dumps(verification_output, indent=2))
    print("\nHuman-in-the-loop: bbox from Step 2; confirmed_point_voxel from doctor click; then POST /segment uses this.")

    # ----- Optional: live API calls -----
    if case_id and api_url:
        try:
            import httpx
            with httpx.Client(timeout=15.0) as client:
                # Confirm point
                r = client.post(
                    f"{api_url}/cases/{case_id}/confirm-point",
                    json=confirmed_point_json,
                )
                r.raise_for_status()
                print("\n--- Live API: confirm-point OK ---")
                print(json.dumps(r.json(), indent=2))
                # Verification
                r2 = client.get(f"{api_url}/cases/{case_id}/verification")
                r2.raise_for_status()
                print("\n--- Live API: GET verification ---")
                print(json.dumps(r2.json(), indent=2))
        except Exception as e:
            print(f"\nLive API call failed (is the server running?): {e}")

    print("\n" + "=" * 60)
    print("Done. To run full pipeline: upload LIDC-IDRI-0003 CT (ZIP or NIfTI), create case with this report, then ground -> verify -> segment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
