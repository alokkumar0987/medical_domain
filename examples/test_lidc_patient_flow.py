"""
Test flow for one patient (LIDC-IDRI style): report -> NLP -> bbox -> human verification -> output before segmentation.

Uses a lung-nodule report description and shows:
  - Step 1: Clinical intent (NLP) JSON
  - Step 2: Anatomical grounding bbox JSON (example; real bbox requires TotalSegmentator + NIfTI)
  - Step 3: Human-in-the-loop: confirmed point JSON
  - GET /verification: full output before sending to segmentation model

Patient: LIDC-IDRI-0001 (manifest-1585232716547)
Run from project root:  python -m examples.test_lidc_patient_flow
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


# ----- 1. Test report description (LIDC-IDRI lung nodule style) -----
LIDC_TEST_REPORT = (
    "CT chest: A 12 mm solid nodule is noted in the right upper lobe, "
    "contour smooth. No significant mediastinal lymphadenopathy."
)

# For abdomen/kidney (your main pipeline) you can use:
KIDNEY_TEST_REPORT = (
    "A 4.5 cm hypo-attenuating mass is noted in the superior pole of the right kidney."
)


def main():
    print("=" * 60)
    print("PATIENT: LIDC-IDRI-0001 (manifest-1585232716547)")
    print("=" * 60)

    # ----- Step 1: Clinical intent (NLP) -----
    print("\n--- Step 1: Clinical intent extraction (NLP) ---")
    print("Input report:")
    print(f"  {LIDC_TEST_REPORT}")
    intent = extract_clinical_intent(LIDC_TEST_REPORT)
    print("Output JSON (structured organ/region/finding):")
    print(json.dumps(intent, indent=2, ensure_ascii=False))

    # ----- Step 2: Anatomical grounding (example bbox) -----
    print("\n--- Step 2: Anatomical grounding (TotalSegmentator) ---")
    print("Given organ from NLP, TotalSegmentator returns a rough 3D bounding box.")
    print("Example output JSON (voxel coordinates):")
    example_bbox = {
        "x": 120,
        "y": 200,
        "z": 80,
        "width": 50,
        "height": 60,
        "depth": 40,
    }
    print(json.dumps(example_bbox, indent=2))
    print("(Real values come from POST /cases/{case_id}/ground after scan upload.)")

    # ----- Step 3: Human-in-the-loop verification -----
    print("\n--- Step 3: Human-in-the-loop (verification) ---")
    print("The doctor sees the CT slice with the bbox, then clicks the lesion center.")
    print("That click is sent as the confirmed point. Example request body:")
    confirmed_point_json = {"x": 130, "y": 210, "z": 85}
    print("  POST /cases/{case_id}/confirm-point")
    print("  Body:", json.dumps(confirmed_point_json, indent=2))
    print("Example response:")
    print(json.dumps({
        "case_id": "<case_id>",
        "message": "Verified point stored. Safe to call /segment.",
        "confirmed_point_voxel": confirmed_point_json,
    }, indent=2))

    # ----- Output before segmentation -----
    print("\n--- Output before sending to segmentation model ---")
    print("  GET /cases/{case_id}/verification")
    verification_output = {
        "case_id": "<case_id>",
        "bbox_voxel": example_bbox,
        "confirmed_point_voxel": confirmed_point_json,
        "verification_status": "verified",
    }
    print("Response (this is the JSON of coordinates + status):")
    print(json.dumps(verification_output, indent=2))
    print("\nThis is what the pipeline has after human verification:")
    print("  - bbox_voxel: rough organ/region box from Step 2")
    print("  - confirmed_point_voxel: doctor-approved (x, y, z) for the lesion")
    print("  - verification_status: verified -> safe to call POST /segment")

    # ----- Coordinate JSON summary -----
    print("\n--- Coordinate JSON sent to segmentation model ---")
    print("Either pass the same point in the request body:")
    print("  POST /cases/{case_id}/segment")
    print("  Body:", json.dumps({"x": 130, "y": 210, "z": 85}))
    print("Or omit body to use the stored confirmed_point_voxel from verification.")

    print("\n" + "=" * 60)
    print("Done. Run the API and upload a scan to exercise grounding + confirm-point + segment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
