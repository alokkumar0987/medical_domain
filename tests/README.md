# Tests for the Clinical 3D Pipeline

This folder contains tests for the **final project** described in [DATA_FLOW.md](../docs/DATA_FLOW.md).

## Pipeline (DATA_FLOW.md)

| Step | Description        | API / code                    | Test(s) that cover it                    |
|------|--------------------|-------------------------------|------------------------------------------|
| **1** | Create case        | `POST /cases` → ingest + NLP  | `test_pipeline_api`, `test_e2e_pipeline` |
| **2** | Ground             | `POST /cases/{id}/ground`     | `test_e2e_pipeline` (optional)           |
| **3** | Human verification | `GET volume_slice`, `POST confirm-point` | `test_pipeline_api`, `test_e2e_pipeline` |
| **4** | Segment            | `POST /cases/{id}/segment`    | `test_pipeline_api`, `test_e2e_pipeline` |
| **5** | Analyze            | `POST /cases/{id}/analyze`    | `test_pipeline_api`, `test_e2e_pipeline` |

**Data stores** (see DATA_FLOW §2): `data/uploads/`, `data/nifti/`, `data/cases/`, `data/masks/`, `data/meshes/`. The same `case_id` ties NIfTI, case JSON, mask, and mesh together.

## Test files

| File | Purpose |
|------|--------|
| **test_pipeline_api.py** | API contract tests with FastAPI `TestClient` (no server). Covers Steps 1–5: create case, confirm-point, volume_slice, segment, analyze. |
| **test_e2e_pipeline.py** | End-to-end script against a **running** server (`http://127.0.0.1:8000`). Run after `uvicorn main:app --port 8000`. |
| **test_clinical_intent.py** | Step 1 NLP: `extract_clinical_intent(report_text)`. Needs `OPENROUTER_API_KEY` (or other provider) in env. |
| **verify_lidc_idri_0001.py** | Checks that `test_data/LIDC-IDRI-0001` DICOMs match the patient Excel digest (not a pipeline test). |
| **read_csv.py** | Demo script: load test_data DICOMs + Excel, show volume and slice viewer. |

## How to run

From project root:

```bash
# API tests (no server, uses temp data)
# With pytest:
python -m pytest tests/test_pipeline_api.py -v
# Without pytest (script mode):
python -m tests.test_pipeline_api

# Or run the script-style e2e (requires server)
uvicorn main:app --host 127.0.0.1 --port 8000
python -m tests.test_e2e_pipeline

# Clinical intent (needs API key)
python -m tests.test_clinical_intent
```

## Notes

- **Voxel vs world**: Display and `pixel_to_voxel` must use the same `radiological_display` flag so the confirmed point matches the slice (DATA_FLOW §2b).
- **case_id**: All steps use the same `case_id` returned from `POST /cases`.
