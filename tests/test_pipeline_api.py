"""
API tests for the Clinical 3D Pipeline (DATA_FLOW.md).

Uses FastAPI TestClient — no live server. Covers Steps 1–5:
  Create case → Confirm point → Volume slice → Segment → Analyze.

Runs with: python -m pytest tests/test_pipeline_api.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

# Project root on path before importing app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use a temp DATA_ROOT so tests don't touch real data
_TEST_DATA_ROOT = Path(os.getenv("TEST_DATA_ROOT", tempfile.mkdtemp(prefix="pipeline_test_")))


def _make_mini_nifti_bytes() -> bytes:
    """Minimal 3D NIfTI (20,20,20) with a small blob for mask/volume."""
    shape = (20, 20, 20)
    data = np.zeros(shape, dtype=np.float32)
    data[8:13, 8:13, 8:13] = 100.0
    img = nib.Nifti1Image(data, np.eye(4))
    path = Path(tempfile.gettempdir()) / f"_pipeline_test_{os.getpid()}.nii.gz"
    try:
        nib.save(img, str(path))
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


# Minimal report for NLP (may be mocked or need API key in e2e)
REPORT_TEXT = "A 4.5 cm hypo-attenuating mass in the superior pole of the right kidney."
MINI_NIFTI_BYTES = _make_mini_nifti_bytes()

if pytest is not None:

    @pytest.fixture(scope="module")
    def data_root(tmp_path_factory):
        root = tmp_path_factory.mktemp("data")
        (root / "nifti").mkdir()
        (root / "uploads").mkdir()
        (root / "cases").mkdir()
        (root / "masks").mkdir()
        (root / "meshes").mkdir()
        return root

    @pytest.fixture(scope="module")
    def app_with_data_root(data_root):
        import imaging.ingest as ingest
        import data.case_store as case_store
        import segmentation.medsam2_runner as medsam2
        import analysis.mesh as mesh_mod
        orig_ingest_root, orig_upload, orig_nifti = ingest.DATA_ROOT, ingest.UPLOAD_ROOT, ingest.NIFTI_ROOT
        orig_cases = case_store.CASES_DIR
        orig_mask, orig_nifti_m = medsam2.MASK_ROOT, medsam2.NIFTI_ROOT
        orig_mesh_root, orig_mask_m = mesh_mod.MESH_ROOT, getattr(mesh_mod, "MASK_ROOT", None)
        root = Path(data_root)
        ingest.DATA_ROOT, ingest.UPLOAD_ROOT, ingest.NIFTI_ROOT = root, root / "uploads", root / "nifti"
        case_store.CASES_DIR = root / "cases"
        medsam2.MASK_ROOT, medsam2.NIFTI_ROOT = root / "masks", root / "nifti"
        mesh_mod.MESH_ROOT = root / "meshes"
        if hasattr(mesh_mod, "MASK_ROOT"):
            mesh_mod.MASK_ROOT = root / "masks"
        try:
            from main import app
            yield app
        finally:
            ingest.DATA_ROOT, ingest.UPLOAD_ROOT, ingest.NIFTI_ROOT = orig_ingest_root, orig_upload, orig_nifti
            case_store.CASES_DIR, medsam2.MASK_ROOT, medsam2.NIFTI_ROOT = orig_cases, orig_mask, orig_nifti_m
            mesh_mod.MESH_ROOT = orig_mesh_root
            if orig_mask_m is not None:
                mesh_mod.MASK_ROOT = orig_mask_m

    @pytest.fixture
    def client(app_with_data_root):
        from fastapi.testclient import TestClient
        return TestClient(app_with_data_root)

    def test_health(client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_create_case(client):
        r = client.post("/cases", files={"scan_file": ("mini.nii.gz", MINI_NIFTI_BYTES, "application/gzip")}, data={"modality": "ct", "report_text": REPORT_TEXT})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "case_id" in body and body["scan"]["shape"] == [20, 20, 20] and "intent" in body

    def test_full_pipeline(client):
        r = client.post("/cases", files={"scan_file": ("mini.nii.gz", MINI_NIFTI_BYTES, "application/gzip")}, data={"modality": "ct", "report_text": REPORT_TEXT})
        assert r.status_code == 200, r.text
        case_id = r.json()["case_id"]
        r = client.post(f"/cases/{case_id}/confirm-point", json={"x": 10, "y": 10, "z": 10})
        assert r.status_code == 200 and r.json()["confirmed_point_voxel"] == {"x": 10, "y": 10, "z": 10}
        r = client.get(f"/cases/{case_id}/volume_slice?axis=z&index=10")
        assert r.status_code == 200 and r.headers.get("content-type", "").startswith("image/")
        r = client.get(f"/cases/{case_id}/verification")
        assert r.status_code == 200 and r.json().get("verification_status") == "verified"
        r = client.post(f"/cases/{case_id}/segment")
        assert r.status_code == 200 and "mask_path" in r.json()
        r = client.post(f"/cases/{case_id}/analyze")
        assert r.status_code == 200
        m = r.json()
        assert "mesh_path" in m and "volume_cm3" in m and "max_diameter_cm" in m and "voxel_spacing_mm" in m

    def test_volume_info(client):
        r = client.post("/cases", files={"scan_file": ("mini.nii.gz", MINI_NIFTI_BYTES, "application/gzip")}, data={"modality": "ct", "report_text": REPORT_TEXT})
        assert r.status_code == 200
        r = client.get(f"/cases/{r.json()['case_id']}/volume_info")
        assert r.status_code == 200 and r.json()["shape"] == [20, 20, 20]

    def test_volume_affine(client):
        r = client.post("/cases", files={"scan_file": ("mini.nii.gz", MINI_NIFTI_BYTES, "application/gzip")}, data={"modality": "ct", "report_text": REPORT_TEXT})
        assert r.status_code == 200
        r = client.get(f"/cases/{r.json()['case_id']}/volume_affine")
        assert r.status_code == 200 and len(r.json()["affine"]) == 4 and len(r.json()["affine"][0]) == 4

    def test_segment_requires_prompt(client):
        r = client.post("/cases", files={"scan_file": ("mini.nii.gz", MINI_NIFTI_BYTES, "application/gzip")}, data={"modality": "ct", "report_text": REPORT_TEXT})
        assert r.status_code == 200
        r = client.post(f"/cases/{r.json()['case_id']}/segment")
        assert r.status_code == 400


# --- Run without pytest: python -m tests.test_pipeline_api ---
def _run_without_pytest():
    """Single full-pipeline run using TestClient (no pytest)."""
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    for d in ("nifti", "uploads", "cases", "masks", "meshes"):
        (root / d).mkdir(parents=True)

    import imaging.ingest as ingest
    import data.case_store as case_store
    import segmentation.medsam2_runner as medsam2
    import analysis.mesh as mesh_mod

    orig_ingest = (ingest.DATA_ROOT, ingest.UPLOAD_ROOT, ingest.NIFTI_ROOT)
    orig_cases = case_store.CASES_DIR
    orig_medsam2 = (medsam2.MASK_ROOT, medsam2.NIFTI_ROOT)
    orig_mesh = (mesh_mod.MESH_ROOT, getattr(mesh_mod, "MASK_ROOT", None))

    ingest.DATA_ROOT, ingest.UPLOAD_ROOT, ingest.NIFTI_ROOT = root, root / "uploads", root / "nifti"
    case_store.CASES_DIR = root / "cases"
    medsam2.MASK_ROOT, medsam2.NIFTI_ROOT = root / "masks", root / "nifti"
    mesh_mod.MESH_ROOT = root / "meshes"
    if hasattr(mesh_mod, "MASK_ROOT"):
        mesh_mod.MASK_ROOT = root / "masks"

    try:
        from main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        # Full pipeline
        r = client.post("/cases", files={"scan_file": ("mini.nii.gz", MINI_NIFTI_BYTES, "application/gzip")}, data={"modality": "ct", "report_text": REPORT_TEXT})
        if r.status_code != 200:
            print(f"[FAIL] POST /cases: {r.status_code} - {r.text[:500]}")
            raise SystemExit(1)
        case_id = r.json()["case_id"]
        r = client.post(f"/cases/{case_id}/confirm-point", json={"x": 10, "y": 10, "z": 10})
        if r.status_code != 200:
            print(f"[FAIL] confirm-point: {r.status_code} - {r.text[:500]}")
            raise SystemExit(1)
        r = client.get(f"/cases/{case_id}/volume_slice?axis=z&index=10")
        if r.status_code != 200:
            print(f"[FAIL] volume_slice: {r.status_code} - {r.text[:500]}")
            raise SystemExit(1)
        r = client.post(f"/cases/{case_id}/segment")
        if r.status_code != 200:
            print(f"[FAIL] segment: {r.status_code} - {r.text[:500]}")
            raise SystemExit(1)
        r = client.post(f"/cases/{case_id}/analyze")
        if r.status_code != 200:
            print(f"[FAIL] analyze: {r.status_code} - {r.text[:500]}")
            raise SystemExit(1)
        print("[OK] test_pipeline_api full pipeline (TestClient)")
    finally:
        ingest.DATA_ROOT, ingest.UPLOAD_ROOT, ingest.NIFTI_ROOT = orig_ingest
        case_store.CASES_DIR = orig_cases
        medsam2.MASK_ROOT, medsam2.NIFTI_ROOT = orig_medsam2
        mesh_mod.MESH_ROOT = orig_mesh[0]
        if orig_mesh[1] is not None:
            mesh_mod.MASK_ROOT = orig_mesh[1]


if __name__ == "__main__":
    _run_without_pytest()
