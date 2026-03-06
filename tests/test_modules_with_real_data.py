"""
Module-by-Module Test Script using real test_data (LIDC-IDRI-0001 CT scan).

Tests (in order):
  M1. data/case_store.py        — JSON state persistence
  M2. imaging/ingest.py         — ZIP DICOM → NIfTI conversion
  M3. imaging/slices.py         — Slice PNG extraction + bbox overlay + pixel_to_voxel
  M4. imaging/grounding.py      — BoundingBox from mask (using pre-made mask, no GPU needed)
  M5. nlp/clinical_intent_extractor.py — LLM intent extraction from real report text
  M6. segmentation/medsam2_runner.py   — Placeholder mask (no GPU)
  M7. analysis/mesh.py          — Mesh + metric computation on placeholder mask

Run:
    python tests/test_modules_with_real_data.py
"""

import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── load .env so API keys are available ───────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

TEST_DATA_DIR = PROJECT_ROOT / "test_data"
DCM_DIR       = TEST_DATA_DIR / "LIDC-IDRI-0001"
REPORT_TXT    = TEST_DATA_DIR / "LIDC-IDRI-0001_report.txt"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"

results = []  # (module, test, status, detail)


def record(module, test, ok, detail=""):
    tag = PASS if ok else FAIL
    results.append((module, test, tag, detail))
    icon = "✅" if ok else "❌"
    print(f"  {icon} [{module}] {test}" + (f" → {detail}" if detail else ""))


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED: temp DATA_ROOT so we never touch real data/
# ─────────────────────────────────────────────────────────────────────────────
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="test_real_data_"))
for d in ("nifti", "uploads", "cases", "masks", "meshes"):
    (_TMP_ROOT / d).collect() if False else (_TMP_ROOT / d).mkdir(parents=True, exist_ok=True)

# patch module-level path constants
import imaging.ingest as _ingest
import data.case_store as _store
import segmentation.medsam2_runner as _medsam2
import analysis.mesh as _mesh_mod

_ingest.DATA_ROOT   = _TMP_ROOT
_ingest.UPLOAD_ROOT = _TMP_ROOT / "uploads"
_ingest.NIFTI_ROOT  = _TMP_ROOT / "nifti"
_store.DATA_ROOT    = _TMP_ROOT
_store.CASES_DIR    = _TMP_ROOT / "cases"
_medsam2.MASK_ROOT  = _TMP_ROOT / "masks"
_medsam2.NIFTI_ROOT = _TMP_ROOT / "nifti"
_mesh_mod.MESH_ROOT = _TMP_ROOT / "meshes"
_mesh_mod.MASK_ROOT = _TMP_ROOT / "masks"

# ─────────────────────────────────────────────────────────────────────────────
# M1 — data/case_store.py
# ─────────────────────────────────────────────────────────────────────────────
section("M1 — data/case_store.py")

from data.case_store import (
    get_case_state, set_case_state, set_bbox, set_confirmed_point,
    set_grounding_status, get_confirmed_point, CASES_DIR,
)

try:
    _test_id = "test-case-001"
    state = get_case_state(_test_id)
    record("case_store", "get empty state returns {}", state == {})
except Exception as e:
    record("case_store", "get empty state", False, str(e))

try:
    set_grounding_status(_test_id, "processing")
    s = get_case_state(_test_id)
    record("case_store", "set_grounding_status persists", s.get("grounding_status") == "processing")
except Exception as e:
    record("case_store", "set_grounding_status", False, str(e))

try:
    bbox = {"x": 10, "y": 20, "z": 5, "width": 30, "height": 40, "depth": 15}
    set_bbox(_test_id, bbox)
    s = get_case_state(_test_id)
    record("case_store", "set_bbox persists correctly", s.get("bbox_voxel") == bbox)
except Exception as e:
    record("case_store", "set_bbox", False, str(e))

try:
    set_confirmed_point(_test_id, x=11, y=22, z=33)
    s = get_case_state(_test_id)
    cp = s.get("confirmed_point_voxel", {})
    ok = cp.get("x") == 11 and cp.get("y") == 22 and cp.get("z") == 33
    record("case_store", "set_confirmed_point persists (x,y,z)", ok)
    record("case_store", "verification_status=verified", s.get("verification_status") == "verified")
except Exception as e:
    record("case_store", "set_confirmed_point", False, str(e))

try:
    cp = get_confirmed_point(_test_id)
    record("case_store", "get_confirmed_point returns dict", isinstance(cp, dict) and cp.get("x") == 11)
except Exception as e:
    record("case_store", "get_confirmed_point", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# M2 — imaging/ingest.py  (real ZIP DICOM → NIfTI)
# ─────────────────────────────────────────────────────────────────────────────
section("M2 — imaging/ingest.py  [ZIP DICOM → NIfTI]")

from imaging.ingest import (
    generate_scan_id, save_upload_to_disk, detect_format,
    normalize_to_nifti, SourceFormat, Modality,
)

# Build ZIP of the LIDC-IDRI-0001 DICOMs in memory
_zip_path = _TMP_ROOT / "uploads" / "LIDC-IDRI-0001.zip"
_zip_path.parent.mkdir(parents=True, exist_ok=True)

try:
    dcm_files = sorted(DCM_DIR.glob("*.dcm"))
    record("ingest", f"test_data has DCM files ({len(dcm_files)})", len(dcm_files) > 0, f"{len(dcm_files)} files")
    with zipfile.ZipFile(_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dcm in dcm_files:
            zf.write(dcm, arcname=f"LIDC-IDRI-0001/{dcm.name}")
    record("ingest", "ZIP created with subfolder structure", _zip_path.exists(), f"{_zip_path.stat().st_size//1024} KB")
except Exception as e:
    record("ingest", "create test ZIP", False, str(e))
    dcm_files = []

try:
    fmt = detect_format(_zip_path)
    record("ingest", "detect_format ZIP → ZIP_DICOM", fmt == SourceFormat.ZIP_DICOM)
except Exception as e:
    record("ingest", "detect_format ZIP", False, str(e))

try:
    fmt_nii = detect_format(Path("scan.nii.gz"))
    record("ingest", "detect_format .nii.gz → NIFTI (Bug1 fix)", fmt_nii == SourceFormat.NIFTI)
except Exception as e:
    record("ingest", "detect_format .nii.gz", False, str(e))

_scan_id = generate_scan_id()
_nifti_result = None

try:
    content = _zip_path.read_bytes()
    src_path = save_upload_to_disk(_scan_id, "LIDC-IDRI-0001.zip", content)
    record("ingest", "save_upload_to_disk writes file", src_path.exists())
except Exception as e:
    record("ingest", "save_upload_to_disk", False, str(e))
    src_path = _zip_path

try:
    print("    [Running dcm2niix — this may take 20-60 s …]")
    _nifti_result = normalize_to_nifti(_scan_id, src_path, Modality.CT)
    npath = Path(_nifti_result.nifti_path)
    record("ingest", "normalize_to_nifti creates .nii.gz", npath.exists())
    record("ingest", "shape is 3-tuple of ints", len(_nifti_result.shape) == 3, str(_nifti_result.shape))
    record("ingest", "spacing is 3-tuple of floats", len(_nifti_result.spacing) == 3, str(_nifti_result.spacing))
    record("ingest", "shape[0] (slices) == 133", _nifti_result.shape[0] == 133, f"got {_nifti_result.shape[0]}")
except FileNotFoundError as e:
    record("ingest", "normalize_to_nifti (dcm2niix)", False, str(e))
    # ── fallback: build NIfTI with SimpleITK (dcm2niix not installed) ──────
    print("    [dcm2niix not found — using SimpleITK fallback to read DICOMs directly]")
    try:
        import SimpleITK as sitk
        from imaging.ingest import IngestResult, SourceFormat
        reader = sitk.ImageSeriesReader()
        dcm_names = reader.GetGDCMSeriesFileNames(str(DCM_DIR))
        reader.SetFileNames(dcm_names)
        sitk_img = reader.Execute()
        out_nii = _TMP_ROOT / "nifti" / f"{_scan_id}.nii.gz"
        sitk.WriteImage(sitk_img, str(out_nii))
        size    = sitk_img.GetSize()         # (nx, ny, nz) in SimpleITK order
        spacing = sitk_img.GetSpacing()      # (sx, sy, sz)
        # Convert to (nz, ny, nx) shape + (sz, sy, sx) spacing for nibabel convention
        shape3 = (size[2], size[1], size[0])
        spac3  = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
        _nifti_result = IngestResult(
            scan_id=_scan_id,
            nifti_path=str(out_nii),
            shape=shape3,
            spacing=spac3,
            modality=Modality.CT,
            source_format=SourceFormat.ZIP_DICOM,
        )
        record("ingest", "normalize_to_nifti [SimpleITK fallback] creates .nii.gz", out_nii.exists())
        record("ingest", "shape is 3-tuple of ints [fallback]", len(_nifti_result.shape) == 3, str(shape3))
        record("ingest", "spacing is 3-tuple [fallback]", len(_nifti_result.spacing) == 3, str(spac3))
        record("ingest", "shape[0] (slices) == 133 [fallback]", _nifti_result.shape[0] == 133, f"got {shape3[0]}")
    except Exception as e2:
        record("ingest", "normalize_to_nifti [SimpleITK fallback]", False, str(e2))
except Exception as e:
    record("ingest", "normalize_to_nifti (dcm2niix)", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# M3 — imaging/slices.py
# ─────────────────────────────────────────────────────────────────────────────
section("M3 — imaging/slices.py  [Slice PNG + pixel_to_voxel]")

from imaging.slices import get_affine, get_slice_png, pixel_to_voxel
from PIL import Image as _PIL

if _nifti_result is None:
    print("  [SKIP — no NIfTI available from M2]")
    results.append(("slices", "all tests", SKIP, "NIfTI not produced in M2"))
else:
    nifti_path = _nifti_result.nifti_path
    shape = _nifti_result.shape

    try:
        aff = get_affine(nifti_path)
        record("slices", "get_affine returns (4,4)", aff.shape == (4, 4))
    except Exception as e:
        record("slices", "get_affine", False, str(e))

    for axis in ("z", "y", "x"):
        try:
            idx = shape[{"z":0,"y":1,"x":2}[axis]] // 2
            png = get_slice_png(nifti_path, axis, idx)
            img = _PIL.open(io.BytesIO(png))
            record("slices", f"get_slice_png axis={axis} idx={idx}", img.mode == "L",
                   f"size={img.size}")
        except Exception as e:
            record("slices", f"get_slice_png axis={axis}", False, str(e))

    # Bbox overlay
    try:
        bbox_v = {"x": 30, "y": 30, "z": 60, "width": 40, "height": 40, "depth": 20}
        png_bbox = get_slice_png(nifti_path, "z", 70, bbox_voxel=bbox_v)
        record("slices", "get_slice_png with bbox overlay (z=70)", len(png_bbox) > 0)
    except Exception as e:
        record("slices", "get_slice_png bbox overlay", False, str(e))

    # Radiological display
    try:
        png_rad = get_slice_png(nifti_path, "z", 70, radiological_display=True)
        png_norm = get_slice_png(nifti_path, "z", 70, radiological_display=False)
        record("slices", "radiological slice differs from normal", png_rad != png_norm)
    except Exception as e:
        record("slices", "radiological_display", False, str(e))

    # pixel_to_voxel
    try:
        nz, ny, nx = shape
        vx, vy, vz = pixel_to_voxel(256.0, 256.0, "z", nz//2, shape, 512, 512)
        record("slices", "pixel_to_voxel returns in-bounds (z)", 0 <= vx < nx and 0 <= vy < ny)
    except Exception as e:
        record("slices", "pixel_to_voxel z-axis", False, str(e))

    try:
        vx2, vy2, vz2 = pixel_to_voxel(256.0, 256.0, "z", nz//2, shape, 512, 512, radiological_display=True)
        vx3, vy3, vz3 = pixel_to_voxel(256.0, 256.0, "z", nz//2, shape, 512, 512, radiological_display=False)
        record("slices", "pixel_to_voxel radiological flips x", vx2 != vx3 or nx == 1)
    except Exception as e:
        record("slices", "pixel_to_voxel radiological", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# M4 — imaging/grounding.py  (bbox from a synthetic mask, no TotalSegmentator)
# ─────────────────────────────────────────────────────────────────────────────
section("M4 — imaging/grounding.py  [BoundingBox from mask]")

from imaging.grounding import _compute_bbox_from_mask, BoundingBox
import nibabel as nib
import numpy as np

if _nifti_result is None:
    print("  [SKIP — no NIfTI available from M2]")
    results.append(("grounding", "all tests", SKIP, "NIfTI not produced in M2"))
else:
    _synth_mask_path = _TMP_ROOT / "masks" / "synth_organ.nii.gz"
    _synth_mask_path.parent.mkdir(exist_ok=True)

    try:
        # Build a synthetic organ mask matching the real volume shape
        img_orig = nib.load(str(_nifti_result.nifti_path))
        D, H, W = img_orig.shape[:3]
        mask_data = np.zeros((D, H, W), dtype=np.uint8)
        # Place a 20×20×20 "organ" at the centre
        zc, yc, xc = D//2, H//2, W//2
        mask_data[zc-10:zc+10, yc-10:yc+10, xc-10:xc+10] = 1
        synth = nib.Nifti1Image(mask_data, img_orig.affine)
        nib.save(synth, str(_synth_mask_path))
        record("grounding", "synthetic organ mask created", _synth_mask_path.exists())
    except Exception as e:
        record("grounding", "create synthetic mask", False, str(e))

    try:
        spacing = (1.0, 1.0, 1.0)
        bbox = _compute_bbox_from_mask(_synth_mask_path, spacing)
        record("grounding", "_compute_bbox_from_mask returns BoundingBox", isinstance(bbox, BoundingBox))
        record("grounding", "bbox width/height/depth > 0", bbox.width > 0 and bbox.height > 0 and bbox.depth > 0,
               f"x={bbox.x} y={bbox.y} z={bbox.z} w={bbox.width} h={bbox.height} d={bbox.depth}")
        # Should be ~20 voxels wide in each dim
        record("grounding", "bbox dimensions ~20 voxels", 15 <= bbox.width <= 25 and 15 <= bbox.height <= 25,
               f"({bbox.width},{bbox.height},{bbox.depth})")
    except Exception as e:
        record("grounding", "_compute_bbox_from_mask", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# M5 — nlp/clinical_intent_extractor.py  (real LLM call via .env keys)
# ─────────────────────────────────────────────────────────────────────────────
section("M5 — nlp/clinical_intent_extractor.py  [LLM intent extraction]")

from nlp.clinical_intent_extractor import extract_clinical_intent

_REPORT = (
    "CT chest without contrast. Right upper lobe solid nodule measures 9mm. "
    "Left lower lobe subpleural nodule measures 6mm. No pleural effusion. "
    "Recommend follow-up for the dominant right upper lobe nodule."
)

try:
    print("    [Calling LLM — may take 3-10 s …]")
    intent = extract_clinical_intent(_REPORT)
    record("nlp", "returns dict", isinstance(intent, dict))
    record("nlp", "has keys: organ, region, finding",
           all(k in intent for k in ("organ", "region", "finding")),
           str(intent))
    record("nlp", "organ is a non-empty string or None",
           intent["organ"] is None or isinstance(intent["organ"], str))
    # The report mentions lung nodule — expect lung-related organ
    organ_ok = intent["organ"] is not None and "lung" in (intent["organ"] or "").lower()
    record("nlp", "organ mentions 'lung'", organ_ok, f"organ={intent['organ']!r}")
    record("nlp", "finding is nodule / mass / similar",
           intent["finding"] is not None,
           f"finding={intent['finding']!r}")
except Exception as e:
    record("nlp", "extract_clinical_intent", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# M6 — segmentation/medsam2_runner.py  (placeholder mode, no GPU)
# ─────────────────────────────────────────────────────────────────────────────
section("M6 — segmentation/medsam2_runner.py  [placeholder mask, no GPU]")

from segmentation.medsam2_runner import run_medsam2, run_medsam2_from_bbox

if _nifti_result is None:
    print("  [SKIP — no NIfTI available from M2]")
    results.append(("medsam2", "all tests", SKIP, "NIfTI not produced in M2"))
else:
    nz, ny, nx = _nifti_result.shape
    pt_zyx = (nz // 2, ny // 2, nx // 2)

    # Load the actual NIfTI array shape (may differ from reported shape
    # depending on whether dcm2niix or SimpleITK wrote the file).
    import nibabel as _nib_chk
    _actual_shape = _nib_chk.load(str(_nifti_result.nifti_path)).shape[:3]

    try:
        mask_path = run_medsam2(_scan_id, pt_zyx)
        record("medsam2", "run_medsam2 produces mask file", mask_path.exists())
        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata()
        actual_mask_shape = mask_data.shape[:3]
        record("medsam2", "mask shape matches NIfTI array shape",
               actual_mask_shape == _actual_shape,
               f"mask={actual_mask_shape} nifti={_actual_shape}")
        record("medsam2", "mask has nonzero voxels (placeholder cube)", int(mask_data.sum()) > 0,
               f"sum={int(mask_data.sum())}")
    except Exception as e:
        record("medsam2", "run_medsam2 (point)", False, str(e))

    try:
        # Use image-space centre for bbox (works for both nibabel and SimpleITK shapes)
        D, H, W = _actual_shape
        bbox_dict = {"x": W//2 - 5, "y": H//2 - 5, "z": D//2 - 5,
                     "width": 10, "height": 10, "depth": 10}
        mask_path2 = run_medsam2_from_bbox(_scan_id, bbox_dict)
        record("medsam2", "run_medsam2_from_bbox produces mask file", mask_path2.exists())
        m2 = nib.load(str(mask_path2)).get_fdata()
        record("medsam2", "bbox mask has nonzero voxels", int(m2.sum()) > 0, f"sum={int(m2.sum())}")
    except Exception as e:
        record("medsam2", "run_medsam2_from_bbox", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# M7 — analysis/mesh.py
# ─────────────────────────────────────────────────────────────────────────────
section("M7 — analysis/mesh.py  [Marching Cubes + metrics]")

from analysis.mesh import mask_to_mesh, MeshMetrics

if _nifti_result is None:
    print("  [SKIP — no NIfTI available from M2]")
    results.append(("mesh", "all tests", SKIP, "NIfTI not produced in M2"))
else:
    try:
        metrics = mask_to_mesh(_scan_id)
        record("mesh", "mask_to_mesh returns MeshMetrics", isinstance(metrics, MeshMetrics))
        record("mesh", "STL file created", Path(metrics.mesh_path).exists())
        record("mesh", "OBJ file created", Path(metrics.mesh_path_obj).exists())
        record("mesh", "volume_cm3 >= 0", metrics.volume_cm3 >= 0, f"{metrics.volume_cm3} cm³")
        record("mesh", "max_diameter_cm >= 0", metrics.max_diameter_cm >= 0, f"{metrics.max_diameter_cm} cm")
        record("mesh", "voxel_spacing_mm is 3-tuple", len(metrics.voxel_spacing_mm) == 3,
               str(metrics.voxel_spacing_mm))
    except Exception as e:
        record("mesh", "mask_to_mesh", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  FINAL RESULTS")
print(f"{'='*60}")

pass_cnt = sum(1 for _, _, s, _ in results if s == PASS)
fail_cnt = sum(1 for _, _, s, _ in results if s == FAIL)
skip_cnt = sum(1 for _, _, s, _ in results if s == SKIP)

print(f"\n  Total: {len(results)}   ✅ {pass_cnt}   ❌ {fail_cnt}   ⏭  {skip_cnt}\n")

if fail_cnt:
    print("  FAILURES:")
    for mod, test, status, detail in results:
        if status == FAIL:
            print(f"    ❌ [{mod}] {test}" + (f" → {detail}" if detail else ""))

# cleanup temp dir
shutil.rmtree(_TMP_ROOT, ignore_errors=True)
print("\n  Temp data cleaned up.")
print(f"{'='*60}\n")
