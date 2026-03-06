from pathlib import Path
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from imaging.ingest import IngestResult, Modality, NIFTI_ROOT, generate_scan_id, normalize_to_nifti, save_upload_to_disk
from imaging.slices import get_affine, get_slice_png, pixel_to_voxel
from imaging.grounding import BoundingBox, run_grounding_for_case
from nlp.clinical_intent_extractor import extract_clinical_intent
from segmentation.medsam2_runner import run_medsam2, run_medsam2_from_bbox
from analysis.mesh import MeshMetrics, mask_to_mesh

try:
    from data.case_store import get_case_state, set_bbox, set_confirmed_point, set_grounding_status
except ImportError:
    # Allow running from project root when data is a package
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data.case_store import get_case_state, set_bbox, set_confirmed_point, set_grounding_status


app = FastAPI(title="Clinical 3D Imaging Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NLPRequest(BaseModel):
    report_text: str


@app.post("/upload/scan", response_model=IngestResult)
async def upload_scan(
    modality: Modality = Form(...),
    body_region: Optional[str] = Form(None),  # reserved for future use
    scan_file: UploadFile = File(...),
) -> IngestResult:
    """
    Upload a scan (ZIP DICOM, NIfTI, NRRD, image) and normalize it to NIfTI.
    """
    try:
        content = await scan_file.read()
        scan_id = generate_scan_id()
        source_path = save_upload_to_disk(scan_id, scan_file.filename or "scan", content)
        result = normalize_to_nifti(scan_id, source_path, modality)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to ingest scan: {exc}") from exc


@app.post("/nlp/intent")
async def nlp_intent(req: NLPRequest):
    """
    Extract clinical intent (organ, region, finding) from a radiology report.
    """
    try:
        intent = extract_clinical_intent(req.report_text)
        return intent
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract clinical intent: {exc}") from exc


# Combined endpoint: create a case by uploading scan + report in one call.
class CaseCreateResponse(BaseModel):
    case_id: str
    scan: IngestResult
    intent: dict


@app.post("/cases", response_model=CaseCreateResponse)
async def create_case(
    modality: Modality = Form(...),
    report_text: str = Form(...),
    body_region: Optional[str] = Form(None),
    scan_file: UploadFile = File(...),
):
    """
    Create a case by uploading a scan and providing the report text.

    Returns a case_id that can be used for later steps (grounding, segmentation, etc.).
    For now, case_id is the same as scan_id; a real system would persist this in a DB.
    """
    try:
        content = await scan_file.read()
        scan_id = generate_scan_id()
        source_path = save_upload_to_disk(scan_id, scan_file.filename or "scan", content)
        ingest = normalize_to_nifti(scan_id, source_path, modality)

        intent = extract_clinical_intent(report_text)

        return CaseCreateResponse(case_id=scan_id, scan=ingest, intent=intent)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to create case: {exc}") from exc


@app.post("/cases/{case_id}/ground", response_model=BoundingBox)
async def ground_case(case_id: str, organ: Optional[str] = None):
    """
    Run anatomical grounding with TotalSegmentator and return a bounding box
    for the given organ (default kidney_right). Persists bbox for human verification.
    Sets grounding_status to 'processing' while running so the UI can show a spinner;
    then 'completed' or 'failed'.
    """
    organ_label = organ or "kidney_right"
    set_grounding_status(case_id, "processing")
    try:
        bbox = run_grounding_for_case(case_id, organ_label=organ_label)
        set_bbox(case_id, bbox.model_dump())
        set_grounding_status(case_id, "completed")
        return bbox
    except FileNotFoundError as exc:
        set_grounding_status(case_id, "failed")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        set_grounding_status(case_id, "failed")
        raise HTTPException(status_code=400, detail=f"Failed to ground case: {exc}") from exc


class ConfirmPointRequest(BaseModel):
    x: int
    y: int
    z: int


class PixelToVoxelRequest(BaseModel):
    px: float
    py: float
    axis: str
    index: int
    img_width: int
    img_height: int
    radiological_display: bool = False


@app.post("/cases/{case_id}/confirm-point")
async def confirm_point(case_id: str, point: ConfirmPointRequest):
    """
    Human-in-the-loop: store the doctor-confirmed 3D point (voxel indices).
    This is the safety verification step before sending to the segmentation model.
    Returns the stored coordinate JSON.
    """
    set_confirmed_point(case_id, x=point.x, y=point.y, z=point.z)
    return {
        "case_id": case_id,
        "message": "Verified point stored. Safe to call /segment.",
        "confirmed_point_voxel": {"x": point.x, "y": point.y, "z": point.z},
    }


@app.post("/cases/{case_id}/pixel-to-voxel")
async def pixel_to_voxel_endpoint(case_id: str, req: PixelToVoxelRequest):
    """
    Map (px, py) in the displayed slice image to voxel (x, y, z).
    Use the same radiological_display value as for the slice so the point matches the display.
    """
    nifti_path = NIFTI_ROOT / f"{case_id}.nii.gz"
    if not nifti_path.exists():
        raise HTTPException(status_code=404, detail=f"NIfTI not found for case_id={case_id}")
    if req.axis not in ("z", "y", "x"):
        raise HTTPException(status_code=400, detail="axis must be z, y, or x")
    import nibabel as nib
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    shape = tuple(int(x) for x in data.shape[-3:])
    x, y, z = pixel_to_voxel(
        req.px, req.py, req.axis, req.index, shape,
        req.img_width, req.img_height, req.radiological_display,
    )
    return {"x": x, "y": y, "z": z}


@app.get("/cases/{case_id}/volume_info")
async def volume_info(case_id: str):
    """Return volume shape (nz, ny, nx) for slice slider bounds."""
    import nibabel as nib
    nifti_path = NIFTI_ROOT / f"{case_id}.nii.gz"
    if not nifti_path.exists():
        raise HTTPException(status_code=404, detail=f"NIfTI not found for case_id={case_id}")
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    shape = data.shape[-3:]
    return {"shape": [int(shape[0]), int(shape[1]), int(shape[2])]}


@app.get("/cases/{case_id}/volume_affine")
async def volume_affine(case_id: str):
    """
    Return the 4x4 affine matrix from the NIfTI header (voxel → world mm).
    Use for world ↔ voxel conversion or to verify display orientation.
    """
    nifti_path = NIFTI_ROOT / f"{case_id}.nii.gz"
    if not nifti_path.exists():
        raise HTTPException(status_code=404, detail=f"NIfTI not found for case_id={case_id}")
    aff = get_affine(str(nifti_path))
    return {"affine": aff.tolist()}


@app.get("/cases/{case_id}/volume_slice", response_class=Response)
async def volume_slice(
    case_id: str,
    axis: str = "z",
    index: int = 0,
    with_bbox: bool = True,
    radiological_display: bool = False,
):
    """
    Return a 2D slice of the case volume as PNG. Optional bbox overlay from grounding.

    If radiological_display is True, axial (z) slices are flipped so "left" on screen
    is patient left. When using this, the client must use the same flag in
    POST /cases/{id}/pixel-to-voxel so the confirmed point matches the display.
    """
    nifti_path = NIFTI_ROOT / f"{case_id}.nii.gz"
    if not nifti_path.exists():
        raise HTTPException(status_code=404, detail=f"NIfTI not found for case_id={case_id}")
    if axis not in ("z", "y", "x"):
        raise HTTPException(status_code=400, detail="axis must be z, y, or x")
    state = get_case_state(case_id) if with_bbox else {}
    bbox = state.get("bbox_voxel")
    try:
        png_bytes = get_slice_png(
            str(nifti_path), axis, index, bbox_voxel=bbox, radiological_display=radiological_display
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    response = Response(content=png_bytes, media_type="image/png")
    if radiological_display:
        response.headers["X-Radiological-Display"] = "true"
    return response


@app.get("/cases/{case_id}/verification")
async def get_verification(case_id: str):
    """
    Output before segmentation: bbox from grounding and (if done) confirmed point.
    Use this to show the doctor what will be sent to the segmentation model.
    """
    state = get_case_state(case_id)
    return {
        "case_id": case_id,
        "bbox_voxel": state.get("bbox_voxel"),
        "confirmed_point_voxel": state.get("confirmed_point_voxel"),
        "verification_status": state.get("verification_status", "pending_verification"),
    }


@app.post("/cases/{case_id}/segment")
async def segment_case(case_id: str, point: Optional[ConfirmPointRequest] = Body(None)):
    """
    Run MedSAM2 using a prompt from the pipeline (bridge from Step 3):

    - If a point is provided in the body, or a confirmed point exists from HITL, uses that point.
    - Else if a bounding box exists from grounding (TotalSegmentator), uses the 3D bbox as the prompt.
    """
    try:
        if point is not None:
            z, y, x = point.z, point.y, point.x
            mask_path = run_medsam2(case_id, (z, y, x))
            return {"mask_path": str(mask_path)}
        state = get_case_state(case_id)
        cp = state.get("confirmed_point_voxel")
        if cp:
            z, y, x = int(cp["z"]), int(cp["y"]), int(cp["x"])
            mask_path = run_medsam2(case_id, (z, y, x))
            return {"mask_path": str(mask_path)}
        bbox = state.get("bbox_voxel")
        if bbox:
            mask_path = run_medsam2_from_bbox(case_id, bbox)
            return {"mask_path": str(mask_path)}
        raise HTTPException(
            status_code=400,
            detail="No prompt. Call POST /cases/{id}/confirm-point (HITL) or POST /cases/{id}/ground first (bbox).",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to segment case: {exc}") from exc


@app.post("/cases/{case_id}/analyze", response_model=MeshMetrics)
async def analyze_case(case_id: str):
    """
    Convert the lesion mask to a mesh and compute volume / diameter metrics.
    """
    try:
        metrics = mask_to_mesh(case_id)
        return metrics
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to analyze case: {exc}") from exc


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

