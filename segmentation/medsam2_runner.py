"""
MedSAM2 segmentation runner.

Supports two prompt modes:
- Point prompt: NIfTI + user-confirmed 3D point (z, y, x) from HITL.
- Box prompt (bridge from TotalSegmentator): NIfTI + 3D bbox (x, y, z, width, height, depth).

Uses the MedSAM2 repo (infer_single_nifti_point.py) when MEDSAM2_ROOT is set and a
checkpoint is present; otherwise falls back to a placeholder mask for pipeline verification.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Union

import nibabel as nib
import numpy as np

from imaging.ingest import DATA_ROOT, NIFTI_ROOT

_log = logging.getLogger(__name__)


# MedSAM2 repo root (clone from https://github.com/bowang-lab/MedSAM2)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDSAM2_ROOT = Path(os.getenv("MEDSAM2_ROOT", _PROJECT_ROOT / "MedSAM2"))
MASK_ROOT = Path(os.getenv("MASK_ROOT", str(DATA_ROOT / "masks")))


def _ensure_dirs() -> None:
    MASK_ROOT.mkdir(parents=True, exist_ok=True)


def _medsam2_available() -> bool:
    """True if MedSAM2 repo has the wrapper script and at least one checkpoint."""
    wrapper = MEDSAM2_ROOT / "infer_single_nifti_point.py"
    if not wrapper.is_file():
        return False
    ckpt_dir = MEDSAM2_ROOT / "checkpoints"
    for name in ("MedSAM2_latest.pt", "MedSAM2_CTLesion.pt"):
        if (ckpt_dir / name).is_file():
            return True
    return False


def _run_medsam2_subprocess(nifti_path: Path, point_zyx: Tuple[int, int, int], out_path: Path) -> bool:
    """Run MedSAM2 infer_single_nifti_point.py with point prompt; return True on success."""
    wrapper = MEDSAM2_ROOT / "infer_single_nifti_point.py"
    z, y, x = point_zyx
    cmd = [
        sys.executable,
        str(wrapper),
        "--nifti", str(nifti_path),
        "--point", str(z), str(y), str(x),
        "--output", str(out_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(MEDSAM2_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            # Bug 8 fix: log stderr/stdout so failures are visible in app logs.
            _log.error(
                "MedSAM2 subprocess (point) failed (rc=%d).\nSTDOUT: %s\nSTDERR: %s",
                result.returncode,
                result.stdout[:2000],
                result.stderr[:2000],
            )
            return False
        return out_path.is_file()
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as exc:
        _log.error("MedSAM2 subprocess (point) raised %s: %s", type(exc).__name__, exc)
        return False


def _run_medsam2_subprocess_bbox(
    nifti_path: Path,
    bbox: Union[dict, object],
    out_path: Path,
) -> bool:
    """Run MedSAM2 infer_single_nifti_point.py with 3D bbox (TotalSegmentator bridge); return True on success."""
    wrapper = MEDSAM2_ROOT / "infer_single_nifti_point.py"
    if isinstance(bbox, dict):
        x, y, z = int(bbox["x"]), int(bbox["y"]), int(bbox["z"])
        w, h, d = int(bbox["width"]), int(bbox["height"]), int(bbox["depth"])
    else:
        x, y, z = int(bbox.x), int(bbox.y), int(bbox.z)
        w, h, d = int(bbox.width), int(bbox.height), int(bbox.depth)
    cmd = [
        sys.executable,
        str(wrapper),
        "--nifti", str(nifti_path),
        "--bbox", str(x), str(y), str(z), str(w), str(h), str(d),
        "--output", str(out_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(MEDSAM2_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            # Bug 8 fix: surface stderr/stdout so bbox failures are visible.
            _log.error(
                "MedSAM2 subprocess (bbox) failed (rc=%d).\nSTDOUT: %s\nSTDERR: %s",
                result.returncode,
                result.stdout[:2000],
                result.stderr[:2000],
            )
            return False
        return out_path.is_file()
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as exc:
        _log.error("MedSAM2 subprocess (bbox) raised %s: %s", type(exc).__name__, exc)
        return False


def _placeholder_mask(nifti_path: Path, point_zyx: Tuple[int, int, int], out_path: Path) -> Path:
    """Write a small cube mask around the point (for pipeline verification without GPU)."""
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    zc, yc, xc = point_zyx
    radius = 3
    mask = np.zeros_like(data, dtype=np.uint8)
    # data.shape is typically (X, Y, Z). Let's use dim0=X, dim1=Y, dim2=Z.
    x_min = max(0, xc - radius)
    x_max = min(data.shape[0] - 1, xc + radius)
    y_min = max(0, yc - radius)
    y_max = min(data.shape[1] - 1, yc + radius)
    z_min = max(0, zc - radius)
    z_max = min(data.shape[2] - 1, zc + radius)
    mask[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1] = 1
    out_img = nib.Nifti1Image(mask, affine=img.affine)
    nib.save(out_img, str(out_path))
    return out_path


def _placeholder_mask_from_bbox(nifti_path: Path, bbox: Union[dict, object], out_path: Path) -> Path:
    """Write a mask that is 1 inside the 3D bbox (for pipeline verification without GPU)."""
    if isinstance(bbox, dict):
        x, y, z = int(bbox["x"]), int(bbox["y"]), int(bbox["z"])
        w, h, d = int(bbox["width"]), int(bbox["height"]), int(bbox["depth"])
    else:
        x, y, z = int(bbox.x), int(bbox.y), int(bbox.z)
        w, h, d = int(bbox.width), int(bbox.height), int(bbox.depth)
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    D, H, W = data.shape
    mask = np.zeros_like(data, dtype=np.uint8)
    z_min = max(0, z)
    z_max = min(D - 1, z + d - 1)
    y_min = max(0, y)
    y_max = min(H - 1, y + h - 1)
    x_min = max(0, x)
    x_max = min(W - 1, x + w - 1)
    mask[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = 1
    out_img = nib.Nifti1Image(mask, affine=img.affine)
    nib.save(out_img, str(out_path))
    return out_path


def run_medsam2(case_id: str, point_zyx: Tuple[int, int, int]) -> Path:
    """
    Run MedSAM2 for the given case and seed point.

    Parameters
    ----------
    case_id:
        Identifier that maps to the NIfTI volume {case_id}.nii.gz.
    point_zyx:
        Seed point in voxel indices (z, y, x) as confirmed by the user.

    Returns
    -------
    Path
        Path to the generated 3D binary mask (.nii.gz) under MASK_ROOT.

    Notes
    -----
    If MEDSAM2_ROOT points to a MedSAM2 clone with checkpoints and
    infer_single_nifti_point.py, runs real inference. Otherwise uses a
    placeholder mask so the pipeline can be verified without GPU.
    """
    _ensure_dirs()
    nifti_path = NIFTI_ROOT / f"{case_id}.nii.gz"
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI volume not found for case_id={case_id}: {nifti_path}")

    out_path = MASK_ROOT / f"{case_id}_lesion_mask.nii.gz"

    if _medsam2_available():
        if _run_medsam2_subprocess(nifti_path, point_zyx, out_path):
            return out_path
    return _placeholder_mask(nifti_path, point_zyx, out_path)


def run_medsam2_from_bbox(case_id: str, bbox: Union[dict, object]) -> Path:
    """
    Run MedSAM2 using the 3D bounding box from TotalSegmentator (Step 3 → MedSAM2 bridge).

    Use this when you have already run grounding and want to segment from the organ bbox
    without a human-confirmed point. The key slice is the center of the bbox along z;
    the 2D box on that slice is passed to the MedSAM2 predictor.

    Parameters
    ----------
    case_id:
        Identifier that maps to the NIfTI volume {case_id}.nii.gz.
    bbox:
        Either a dict with keys x, y, z, width, height, depth (voxels), or an object
        with those attributes (e.g. imaging.grounding.BoundingBox).

    Returns
    -------
    Path
        Path to the generated 3D binary mask (.nii.gz) under MASK_ROOT.
    """
    _ensure_dirs()
    nifti_path = NIFTI_ROOT / f"{case_id}.nii.gz"
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI volume not found for case_id={case_id}: {nifti_path}")

    out_path = MASK_ROOT / f"{case_id}_lesion_mask.nii.gz"

    if _medsam2_available():
        if _run_medsam2_subprocess_bbox(nifti_path, bbox, out_path):
            return out_path
    return _placeholder_mask_from_bbox(nifti_path, bbox, out_path)


__all__ = ["run_medsam2", "run_medsam2_from_bbox", "MASK_ROOT", "MEDSAM2_ROOT"]

