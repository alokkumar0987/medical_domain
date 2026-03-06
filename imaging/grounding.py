"""
Anatomical grounding utilities using TotalSegmentator.

Given a NIfTI volume, run TotalSegmentator to obtain organ label maps and
compute a rough 3D bounding box for a specific organ (e.g., kidney_right).

This module focuses on file I/O and geometry; orchestration lives in FastAPI.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Tuple

import nibabel as nib
import numpy as np
from pydantic import BaseModel

from imaging.ingest import DATA_ROOT, NIFTI_ROOT


TOTALSEG_ROOT = Path(os.getenv("TOTALSEG_ROOT", DATA_ROOT / "totalseg"))


class BoundingBox(BaseModel):
    x: int
    y: int
    z: int
    width: int
    height: int
    depth: int


def _run_totalsegmentator(nifti_path: Path, output_dir: Path) -> None:
    """
    Run TotalSegmentator in default mode.

    Requires the `TotalSegmentator` CLI to be installed and visible on PATH,
    or configured via TOTALSEG_BIN env var.
    """
    bin_path = os.getenv("TOTALSEG_BIN", "TotalSegmentator")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        bin_path,
        "-i",
        str(nifti_path),
        "-o",
        str(output_dir),
        "--fast",
    ]
    subprocess.run(cmd, check=True)


def _organ_label_path(output_dir: Path, organ_label: str) -> Path:
    """
    Resolve the path to a specific organ label NIfTI from TotalSegmentator output.

    By default, TotalSegmentator produces individual organ files like
    'kidney_right.nii.gz' in the output directory.
    """
    candidate = output_dir / f"{organ_label}.nii.gz"
    if not candidate.exists():
        raise FileNotFoundError(f"Organ label file not found: {candidate}")
    return candidate


def _compute_bbox_from_mask(mask_path: Path, spacing: Tuple[float, float, float]) -> BoundingBox:
    img = nib.load(str(mask_path))
    # Reorient to closest canonical (RAS+) so axis order is deterministic.
    # After as_closest_canonical, data axes are consistently (i, j, k) = (x, y, z)
    # in RAS space. TotalSegmentator outputs NIfTI in the same orientation as the
    # input (which dcm2niix writes as RAS), so no data is lost.
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()
    # Assume binary mask; threshold > 0
    coords = np.argwhere(data > 0)
    if coords.size == 0:
        raise ValueError("No voxels found for organ in segmentation mask.")

    # coords axes: dim0=x(cols), dim1=y(rows), dim2=z(slices) after canonical reorientation.
    # The rest of the pipeline (slices.py, medsam2_runner) stores voxels as (z, y, x);
    # we map back: x_min→x, y_min→y, z_min→z.
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    # Convert to voxel-space bounding box
    x = int(x_min)
    y = int(y_min)
    z = int(z_min)
    width = int(x_max - x_min + 1)
    height = int(y_max - y_min + 1)
    depth = int(z_max - z_min + 1)

    # Physical spacing is available if needed for metric conversion; we keep voxel box here.
    _ = spacing  # currently unused but kept for future extensions

    return BoundingBox(x=x, y=y, z=z, width=width, height=height, depth=depth)


def run_grounding_for_case(
    case_id: str,
    organ_label: Literal["kidney_right", "kidney_left", "liver", "pancreas"] = "kidney_right",
) -> BoundingBox:
    """
    High-level helper:
    - Locate the NIfTI volume for the given case_id.
    - Run TotalSegmentator (if output not already cached).
    - Compute and return a bounding box for the requested organ.
    """
    nifti_path = NIFTI_ROOT / f"{case_id}.nii.gz"
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI volume not found for case_id={case_id}: {nifti_path}")

    case_out_dir = TOTALSEG_ROOT / case_id
    # Reuse previous results if directory already exists.
    if not case_out_dir.exists() or not any(case_out_dir.glob("*.nii.gz")):
        # Use a temp directory to avoid partial results on failure, then move.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir) / "totalseg"
            _run_totalsegmentator(nifti_path, tmp_out)
            case_out_dir.parent.mkdir(parents=True, exist_ok=True)
            if case_out_dir.exists():
                shutil.rmtree(case_out_dir)
            shutil.move(str(tmp_out), str(case_out_dir))

    label_path = _organ_label_path(case_out_dir, organ_label)

    # Get spacing from original volume (canonical reorientation may swap axis order)
    img = nib.load(str(nifti_path))
    img_can = nib.as_closest_canonical(img)
    hdr = img_can.header
    zooms = hdr.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    return _compute_bbox_from_mask(label_path, spacing)


__all__ = ["BoundingBox", "run_grounding_for_case"]

