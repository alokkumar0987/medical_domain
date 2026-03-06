"""
Marching Cubes mesh construction and lesion metrics.

Given a 3D binary mask (e.g., from MedSAM2), construct a surface mesh and
compute derived clinical metrics. Volume is computed as mask voxel count ×
voxel spacing (pixdim from the NIfTI header), so voxel spacing is included
in the output.

Exports both STL (for 3D printing / surgical planning) and OBJ (better for
color metadata, e.g. blood test warnings on the tissue surface).
"""

import os
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from pydantic import BaseModel
from skimage.measure import marching_cubes
import trimesh

from imaging.ingest import DATA_ROOT
from segmentation.medsam2_runner import MASK_ROOT


MESH_ROOT = Path(os.getenv("MESH_ROOT", DATA_ROOT / "meshes"))


class MeshMetrics(BaseModel):
    mesh_path: str
    mesh_path_obj: str
    volume_cm3: float
    max_diameter_cm: float
    voxel_spacing_mm: Tuple[float, float, float]


def _ensure_dirs() -> None:
    MESH_ROOT.mkdir(parents=True, exist_ok=True)


def mask_to_mesh(case_id: str) -> MeshMetrics:
    """
    Convert the lesion mask for a case into an STL mesh and compute metrics.

    Assumes a mask file at MASK_ROOT/{case_id}_lesion_mask.nii.gz.
    """
    _ensure_dirs()

    mask_path = MASK_ROOT / f"{case_id}_lesion_mask.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found for case_id={case_id}: {mask_path}")

    img = nib.load(str(mask_path))
    data = img.get_fdata()
    hdr = img.header
    zooms = hdr.get_zooms()
    # zooms[i] is the voxel size along array axis i (dim0, dim1, dim2).
    # For a canonical NIfTI loaded as (nz, ny, nx):
    #   zooms[0] = z-spacing (slice thickness)
    #   zooms[1] = y-spacing (row pixel spacing)
    #   zooms[2] = x-spacing (col pixel spacing)
    # marching_cubes spacing= must match array axes: (dim0, dim1, dim2).
    # Volume = voxel_count * product of all three spacings (axis-order doesn't matter).
    s0 = float(zooms[0])  # z (dim0)
    s1 = float(zooms[1])  # y (dim1)
    s2 = float(zooms[2])  # x (dim2)
    voxel_spacing_mm = (round(s0, 4), round(s1, 4), round(s2, 4))

    # Volume from voxel count × pixdim (mm³), then to cm³
    voxel_count = int(np.sum(data > 0.5))
    volume_mm3 = voxel_count * s0 * s1 * s2
    volume_cm3 = volume_mm3 / 1000.0

    # Run marching cubes at iso-level 0.5; spacing must match (dim0, dim1, dim2) order.
    verts, faces, _, _ = marching_cubes(data, level=0.5, spacing=(s0, s1, s2))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Step 5 Enhancement: Apply Laplacian smoothing for a more anatomically realistic 3D mesh
    try:
        from trimesh.smoothing import filter_laplacian
        # Apply 10 iterations of Laplacian smoothing to reduce the blocky "voxel" look
        filter_laplacian(mesh, iterations=10)
    except ImportError:
        pass  # if trimesh is too old or missing scipy, fallback to unsmoothed

    # Max diameter from mesh vertices (mm → cm)
    if len(verts) == 0:
        max_diameter_cm = 0.0
    else:
        sample = verts[:: max(1, len(verts) // 2000)]
        diffs = sample[:, None, :] - sample[None, :, :]
        dists = np.sqrt((diffs**2).sum(axis=2))
        max_diameter_mm = float(dists.max())
        max_diameter_cm = max_diameter_mm / 10.0

    out_stl = MESH_ROOT / f"{case_id}_lesion_mesh.stl"
    out_obj = MESH_ROOT / f"{case_id}_lesion_mesh.obj"
    mesh.export(str(out_stl))
    mesh.export(str(out_obj))

    return MeshMetrics(
        mesh_path=str(out_stl),
        mesh_path_obj=str(out_obj),
        volume_cm3=round(volume_cm3, 2),
        max_diameter_cm=round(max_diameter_cm, 2),
        voxel_spacing_mm=voxel_spacing_mm,
    )


__all__ = ["MeshMetrics", "mask_to_mesh", "MESH_ROOT"]

