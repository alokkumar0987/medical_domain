"""
Extract 2D slices from NIfTI volumes and optionally draw a bounding box overlay.
Used by the volume_slice API and Gradio viewer.

Coordinate systems:
- Voxel space: array indices (0,0,0) to (nz-1, ny-1, nx-1). Used consistently for
  confirmed point and for MedSAM2/TotalSegmentator.
- World space: mm coordinates (e.g. RAS) from the NIfTI affine: world = affine @ [i,j,k,1].
  Use get_affine() and invert if you need world ↔ voxel conversion.

Display and click mapping must use the same orientation. When radiological_display=True
for axial (z) slices, the slice is flipped L-R so "left" on screen is patient left;
pixel_to_voxel(..., radiological_display=True) undoes that so stored (x,y,z) are correct
voxel indices for the same NIfTI.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image


def get_affine(nifti_path: str | Path) -> np.ndarray:
    """
    Return the 4x4 affine matrix from the NIfTI header (voxel indices → world mm).
    Use this for world ↔ voxel conversion or to verify display orientation.
    """
    img = nib.load(str(nifti_path))
    return np.asarray(img.affine)


def pixel_to_voxel(
    px: float,
    py: float,
    axis: Literal["z", "y", "x"],
    index: int,
    shape: Tuple[int, int, int],
    img_width: int,
    img_height: int,
    radiological_display: bool = False,
) -> Tuple[int, int, int]:
    """
    Map (px, py) in the displayed slice image to voxel (x, y, z).

    Must use the same radiological_display value as get_slice_png for this slice,
    so that the returned (x, y, z) match the array indices of the NIfTI volume
    that MedSAM2 and TotalSegmentator use.

    Returns (x, y, z) in array order: data[z, y, x].
    """
    nz, ny, nx = shape[0], shape[1], shape[2]
    iw = max(1, img_width)
    ih = max(1, img_height)
    if axis == "z":
        x = round(px * nx / iw)
        y = round(py * ny / ih)
        z = index
        if radiological_display:
            x = nx - 1 - x
    elif axis == "y":
        x = round(px * nx / iw)
        z = round(py * nz / ih)
        y = index
        if radiological_display:
            x = nx - 1 - x
    else:
        z = round(px * nz / iw)
        y = round(py * ny / ih)
        x = index
        if radiological_display:
            z = nz - 1 - z
    x = max(0, min(x, nx - 1))
    y = max(0, min(y, ny - 1))
    z = max(0, min(z, nz - 1))
    return (x, y, z)


def get_slice_png(
    nifti_path: str | Path,
    axis: Literal["z", "y", "x"],
    index: int,
    bbox_voxel: Optional[Dict[str, int]] = None,
    radiological_display: bool = False,
) -> bytes:
    """
    Extract a 2D slice from a NIfTI volume and return PNG bytes.

    Volume is assumed in (z, y, x) = (dim0, dim1, dim2). The NIfTI affine is not
    applied for reorientation; display is in array order unless radiological_display
    is used.

    If radiological_display is True, axial (z) slices are flipped left–right so
    "left" on screen is patient left (radiological convention). The UI must then
    use pixel_to_voxel(..., radiological_display=True) when mapping click to voxel.

    If bbox_voxel is provided (keys x, y, z, width, height, depth), draw the
    rectangle that intersects this slice.
    """
    img = nib.load(str(nifti_path))
    data = np.asarray(img.get_fdata(), dtype=np.float64)
    # Use last 3 dims for 3D (handle 4D by taking first time volume).
    # NIfTI 4D shape is (nz, ny, nx, nt); data[..., 0] picks t=0 correctly.
    # data[0] would take the first Z-slice across all time points — wrong.
    if data.ndim > 3:
        data = data[..., 0]
    shape = data.shape
    nz, ny, nx = shape[0], shape[1], shape[2]
    index = max(0, min(index, (nz if axis == "z" else ny if axis == "y" else nx) - 1))

    if axis == "z":
        slice_2d = data[index, :, :]  # (y, x)
    elif axis == "y":
        slice_2d = data[:, index, :]   # (z, x)
    else:
        slice_2d = data[:, :, index]   # (z, y)

    if radiological_display and axis == "z":
        slice_2d = np.flip(slice_2d, axis=1)

    # Normalize to 0-255 for display (window/level style: use percentiles to avoid outliers)
    lo, hi = np.percentile(slice_2d, (1, 99))
    if hi <= lo:
        hi = lo + 1
    slice_norm = (np.clip(slice_2d, lo, hi) - lo) / (hi - lo) * 255
    slice_uint8 = np.asarray(slice_norm, dtype=np.uint8)
    pil = Image.fromarray(slice_uint8, mode="L")

    if bbox_voxel:
        # Bbox: x, y, z, width, height, depth in voxel coords
        bx = bbox_voxel.get("x", 0)
        by = bbox_voxel.get("y", 0)
        bz = bbox_voxel.get("z", 0)
        bw = bbox_voxel.get("width", 0)
        bh = bbox_voxel.get("height", 0)
        bd = bbox_voxel.get("depth", 0)
        # Rectangle in slice: which 2D bbox intersects this slice?
        if axis == "z":
            if bz <= index < bz + bd:
                # In slice: (x, y) from (bx, by) to (bx+bw, by+bh)
                x1, y1 = bx, by
                x2, y2 = bx + bw, by + bh
            else:
                x1 = y1 = x2 = y2 = None
        elif axis == "y":
            if by <= index < by + bh:
                x1, y1 = bx, bz
                x2, y2 = bx + bw, bz + bd
            else:
                x1 = y1 = x2 = y2 = None
        else:
            if bx <= index < bx + bw:
                x1, y1 = bz, by
                x2, y2 = bz + bd, by + bh
            else:
                x1 = y1 = x2 = y2 = None

        if x1 is not None:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(pil)
            w, h = pil.size
            x1c = max(0, min(x1, w - 1))
            x2c = max(0, min(x2, w))
            y1c = max(0, min(y1, h - 1))
            y2c = max(0, min(y2, h))
            if radiological_display and axis == "z":
                # x1c/x2c are already in pixel space (0..w); use w not nx.
                x1c, x2c = w - 1 - x2c, w - 1 - x1c
                x1c, x2c = max(0, min(x1c, w - 1)), max(0, min(x2c, w))
            draw.rectangle([x1c, y1c, x2c, y2c], outline="yellow", width=2)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


__all__ = ["get_affine", "get_slice_png", "pixel_to_voxel"]
