"""
Ingestion + normalization of medical imaging to NIfTI.

Supports:
- DICOM series provided as a ZIP archive (for CT/MRI)  -> NIfTI via dcm2niix
- Existing NIfTI (.nii / .nii.gz)                     -> validated and copied
- NRRD (.nrrd)                                        -> converted via SimpleITK
- 2D images (X-ray/US: .png/.jpg/.dcm single file)    -> pseudo-3D NIfTI with depth=1

This module is backend-agnostic; FastAPI code imports and uses these helpers.
"""

import os
import shutil
import tempfile
import uuid
import zipfile
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PIL import Image
from pydantic import BaseModel


DATA_ROOT = Path(os.getenv("DATA_ROOT", "data")).resolve()
UPLOAD_ROOT = DATA_ROOT / "uploads"
NIFTI_ROOT = DATA_ROOT / "nifti"


class Modality(str, Enum):
    CT = "ct"
    MRI = "mri"
    XRAY = "xray"
    US = "us"


class SourceFormat(str, Enum):
    ZIP_DICOM = "zip_dicom"
    NIFTI = "nifti"
    NRRD = "nrrd"
    IMAGE = "image"
    DICOM_SINGLE = "dicom_single"


class IngestResult(BaseModel):
    scan_id: str
    nifti_path: str
    shape: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    modality: Modality
    source_format: SourceFormat


def _ensure_dirs() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    NIFTI_ROOT.mkdir(parents=True, exist_ok=True)


def generate_scan_id() -> str:
    return str(uuid.uuid4())


def save_upload_to_disk(scan_id: str, filename: str, content: bytes) -> Path:
    """
    Save an uploaded file under data/uploads/{scan_id}/raw/.
    """
    _ensure_dirs()
    raw_dir = UPLOAD_ROOT / scan_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    target = raw_dir / safe_name
    target.write_bytes(content)
    return target


def detect_format(path: Path) -> SourceFormat:
    """
    Detect source format based on extension and simple heuristics.
    """
    ext = path.suffix.lower()
    if ext == ".zip":
        return SourceFormat.ZIP_DICOM
    if path.name.endswith(".nii.gz") or ext == ".nii":
        # Check full name for .nii.gz (Path.suffix only strips last extension)
        return SourceFormat.NIFTI
    if ext == ".nrrd":
        return SourceFormat.NRRD
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return SourceFormat.IMAGE
    if ext == ".dcm":
        # Single DICOM file (e.g. X-ray)
        return SourceFormat.DICOM_SINGLE
    # Fallback: attempt NIfTI, otherwise treat as image
    try:
        nib.load(str(path))
        return SourceFormat.NIFTI
    except Exception:
        return SourceFormat.IMAGE


def _find_dicom_dir(base: Path) -> Path:
    """
    Given a directory (e.g. the result of extractall on a ZIP), find the
    deepest sub-directory that actually contains .dcm files.

    When users ZIP a folder like LIDC-IDRI-0001/*.dcm, extractall produces:
        tmpdir/LIDC-IDRI-0001/1-001.dcm …
    This helper walks down to the right level so dcm2niix is not given an
    empty parent directory.
    """
    for root, _dirs, files in os.walk(base):
        if any(f.lower().endswith(".dcm") for f in files):
            return Path(root)
    # Fallback: return base and let dcm2niix report the error itself
    return base


def _run_dcm2niix(input_dir: Path, output_nifti: Path) -> None:
    """
    Run dcm2niix to convert a DICOM series directory to a NIfTI file.

    Requires dcm2niix installed and available in PATH, or a custom binary
    specified via DCM2NIIX_BIN env var.
    """
    import subprocess

    bin_path = os.getenv("DCM2NIIX_BIN", "dcm2niix")

    output_dir = output_nifti.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bug 2 fix: Path.stem on "scan.nii.gz" gives "scan.nii", not "scan".
    # Strip both .nii.gz and .nii suffixes to get the bare stem.
    name = output_nifti.name
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        stem = name[: -len(".nii")]
    else:
        stem = output_nifti.stem

    cmd = [
        bin_path,
        "-z",
        "y",  # gzip
        "-f",
        stem,
        "-o",
        str(output_dir),
        str(input_dir),
    ]
    subprocess.run(cmd, check=True)

    # dcm2niix may add suffixes; pick the first matching .nii.gz
    candidates = sorted(output_dir.glob(f"{stem}*.nii*"))
    if not candidates:
        raise RuntimeError("dcm2niix did not produce any NIfTI output.")
    if candidates[0] != output_nifti:
        shutil.move(candidates[0], output_nifti)


def _nifti_metadata(path: Path) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    img = nib.load(str(path))
    data = img.get_fdata()
    shape = tuple(int(x) for x in data.shape[-3:])  # handle channels if present
    hdr = img.header
    zooms = hdr.get_zooms()
    spacing = tuple(float(x) for x in zooms[:3])
    return shape, spacing


def _image_to_pseudo_nifti(path: Path, out_path: Path) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """
    Convert a 2D image (X-ray/US) to a pseudo-3D NIfTI volume with depth=1.
    """
    img = Image.open(path).convert("L")  # grayscale
    arr2d = np.array(img, dtype=np.float32)
    vol = arr2d[None, ...]  # shape (1, H, W)

    affine = np.eye(4, dtype=np.float32)
    nifti = nib.Nifti1Image(vol, affine=affine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti, str(out_path))

    shape = (1, arr2d.shape[0], arr2d.shape[1])
    spacing = (1.0, 1.0, 1.0)
    return shape, spacing


def _nrrd_to_nifti(path: Path, out_path: Path) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """
    Convert NRRD to NIfTI via SimpleITK.
    """
    image = sitk.ReadImage(str(path))
    sitk.WriteImage(image, str(out_path))
    spacing = image.GetSpacing()
    size = image.GetSize()
    # SimpleITK uses (x, y, z) ordering; convert to (z, y, x) for shape
    shape = (size[2], size[1], size[0]) if len(size) == 3 else (1, size[1], size[0])
    spacing3 = (
        float(spacing[2]) if len(spacing) > 2 else 1.0,
        float(spacing[1]) if len(spacing) > 1 else 1.0,
        float(spacing[0]) if len(spacing) > 0 else 1.0,
    )
    return shape, spacing3


def normalize_to_nifti(
    scan_id: str,
    source_path: Path,
    modality: Modality,
) -> IngestResult:
    """
    Normalize an uploaded scan to a NIfTI file under data/nifti/{scan_id}.nii.gz.
    """
    _ensure_dirs()
    src_format = detect_format(source_path)
    out_path = NIFTI_ROOT / f"{scan_id}.nii.gz"

    if src_format == SourceFormat.ZIP_DICOM:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(source_path, "r") as zf:
                zf.extractall(tmpdir)
            # Bug A fix: ZIP may contain a subfolder (e.g. LIDC-IDRI-0001/*.dcm).
            # Walk into the first directory that actually has .dcm files.
            dicom_dir = _find_dicom_dir(Path(tmpdir))
            _run_dcm2niix(dicom_dir, out_path)
        shape, spacing = _nifti_metadata(out_path)
    elif src_format == SourceFormat.NIFTI:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, out_path)
        shape, spacing = _nifti_metadata(out_path)
    elif src_format == SourceFormat.NRRD:
        shape, spacing = _nrrd_to_nifti(source_path, out_path)
    elif src_format in (SourceFormat.IMAGE, SourceFormat.DICOM_SINGLE):
        # Treat as 2D X-ray/US image
        shape, spacing = _image_to_pseudo_nifti(source_path, out_path)
    else:
        raise ValueError(f"Unsupported source format: {src_format}")

    return IngestResult(
        scan_id=scan_id,
        nifti_path=str(out_path),
        shape=shape,
        spacing=spacing,
        modality=modality,
        source_format=src_format,
    )


__all__ = [
    "Modality",
    "SourceFormat",
    "IngestResult",
    "generate_scan_id",
    "save_upload_to_disk",
    "detect_format",
    "normalize_to_nifti",
]

