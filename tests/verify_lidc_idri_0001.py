"""
Verify LIDC-IDRI-0001: test_data DICOMs match the patient Excel (same patient, readable Excel).
"""
import os
import pydicom
import pandas as pd

PROJECT_ROOT = r"D:\project\medical_3Dmodel"
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")
PATIENT_ID = "LIDC-IDRI-0001"
DCM_DIR = os.path.join(TEST_DATA_DIR, PATIENT_ID)
EXCEL_PATH = os.path.join(TEST_DATA_DIR, "LIDC-IDRI-0001_digest.xlsx")


def main():
    print("=== Verify LIDC-IDRI-0001 (DICOMs vs patient Excel) ===\n")

    # 1. Read patient Excel file
    if not os.path.isfile(EXCEL_PATH):
        print(f"FAIL: Excel not found: {EXCEL_PATH}")
        return
    df = pd.read_excel(EXCEL_PATH)
    print(f"OK: Read patient Excel: {EXCEL_PATH}")
    print(f"    Rows: {len(df)}, Columns: {list(df.columns)[:8]}...")
    print(f"    Patient ID in Excel: {df['Patient ID'].unique().tolist()}\n")

    # 2. Load DICOMs and read patient/series from tags
    if not os.path.isdir(DCM_DIR):
        print(f"FAIL: DICOM folder not found: {DCM_DIR}")
        return
    dcm_files = [f for f in os.listdir(DCM_DIR) if f.lower().endswith(".dcm")]
    print(f"OK: Found {len(dcm_files)} .dcm files in {DCM_DIR}")

    # Read first and middle DICOM for Patient ID / Series
    first = pydicom.dcmread(os.path.join(DCM_DIR, dcm_files[0]))
    dcm_patient_id = getattr(first, "PatientID", "") or ""
    dcm_series_uid = getattr(first, "SeriesInstanceUID", None)
    dcm_modality = getattr(first, "Modality", None)
    print(f"    DICOM PatientID: {dcm_patient_id}")
    print(f"    DICOM Modality:   {dcm_modality}")
    print(f"    DICOM Series UID: {str(dcm_series_uid)[:50]}...\n")

    # 3. Cross-check: Excel Patient ID == DICOM PatientID
    excel_ids = set(df["Patient ID"].dropna().astype(str))
    match = PATIENT_ID in excel_ids and str(dcm_patient_id).strip() == PATIENT_ID
    if match:
        print(f"OK: Patient ID match — Excel and DICOM both '{PATIENT_ID}'")
    else:
        print(f"MISMATCH: Excel Patient ID(s)={excel_ids}, DICOM PatientID='{dcm_patient_id}'")

    # 4. Image count: Excel CT row vs actual .dcm count
    ct_rows = df[(df["Patient ID"] == PATIENT_ID) & (df["Image Count"] > 10)]
    if not ct_rows.empty:
        expected_count = int(ct_rows["Image Count"].iloc[0])
        if expected_count == len(dcm_files):
            print(f"OK: Image count match — Excel CT series={expected_count}, .dcm files={len(dcm_files)}")
        else:
            print(f"NOTE: Excel CT Image Count={expected_count}, .dcm files={len(dcm_files)}")
    else:
        print("NOTE: No CT row with Image Count > 10 in Excel for this patient.")

    print("\n=== Verification done ===")


if __name__ == "__main__":
    main()
