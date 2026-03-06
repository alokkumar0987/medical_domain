"""
Create test_data folder with LIDC-IDRI-0001: copy .dcm files and save a test manifest (Excel rows for that patient).
"""
import os
import shutil
import pandas as pd

PROJECT_ROOT = r"D:\project\medical_3Dmodel"
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")
PATIENT_ID = "LIDC-IDRI-0001"

# Source paths from read_csv.py
DICOM_SOURCE = os.path.join(
    PROJECT_ROOT,
    "manifest-1600709154662", "LIDC-IDRI", "LIDC-IDRI-0001",
    "01-01-2000-NA-NA-30178", "3000566.000000-NA-03192"
)
EXCEL_SOURCE = os.path.join(PROJECT_ROOT, "TCIA_LIDC-IDRI_20200921-nbia-digest.xlsx")


def main():
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    patient_dcm_dir = os.path.join(TEST_DATA_DIR, PATIENT_ID)
    os.makedirs(patient_dcm_dir, exist_ok=True)

    # Copy .dcm files
    if os.path.isdir(DICOM_SOURCE):
        count = 0
        for f in os.listdir(DICOM_SOURCE):
            if f.lower().endswith(".dcm"):
                src = os.path.join(DICOM_SOURCE, f)
                dst = os.path.join(patient_dcm_dir, f)
                shutil.copy2(src, dst)
                count += 1
        print(f"Copied {count} .dcm files to {patient_dcm_dir}")
    else:
        print(f"DICOM source not found: {DICOM_SOURCE}")

    # Filter Excel to this patient, keep only CT series (Image Count > 10), save as test file
    if os.path.isfile(EXCEL_SOURCE):
        df = pd.read_excel(EXCEL_SOURCE)
        patient_df = df[df["Patient ID"] == PATIENT_ID]
        # Remove DX row (2 .dcm); keep only CT series (133 .dcm)
        patient_df = patient_df[patient_df["Image Count"] > 10]
        test_excel_path = os.path.join(TEST_DATA_DIR, "LIDC-IDRI-0001_digest.xlsx")
        patient_df.to_excel(test_excel_path, index=False)
        print(f"Saved test manifest: {test_excel_path} ({len(patient_df)} rows, CT only)")
    else:
        print(f"Excel source not found: {EXCEL_SOURCE}")

    print(f"test_data ready: {TEST_DATA_DIR}")


if __name__ == "__main__":
    main()
