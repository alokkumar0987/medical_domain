"""
Read LIDC-IDRI-0001 digest Excel and write a human-readable report .txt file.
"""
import os
import pandas as pd

EXCEL_PATH = r"D:\project\medical_3Dmodel\test_data\LIDC-IDRI-0001_digest.xlsx"
REPORT_PATH = r"D:\project\medical_3Dmodel\test_data\LIDC-IDRI-0001_report.txt"


def main():
    if not os.path.isfile(EXCEL_PATH):
        print(f"Error: Excel not found: {EXCEL_PATH}")
        return

    df = pd.read_excel(EXCEL_PATH)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 200)

    lines = []
    lines.append("=" * 60)
    lines.append("LIDC-IDRI-0001 Digest Report")
    lines.append(f"Source: {EXCEL_PATH}")
    lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    lines.append("=" * 60)

    for idx, row in df.iterrows():
        lines.append("")
        lines.append(f"--- Series / Row {idx + 1} ---")
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                val = ""
            else:
                val = str(val).strip()
            lines.append(f"  {col}: {val}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("End of report")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report written: {REPORT_PATH}")
    print(f"Lines: {len(lines)}")


if __name__ == "__main__":
    main()
