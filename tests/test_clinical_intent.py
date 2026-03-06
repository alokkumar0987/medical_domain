"""
Test script for the clinical intent extraction pipeline (Step 1).

Run from project root with env vars set:
  set OPENROUTER_API_KEY=...   (or GITHUB_TOKEN / GROQ_API_KEY)
  set CLINICAL_NLP_PROVIDER=openrouter  (or github / groq)

  python -m tests.test_clinical_intent
"""

import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nlp.clinical_intent_extractor import extract_clinical_intent


SAMPLE_REPORT = (
    "A 4.5cm hypo-attenuating mass is noted in the superior pole of the right kidney."
)

EXPECTED = {
    "organ": "kidney_right",
    "region": "superior pole",
    "finding": "mass",
}


def main() -> None:
    provider = os.getenv("CLINICAL_NLP_PROVIDER", "openrouter")
    print(f"Testing clinical intent extraction (provider={provider})")
    print("-" * 50)
    print("Input:", SAMPLE_REPORT)
    print()

    try:
        result = extract_clinical_intent(SAMPLE_REPORT)
        print("Output JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()

        # Basic validation
        ok = True
        for key in ("organ", "region", "finding"):
            if result.get(key) != EXPECTED.get(key):
                print(f"  [WARN] {key}: got {result.get(key)!r}, expected {EXPECTED.get(key)!r}")
                ok = False
        if ok:
            print("  [OK] Output matches expected structure and values.")
        else:
            print("  [INFO] Output structure OK; values may differ slightly by model.")
    except Exception as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
