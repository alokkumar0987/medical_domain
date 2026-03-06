import json
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nlp.clinical_intent_extractor import extract_clinical_intent


def main() -> None:
    """
    Minimal example that runs Step 1 (clinical intent extraction)
    on the kidney mass case from the project description.
    """
    report_text = (
        "A 4.5cm hypo-attenuating mass is noted in the superior pole of the right kidney."
    )

    result = extract_clinical_intent(report_text)

    print("Input report:")
    print(report_text)
    print("\nExtracted clinical intent JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

