import json
from typing import Any, Dict, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from nlp.llm_providers import get_clinical_llm


class ClinicalIntent(TypedDict):
    """
    JSON contract for clinical intent extraction.

    Keys
    ----
    organ:
        Primary organ involved, using snake_case and side when clear, e.g.
        "kidney_right", "kidney_left", "liver", "pancreas". May be null if
        not clearly specified.
    region:
        Sub-region or location within the organ, e.g. "superior pole",
        "inferior pole", "hilum", "upper lobe". May be null.
    finding:
        Short noun phrase summarizing the main abnormality, e.g. "mass",
        "cyst", "lesion", "aneurysm". May be null.
    """

    organ: Optional[str]
    region: Optional[str]
    finding: Optional[str]


def _build_messages(report_text: str) -> list:
    """
    Build the chat messages payload for clinical intent extraction.
    """
    system_message = (
        "You are an expert radiology NLP assistant. "
        "Your task is to read complex radiology report text (usually CT or MRI findings) "
        "and extract a minimal, structured representation of the primary finding by identifying the 'Where' and the 'What'.\n\n"
        "Output must be STRICT JSON, with no explanations or extra text. The JSON "
        "must have exactly these keys:\n"
        '  - "organ" (The Where - Major): the primary organ involved, using snake_case and side when clear, '
        'e.g. "kidney_right", "kidney_left", "liver", "pancreas".\n'
        '  - "region" (The Where - Minor): a short phrase describing the sub-region or location within the organ, '
        'e.g. "superior pole", "inferior pole", "hilum", "upper lobe". If not specified, use null.\n'
        '  - "finding" (The What): a short noun phrase summarizing the main abnormality, e.g. '
        '"mass", "cyst", "lesion", "aneurysm".\n'
        "If you are uncertain about a field, set it to null rather than guessing.\n"
    )

    # Few-shot example using the user's kidney mass case
    example_user = (
        "A 4.5cm hypo-attenuating mass is noted in the superior pole of the right kidney."
    )
    example_assistant = json.dumps(
        {
            "organ": "kidney_right",
            "region": "superior pole",
            "finding": "mass",
        }
    )

    instructions = (
        "Now process the following radiology finding text.\n\n"
        "Return ONLY a single JSON object with keys organ, region, and finding.\n"
        "Do not include any extra commentary or markdown.\n"
    )

    return [
        SystemMessage(content=system_message),
        HumanMessage(content=example_user),
        AIMessage(content=example_assistant),
        HumanMessage(content=instructions + report_text),
    ]


def _call_model(
    report_text: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Call the LangChain chat model and return the raw string content.
    """
    llm = get_clinical_llm(provider=provider, model=model)
    messages = _build_messages(report_text)
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    if not content or not str(content).strip():
        raise RuntimeError("Model returned empty content for clinical intent extraction.")
    return str(content)


def _extract_json_from_text(raw: str) -> str:
    """
    Attempt to extract a JSON object from model output (recovery for markdown/prefix).
    """
    raw = raw.strip()
    # Try direct parse first
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    # Strip common markdown wrappers
    for prefix in ("```json", "```"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix) :].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    # Find first { and last } to extract JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return raw


def _coerce_to_json_object(raw: str) -> Dict[str, Any]:
    """
    Parse the raw model output as JSON, with recovery for non-JSON prefix/suffix.
    """
    extracted = _extract_json_from_text(raw)
    try:
        data = json.loads(extracted)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model output as JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Model output JSON is not an object.")
    return data


def extract_clinical_intent(
    report_text: str,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> ClinicalIntent:
    """
    Extract clinical intent from a radiology report text.

    Parameters
    ----------
    report_text:
        The free-text radiology report or key finding sentence.
    model:
        Optional override for the model name.
    provider:
        Optional override for the LLM provider (openrouter | github | groq).

    Returns
    -------
    dict
        A dictionary with keys:
        - organ: str | None
        - region: str | None
        - finding: str | None
    """
    if not report_text or not report_text.strip():
        raise ValueError("report_text must be a non-empty string.")

    raw = _call_model(report_text, model=model, provider=provider)
    data = _coerce_to_json_object(raw)

    # Normalize keys and provide defaults if missing
    organ = data.get("organ")
    region = data.get("region")
    finding = data.get("finding")

    normalized: ClinicalIntent = {
        "organ": organ if isinstance(organ, str) or organ is None else str(organ),
        "region": region if isinstance(region, str) or region is None else str(region),
        "finding": finding if isinstance(finding, str) or finding is None else str(finding),
    }

    return normalized


__all__ = ["extract_clinical_intent"]

