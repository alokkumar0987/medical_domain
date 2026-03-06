"""
LLM provider factory for clinical intent extraction (Step 1).

Goals:
- Simple: one function, `get_clinical_llm`, to obtain a chat model.
- Clean: same interface regardless of provider.
- Flexible: user can pick any compatible model name via `CLINICAL_NLP_MODEL`.

Supported providers (via LangChain):
- OpenRouter  -> many models including ChatGPT equivalents like `openai/gpt-4o`
- GitHub AI   -> GitHub Models endpoint (e.g. `gpt-4.1`, `gpt-4.1-mini`)
- Groq        -> Groq-hosted open models (e.g. `llama-3.3-70b-versatile`)

Environment variables:
- CLINICAL_NLP_PROVIDER:  openrouter | github | groq  (default: openrouter)
- CLINICAL_NLP_MODEL:     optional model name override for all providers
- OPENROUTER_API_KEY:     required when provider=openrouter
- GITHUB_TOKEN:           required when provider=github
- GROQ_API_KEY:           required when provider=groq
"""

import os
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel


ProviderName = Literal["openrouter", "github", "groq"]


def _resolve_provider(provider: Optional[str]) -> ProviderName:
    raw = (provider or os.getenv("CLINICAL_NLP_PROVIDER", "openrouter")).strip().lower()
    if raw in ("openrouter", "github", "groq"):
        return raw  # type: ignore[return-value]
    raise ValueError(
        "Invalid CLINICAL_NLP_PROVIDER value: "
        f"{raw!r}. Expected one of: 'openrouter', 'github', 'groq'."
    )


def get_clinical_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseChatModel:
    """
    Return a LangChain chat model for the configured provider.

    You can control the backend with:
    - CLINICAL_NLP_PROVIDER (openrouter | github | groq)
    - CLINICAL_NLP_MODEL   (e.g. 'openai/gpt-4o', 'gpt-4.1', 'llama-3.3-70b-versatile')
    """
    resolved = _resolve_provider(provider)

    if resolved == "openrouter":
        return _openrouter_llm(model)
    if resolved == "github":
        return _github_llm(model)
    # resolved == "groq"
    return _groq_llm(model)


def _openrouter_llm(model: Optional[str] = None) -> BaseChatModel:
    from langchain_openrouter import ChatOpenRouter

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY must be set for OpenRouter provider.")
    return ChatOpenRouter(
        model=model or os.getenv("CLINICAL_NLP_MODEL", "openai/gpt-4o"),
        api_key=key,
        temperature=0.0,
        max_tokens=512,
    )


def _github_llm(model: Optional[str] = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    key = os.getenv("GITHUB_TOKEN")
    if not key:
        raise ValueError("GITHUB_TOKEN must be set for GitHub provider.")
    return ChatOpenAI(
        base_url="https://models.github.ai/inference",
        api_key=key,
        model=model or os.getenv("CLINICAL_NLP_MODEL", "openai/gpt-4.1"),
        temperature=0.0,
        max_tokens=512,
    )


def _groq_llm(model: Optional[str] = None) -> BaseChatModel:
    from langchain_groq import ChatGroq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY must be set for Groq provider.")
    return ChatGroq(
        model=model or os.getenv("CLINICAL_NLP_MODEL", "llama-3.3-70b-versatile"),
        api_key=key,
        temperature=0.0,
        max_tokens=512,
    )


__all__ = ["get_clinical_llm"]
