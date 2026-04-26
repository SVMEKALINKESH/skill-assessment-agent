"""LLM provider interface and implementation.

Only OpenRouter is supported: a single API key gives access to many
free-tier models (Qwen, Gemma, GPT-OSS, Llama, etc.).

Configuration:
    - OPENROUTER_API_KEY: Set in .env locally or Streamlit Cloud secrets
      for deploy. Get a key at https://openrouter.ai/keys.
    - Model is selected in the UI sidebar; browse models at
      https://openrouter.ai/models (filter for :free tier).
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Structured response returned by an LLMProvider."""

    text: str
    usage: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract interface every LLM backend must implement."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate text from the LLM given a user prompt and optional system prompt."""
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Ask the LLM for JSON and parse the response.

        Uses a lower temperature by default for structured output. Strips
        markdown code fences when the model wraps the JSON in them.
        """
        response = self.generate(prompt, system_prompt, temperature)
        text = response.text.strip()

        # Strip markdown fences such as ```json ... ```.
        if text.startswith("```"):
            lines = text.split("\n")
            # Drop the opening fence (with optional language tag) and closing fence.
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        return json.loads(text)




class OpenRouterProvider(LLMProvider):
    """Default provider backed by OpenRouter's OpenAI-compatible REST API.

    OpenRouter exposes many LLMs (Llama, Mistral, Gemini, Claude, GPT, etc.)
    through a single key. Several models are free-tier.
    Docs: https://openrouter.ai/docs
    """

    DEFAULT_MODEL = "openai/gpt-oss-120b:free"
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        app_name: str = "skill-assessment-agent",
    ):
        resolved_key = api_key or self._load_api_key()
        if not resolved_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY in .env, "
                "Streamlit secrets, or pass api_key explicitly."
            )
        self._api_key = resolved_key
        # Keep model hardcoded at construction time to keep hackathon deploy predictable.
        self._model_name = model or self.DEFAULT_MODEL
        self._app_name = app_name

    @staticmethod
    def _load_api_key() -> str | None:
        """Load the OpenRouter key from Streamlit secrets or an env var."""
        try:
            import streamlit as st

            if "OPENROUTER_API_KEY" in st.secrets:
                return st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            pass
        return os.environ.get("OPENROUTER_API_KEY")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
    ) -> LLMResponse:
        # Lazy import so the module loads even when requests is not installed
        # (for example during syntax checks).
        import requests

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # Optional attribution headers. Override via env vars to appear on
            # the OpenRouter leaderboard; the defaults below are safe placeholders.
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_SITE_URL",
                "https://github.com/your-username/skill-assessment-agent",
            ),
            "X-OpenRouter-Title": os.environ.get(
                "OPENROUTER_SITE_NAME", self._app_name
            ),
        }
        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
        }

        resp = requests.post(self.ENDPOINT, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected OpenRouter response shape: {data}"
            ) from e

        usage: dict[str, Any] = {}
        if "usage" in data:
            u = data["usage"]
            usage = {
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
                "total_tokens": u.get("total_tokens", 0),
            }

        return LLMResponse(text=text, usage=usage)
