import asyncio
import os
import time
import warnings
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_GEMINI_VERSION = re.compile(r"^gemini-(\d+)\.(\d+)")


def _accepts_minimal_thinking(model: str) -> bool:
    """Whether ``thinking_level="minimal"`` is accepted: numbered Flash models
    before 3.8. Pro, 3.8+ and version-less aliases (which move between
    generations) are treated as rejecting it."""
    model_lc = model.lower()
    match = _GEMINI_VERSION.match(model_lc)
    return bool(match) and "pro" not in model_lc and (
        (int(match.group(1)), int(match.group(2))) < (3, 8)
    )


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list of typed blocks.
    This normalizes to string for consistent downstream handling.
    """

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "429" in msg
            or "resource_exhausted" in msg
            or "rate limit" in msg
            or "too many requests" in msg
        )

    @staticmethod
    def _rate_limit_delay_seconds() -> int:
        raw = os.getenv("GOOGLE_429_RETRY_DELAY_SECONDS", "60").strip()
        try:
            value = int(raw)
        except ValueError:
            return 60
        return value if value > 0 else 60

    def _invoke_with_rate_limit_retry(self, call):
        try:
            return call()
        except Exception as exc:
            if not self._is_rate_limited(exc):
                raise
            delay = self._rate_limit_delay_seconds()
            warnings.warn(
                (
                    f"Google API rate limit encountered (429). "
                    f"Retrying once after {delay} seconds."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            time.sleep(delay)
            return call()

    async def _ainvoke_with_rate_limit_retry(self, call):
        try:
            return await call()
        except Exception as exc:
            if not self._is_rate_limited(exc):
                raise
            delay = self._rate_limit_delay_seconds()
            warnings.warn(
                (
                    f"Google API rate limit encountered (429). "
                    f"Retrying once after {delay} seconds."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            await asyncio.sleep(delay)
            return await call()

    def invoke(self, input, config=None, **kwargs):
        parent_invoke = super().invoke
        response = self._invoke_with_rate_limit_retry(
            lambda: parent_invoke(input, config, **kwargs)
        )
        return normalize_content(response)

    def _generate(self, *args, **kwargs):
        parent_generate = super()._generate
        return self._invoke_with_rate_limit_retry(
            lambda: parent_generate(*args, **kwargs)
        )

    async def _agenerate(self, *args, **kwargs):
        parent_agenerate = super()._agenerate
        return await self._ainvoke_with_rate_limit_retry(
            lambda: parent_agenerate(*args, **kwargs)
        )


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    _MODEL_ALIASES = {
        # Retired preview ID: route to its stable 2.5 equivalent. Gemini 3.1
        # models remain supported and accept the current thinking_level API.
        "gemini-3-flash-preview": "gemini-2.5-flash",
    }

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        self.warn_if_unknown_model()
        requested_model = self.model
        model = self._MODEL_ALIASES.get(requested_model, requested_model)
        if model != requested_model:
            warnings.warn(
                (
                    f"Google model '{requested_model}' is deprecated or unavailable; "
                    f"using fallback '{model}'."
                ),
                UserWarning,
                stacklevel=2,
            )

        llm_kwargs = {"model": model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "temperature", "max_output_tokens",
                    "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        # Gemini 3.x takes the string ``thinking_level`` (the integer
        # ``thinking_budget`` was for the now-retired 2.5 line). Pro, Gemini
        # 3.8+ and the -latest aliases reject "minimal" with a 400; "low" is
        # accepted everywhere, so it is the fallback.
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            if model.startswith("gemini-2.5-"):
                llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0
            else:
                if thinking_level == "minimal" and not _accepts_minimal_thinking(model):
                    thinking_level = "low"
                llm_kwargs["thinking_level"] = thinking_level

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
