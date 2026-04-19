"""Unified LLM interface supporting multiple providers."""

import asyncio
import json
import os
import re
import time
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from src.utils.logging_utils import get_logger, log_llm_call, Timer

logger = get_logger(__name__)


class LLMClient:
    """Unified LLM client with retry, logging, and multi-provider support."""

    def __init__(self, config_path: str = "configs/llm_config.yaml"):
        with open(config_path) as f:
            raw = f.read()
        # Resolve environment variables in config
        for match in re.findall(r"\$\{(\w+)\}", raw):
            raw = raw.replace(f"${{{match}}}", os.environ.get(match, ""))
        self.config = yaml.safe_load(raw)

        self.default_provider = self.config.get("default_provider", "anthropic")
        self.call_params = self.config.get("call_params", {})
        self._clients: dict[str, Any] = {}
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize API clients for configured providers."""
        providers = self.config.get("providers", {})

        if "anthropic" in providers:
            try:
                import anthropic
                self._clients["anthropic"] = anthropic.AsyncAnthropic(
                    api_key=providers["anthropic"].get("api_key", "")
                )
            except ImportError:
                logger.warning("anthropic package not installed")

        if "openai" in providers:
            try:
                import openai
                self._clients["openai"] = openai.AsyncOpenAI(
                    api_key=providers["openai"].get("api_key", "")
                )
            except ImportError:
                logger.warning("openai package not installed")

    def _get_model(self, provider: Optional[str] = None) -> str:
        """Get default model for a provider."""
        prov = provider or self.default_provider
        return self.config["providers"][prov].get("default_model", "")

    def _get_call_params(self, params_key: Optional[str] = None) -> dict:
        """Get call parameters for a specific agent/role."""
        if params_key and params_key in self.call_params:
            return dict(self.call_params[params_key])
        return {}

    async def complete(
        self,
        messages: list[dict],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        params_key: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Send a completion request with retry and logging."""
        prov = provider or self.default_provider
        mdl = model or self._get_model(prov)
        params = self._get_call_params(params_key)
        params.update(kwargs)

        max_retries = 3
        delays = [2, 4, 8]

        for attempt in range(max_retries + 1):
            try:
                with Timer() as timer:
                    result = await self._call_provider(prov, mdl, messages, params)

                log_llm_call(
                    logger,
                    model=mdl,
                    prompt_tokens=self._estimate_tokens(messages),
                    completion_tokens=self._estimate_tokens_str(result),
                    latency_ms=timer.elapsed_ms,
                    agent=params_key or "",
                )
                return result

            except Exception as e:
                if attempt < max_retries:
                    delay = delays[attempt]
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"LLM call failed after {max_retries + 1} attempts: {e}")
                    raise

        return ""  # unreachable

    async def complete_json(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        params_key: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Complete and parse response as a Pydantic model."""
        raw = await self.complete(
            messages, provider=provider, model=model,
            params_key=params_key, **kwargs
        )
        return self._parse_json_response(raw, schema)

    async def _call_provider(
        self, provider: str, model: str, messages: list[dict], params: dict
    ) -> str:
        """Dispatch to the appropriate provider."""
        if provider == "anthropic":
            return await self._call_anthropic(model, messages, params)
        elif provider == "openai":
            return await self._call_openai(model, messages, params)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _call_anthropic(
        self, model: str, messages: list[dict], params: dict
    ) -> str:
        client = self._clients.get("anthropic")
        if not client:
            raise RuntimeError("Anthropic client not initialized")

        # Separate system message from conversation
        system_text = ""
        conv_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                conv_messages.append(msg)

        # Ensure at least one user message
        if not conv_messages:
            conv_messages = [{"role": "user", "content": system_text.strip()}]
            system_text = ""

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": conv_messages,
            "max_tokens": params.pop("max_tokens", 4096),
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()
        if "temperature" in params:
            kwargs["temperature"] = params.pop("temperature")
        kwargs.update(params)

        response = await client.messages.create(**kwargs)
        return response.content[0].text

    async def _call_openai(
        self, model: str, messages: list[dict], params: dict
    ) -> str:
        client = self._clients.get("openai")
        if not client:
            raise RuntimeError("OpenAI client not initialized")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if "max_tokens" in params:
            kwargs["max_tokens"] = params.pop("max_tokens")
        if "temperature" in params:
            kwargs["temperature"] = params.pop("temperature")
        kwargs.update(params)

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json_response(raw: str, schema: type[BaseModel]) -> BaseModel:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Try direct parse first
        try:
            return schema.model_validate_json(raw)
        except Exception:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
        if match:
            try:
                return schema.model_validate_json(match.group(1))
            except Exception:
                pass

        # Try finding JSON object in text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return schema.model_validate_json(match.group(0))

        raise ValueError(f"Could not parse JSON from response: {raw[:200]}")

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        """Estimate token count for messages."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4o")
            total = 0
            for msg in messages:
                total += len(enc.encode(msg.get("content", "")))
            return total
        except Exception:
            # Fallback: ~4 chars per token
            return sum(len(msg.get("content", "")) for msg in messages) // 4

    @staticmethod
    def _estimate_tokens_str(text: str) -> int:
        """Estimate token count for a string."""
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4o")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4
