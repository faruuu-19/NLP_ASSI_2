from __future__ import annotations

import json
import os
from typing import AsyncGenerator

import httpx


class LLMUnavailableError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, base_url=None, model=None, timeout_seconds: float = 60.0):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "qwen2.5:1.5b"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=5.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
            models = [model["name"] for model in payload.get("models", [])]
            return {
                "available": True,
                "base_url": self.base_url,
                "model": self.model,
                "installed": self.model in models,
                "models": models,
            }
        except httpx.HTTPError as exc:
            return {
                "available": False,
                "base_url": self.base_url,
                "model": self.model,
                "installed": False,
                "error": str(exc),
            }

    async def chat(self, system_prompt: str, history: list[dict], user_message: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_message}],
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.8},
        }

        try:
            response = await self._client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except (KeyError, httpx.HTTPError) as exc:
            raise LLMUnavailableError(str(exc)) from exc

    async def stream_chat(self, system_prompt: str, history: list[dict], user_message: str) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_message}],
            "stream": True,
            "options": {"temperature": 0.2, "top_p": 0.8},
        }

        try:
            async with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
        except (json.JSONDecodeError, httpx.HTTPError) as exc:
            raise LLMUnavailableError(str(exc)) from exc
