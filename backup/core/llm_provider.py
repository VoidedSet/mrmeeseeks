"""
llm_provider.py — Mr Meeseeks LLM Provider Abstraction

Swappable backends: groq | ollama
Controlled via env vars:
  LLM_BACKEND = groq | ollama          (default: groq)
  LLM_MODEL   = llama-3.1-8b-instant   (default per backend)
  GROQ_API_KEY = your_key              (required for groq)
  OLLAMA_URL  = http://localhost:11434 (optional override)

When Ollama + Qwen3 comes back: set LLM_BACKEND=ollama, LLM_MODEL=qwen3:3b
Note: if Qwen3 emits <tool_call> XML instead of raw JSON, set QWEN_MODE=1
"""

import os
import json
import logging
import httpx
from abc import ABC, abstractmethod

log = logging.getLogger("llm_provider")

# ── Defaults ──────────────────────────────────────────────────────────────────
_GROQ_DEFAULTS = {
    "model":    "llama-3.1-8b-instant",
    "base_url": "https://api.groq.com/openai/v1",
}
_OLLAMA_DEFAULTS = {
    "model":    "qwen2.5:3b",
    "base_url": "http://localhost:11434",
}


# ── Abstract Base ─────────────────────────────────────────────────────────────
class LLMProvider(ABC):

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        force_json: bool = True,
        tools: list[dict] = None,
    ) -> dict:
        """Send a chat completion request."""

    async def stream_complete(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 512,
        tools: list[dict] = None,
    ):
        """Send a streaming chat completion request."""
        """
        Send a chat completion request.
        force_json=True  → tell the model to output only JSON (for ReAct loop)
        force_json=False → plain text response OK (for conversational path)
        tools            → list of OpenAI-format tool schemas
        Returns: {"content": str, "tool_calls": list[dict]}
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging."""


# ── Groq Provider ─────────────────────────────────────────────────────────────
class GroqProvider(LLMProvider):
    """
    Groq Cloud via OpenAI-compatible REST endpoint.
    No SDK dependency — plain httpx.
    """

    def __init__(self):
        self.api_key  = os.environ.get("GROQ_API_KEY", "")
        self.base_url = os.environ.get("GROQ_BASE_URL", _GROQ_DEFAULTS["base_url"])
        self.model    = os.environ.get("LLM_MODEL",    _GROQ_DEFAULTS["model"])

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com"
            )

        log.info(f"[Groq] model={self.model}")

    @property
    def name(self) -> str:
        return f"groq/{self.model}"

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        force_json: bool = True,
        tools: list[dict] = None,
    ) -> dict:
        payload = {
            "model":       self.model,
            "messages":    [{"role": "system", "content": system_prompt}] + messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }

        # Only force JSON for agentic ReAct calls, not conversational
        if force_json and not tools:
            payload["response_format"] = {"type": "json_object"}
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        message = data["choices"][0]["message"]
        content = message.get("content", "") or ""
        
        parsed_tools = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {}
                parsed_tools.append({
                    "name": func.get("name"),
                    "args": args
                })

        log.debug(f"[Groq] raw → {content[:200]}")
        return {"content": content.strip(), "tool_calls": parsed_tools}

    async def stream_complete(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 512,
        tools: list[dict] = None,
    ):
        payload = {
            "model":       self.model,
            "messages":    [{"role": "system", "content": system_prompt}] + messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      True,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield {"content": content}
                            if delta.get("tool_calls"):
                                for tc in delta["tool_calls"]:
                                    func = tc.get("function", {})
                                    args_str = func.get("arguments", "{}")
                                    try:
                                        args = json.loads(args_str)
                                    except Exception:
                                        args = {}
                                    yield {
                                        "tool_calls": [{
                                            "name": func.get("name"),
                                            "args": args
                                        }]
                                    }
                        except Exception:
                            pass


# ── Ollama Provider ───────────────────────────────────────────────────────────
class OllamaProvider(LLMProvider):
    """
    Local Ollama via its native /api/chat endpoint.
    Returns to this when Ollama + Qwen3 is available.
    """

    def __init__(self):
        base     = os.environ.get("OLLAMA_URL", _OLLAMA_DEFAULTS["base_url"])
        self.url = f"{base}/api/chat"
        self.model = os.environ.get("LLM_MODEL", _OLLAMA_DEFAULTS["model"])
        log.info(f"[Ollama] model={self.model} url={self.url}")

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        force_json: bool = True,
        tools: list[dict] = None,
    ) -> dict:
        payload = {
            "model":    self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream":   False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
            },
        }

        # Disable/enable reasoning thinking process via think parameter
        think_val = os.environ.get("OLLAMA_THINK", "false").lower().strip() == "true"
        payload["think"] = think_val

        # Ollama JSON mode (fallback if no tools)
        if force_json and not tools:
            payload["format"] = "json"
            
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self.url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {})
            content = message.get("content", "") or ""
            
            parsed_tools = []
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    func = tc.get("function", {})
                    args_dict = func.get("arguments", {})
                    parsed_tools.append({
                        "name": func.get("name"),
                        "args": args_dict
                    })

        log.debug(f"[Ollama] raw → {content[:200]}")
        return {"content": content.strip(), "tool_calls": parsed_tools}

    async def stream_complete(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 512,
        tools: list[dict] = None,
    ):
        payload = {
            "model":    self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream":   True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
            },
        }

        # Disable/enable reasoning thinking process via think parameter
        think_val = os.environ.get("OLLAMA_THINK", "false").lower().strip() == "true"
        payload["think"] = think_val

        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", self.url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        message = data.get("message", {})
                        content = message.get("content", "")
                        if content:
                            yield {"content": content}
                        if message.get("tool_calls"):
                            parsed_tcs = []
                            for tc in message["tool_calls"]:
                                func = tc.get("function", {})
                                parsed_tcs.append({
                                    "name": func.get("name"),
                                    "args": func.get("arguments", {})
                                })
                            yield {"tool_calls": parsed_tcs}
                    except Exception:
                        pass


# ── Factory ───────────────────────────────────────────────────────────────────
def create_provider() -> LLMProvider:
    """
    Read LLM_BACKEND env var and return the appropriate provider.

    LLM_BACKEND=groq   → GroqProvider   (default)
    LLM_BACKEND=ollama → OllamaProvider

    To switch back to Ollama + Qwen3:
        export LLM_BACKEND=ollama
        export LLM_MODEL=qwen3:3b
    """
    backend = os.environ.get("LLM_BACKEND", "groq").lower().strip()

    if backend == "groq":
        return GroqProvider()
    elif backend == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(
            f"Unknown LLM_BACKEND='{backend}'. Valid values: groq, ollama"
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
# Instantiated once by main.py after env is loaded.
# brain.py imports this reference — never calls create_provider() itself.
provider: LLMProvider | None = None


def init_provider() -> LLMProvider:
    """Call once from main.py after loading .env."""
    global provider
    provider = create_provider()
    log.info(f"LLM provider initialized: {provider.name}")
    return provider
