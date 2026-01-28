"""LLM provider abstraction for pluggable backends."""
import asyncio
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, model: str) -> str:
        """Synchronous completion."""
        pass

    @abstractmethod
    async def complete_async(self, prompt: str, model: str) -> str:
        """Async completion."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI/ChatGPT provider (requires openai package)."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._async_client = None

    def complete(self, prompt: str, model: str) -> str:
        import openai  # Lazy import

        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key)
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def complete_async(self, prompt: str, model: str) -> str:
        import openai  # Lazy import

        async with openai.AsyncOpenAI(api_key=self.api_key) as client:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic/Claude provider (requires anthropic package)."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    def complete(self, prompt: str, model: str) -> str:
        import anthropic  # Lazy import

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        response = self._client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def complete_async(self, prompt: str, model: str) -> str:
        import anthropic  # Lazy import

        async with anthropic.AsyncAnthropic(api_key=self.api_key) as client:
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text


class CallbackProvider(LLMProvider):
    """Custom callback provider for user-supplied LLM function."""

    def __init__(self, callback: Callable[[str, str], str]):
        self.callback = callback

    def complete(self, prompt: str, model: str) -> str:
        return self.callback(prompt, model)

    async def complete_async(self, prompt: str, model: str) -> str:
        result = self.callback(prompt, model)
        if asyncio.iscoroutine(result):
            return await result
        return result


class ClaudeCodeProvider(LLMProvider):
    """Claude Code headless mode provider (uses claude -p CLI).

    Uses Claude Code subscription instead of API calls.
    Supports structured outputs via --json-schema flag.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 600.0,
    ):
        """Initialize Claude Code provider.

        Args:
            model: Model to use (sonnet, opus, haiku). None = default.
            timeout: Subprocess timeout in seconds
        """
        self.model = model
        self.timeout = timeout

    def _build_command(self, prompt: str, model: str | None) -> list[str]:
        """Build the claude CLI command."""
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "1"]
        effective_model = model or self.model
        if effective_model:
            cmd.extend(["--model", effective_model])
        return cmd

    def _parse_response(self, stdout: str) -> str:
        """Parse JSON response and extract result."""
        import json

        response = json.loads(stdout)
        if "result" in response:
            return response["result"]
        raise ValueError(f"Unexpected response: {list(response.keys())}")

    def complete(self, prompt: str, model: str) -> str:
        """Synchronous completion using Claude Code CLI."""
        import subprocess

        cmd = self._build_command(prompt, model)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )
            return self._parse_response(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Claude Code failed: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Claude Code timed out after {self.timeout}s") from e
        except FileNotFoundError as e:
            raise RuntimeError("Claude Code CLI not found") from e

    async def complete_async(self, prompt: str, model: str) -> str:
        """Async completion using Claude Code CLI."""
        cmd = self._build_command(prompt, model)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Claude Code failed: {stderr.decode()}")
            return self._parse_response(stdout.decode())
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"Claude Code timed out after {self.timeout}s") from e
        except FileNotFoundError as e:
            raise RuntimeError("Claude Code CLI not found") from e


def is_claude_code_available() -> bool:
    """Check if Claude Code CLI is installed."""
    import shutil

    return shutil.which("claude") is not None


# Global provider instance (defaults to OpenAI for backward compatibility)
_provider: LLMProvider | None = None


def set_llm_provider(provider: LLMProvider) -> None:
    """Set the global LLM provider."""
    global _provider
    _provider = provider


def get_llm_provider() -> LLMProvider:
    """Get the global LLM provider, creating default if needed."""
    global _provider
    if _provider is None:
        _provider = OpenAIProvider()
    return _provider
