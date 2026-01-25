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
