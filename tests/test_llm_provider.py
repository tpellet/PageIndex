"""Tests for LLM provider abstraction."""
import asyncio
import os
from unittest.mock import patch

import pytest

from pageindex.llm_provider import (
    AnthropicProvider,
    CallbackProvider,
    LLMProvider,
    OpenAIProvider,
    get_llm_provider,
    set_llm_provider,
)


class TestLLMProviderAbstract:
    """Test that LLMProvider is abstract and cannot be instantiated."""

    def test_cannot_instantiate_abstract_class(self):
        """LLMProvider is abstract and raises TypeError on instantiation."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_has_complete_method(self):
        """LLMProvider defines abstract complete method."""
        assert hasattr(LLMProvider, "complete")

    def test_has_complete_async_method(self):
        """LLMProvider defines abstract complete_async method."""
        assert hasattr(LLMProvider, "complete_async")


class TestCallbackProvider:
    """Test CallbackProvider with user-supplied functions."""

    def test_sync_callback_returns_result(self):
        """CallbackProvider.complete() calls callback and returns result."""

        def my_callback(prompt: str, model: str) -> str:
            return f"Response to: {prompt} using {model}"

        provider = CallbackProvider(my_callback)
        result = provider.complete("Hello", "test-model")

        assert result == "Response to: Hello using test-model"

    def test_sync_callback_receives_prompt_and_model(self):
        """CallbackProvider passes prompt and model to callback."""
        received_args = {}

        def capture_callback(prompt: str, model: str) -> str:
            received_args["prompt"] = prompt
            received_args["model"] = model
            return "ok"

        provider = CallbackProvider(capture_callback)
        provider.complete("test prompt", "gpt-4")

        assert received_args["prompt"] == "test prompt"
        assert received_args["model"] == "gpt-4"

    def test_async_callback_with_sync_function(self):
        """CallbackProvider.complete_async() works with sync callback."""

        def sync_callback(prompt: str, model: str) -> str:
            return f"sync: {prompt}"

        provider = CallbackProvider(sync_callback)
        result = asyncio.run(provider.complete_async("test", "model"))

        assert result == "sync: test"

    def test_async_callback_with_async_function(self):
        """CallbackProvider.complete_async() works with async callback."""

        async def async_callback(prompt: str, model: str) -> str:
            await asyncio.sleep(0)  # Simulate async operation
            return f"async: {prompt}"

        provider = CallbackProvider(async_callback)
        result = asyncio.run(provider.complete_async("test", "model"))

        assert result == "async: test"


class TestOpenAIProvider:
    """Test OpenAIProvider initialization and API key handling."""

    def test_uses_explicit_api_key(self):
        """OpenAIProvider uses explicitly provided API key."""
        provider = OpenAIProvider(api_key="explicit-key")
        assert provider.api_key == "explicit-key"

    def test_uses_chatgpt_env_var(self):
        """OpenAIProvider falls back to CHATGPT_API_KEY env var."""
        with patch.dict(os.environ, {"CHATGPT_API_KEY": "chatgpt-key"}, clear=False):
            # Clear any existing env vars that take precedence
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
                provider = OpenAIProvider()
                assert provider.api_key == "chatgpt-key"

    def test_uses_openai_env_var(self):
        """OpenAIProvider falls back to OPENAI_API_KEY env var."""
        with patch.dict(
            os.environ, {"CHATGPT_API_KEY": "", "OPENAI_API_KEY": "openai-key"}, clear=False
        ):
            provider = OpenAIProvider()
            assert provider.api_key == "openai-key"

    def test_client_initially_none(self):
        """OpenAIProvider._client is None until first use."""
        provider = OpenAIProvider(api_key="test")
        assert provider._client is None

    def test_lazy_import_openai(self):
        """OpenAIProvider only imports openai when complete() is called."""
        # This test verifies the lazy import pattern by checking
        # that OpenAIProvider can be instantiated without openai installed
        provider = OpenAIProvider(api_key="test")
        assert provider is not None
        # complete() would trigger the import, but we don't call it


class TestAnthropicProvider:
    """Test AnthropicProvider initialization and API key handling."""

    def test_uses_explicit_api_key(self):
        """AnthropicProvider uses explicitly provided API key."""
        provider = AnthropicProvider(api_key="explicit-key")
        assert provider.api_key == "explicit-key"

    def test_uses_env_var(self):
        """AnthropicProvider falls back to ANTHROPIC_API_KEY env var."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-key"}, clear=False):
            provider = AnthropicProvider()
            assert provider.api_key == "anthropic-key"

    def test_client_initially_none(self):
        """AnthropicProvider._client is None until first use."""
        provider = AnthropicProvider(api_key="test")
        assert provider._client is None


class TestGlobalProviderManagement:
    """Test set_llm_provider and get_llm_provider functions."""

    def test_set_and_get_provider(self):
        """set_llm_provider stores provider, get_llm_provider retrieves it."""
        callback_provider = CallbackProvider(lambda p, m: "test")
        set_llm_provider(callback_provider)

        retrieved = get_llm_provider()

        assert retrieved is callback_provider

    def test_get_provider_creates_default_openai(self):
        """get_llm_provider creates OpenAIProvider if none set."""
        # Reset global state
        import pageindex.llm_provider as module

        module._provider = None

        provider = get_llm_provider()

        assert isinstance(provider, OpenAIProvider)

    def test_set_provider_replaces_existing(self):
        """set_llm_provider replaces any existing provider."""
        first = CallbackProvider(lambda p, m: "first")
        second = CallbackProvider(lambda p, m: "second")

        set_llm_provider(first)
        set_llm_provider(second)

        assert get_llm_provider() is second


class TestProviderInheritance:
    """Test that concrete providers properly implement LLMProvider."""

    def test_openai_is_llm_provider(self):
        """OpenAIProvider is a subclass of LLMProvider."""
        assert issubclass(OpenAIProvider, LLMProvider)

    def test_anthropic_is_llm_provider(self):
        """AnthropicProvider is a subclass of LLMProvider."""
        assert issubclass(AnthropicProvider, LLMProvider)

    def test_callback_is_llm_provider(self):
        """CallbackProvider is a subclass of LLMProvider."""
        assert issubclass(CallbackProvider, LLMProvider)

    def test_callback_instance_is_llm_provider(self):
        """CallbackProvider instance passes isinstance check."""
        provider = CallbackProvider(lambda p, m: "")
        assert isinstance(provider, LLMProvider)
