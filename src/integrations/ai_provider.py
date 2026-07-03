"""Abstract AI provider interface for provider-agnostic LLM integration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol


class ChatMessage(Protocol):
    content: str


class ChatChoice(Protocol):
    message: ChatMessage


class ChatResponse(Protocol):
    choices: List[ChatChoice]


class AIProvider(ABC):
    """Abstract interface for AI providers (NVIDIA NIM, OpenAI, Anthropic, etc.)."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResponse: ...

    @abstractmethod
    async def is_available(self) -> bool: ...

    @abstractmethod
    def get_model_name(self) -> str: ...

    @abstractmethod
    async def close(self) -> None: ...


class OpenAIProvider(AIProvider):
    """Generic OpenAI-compatible provider (NVIDIA NIM, OpenAI, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        alt_model: Optional[str] = None,
        max_retries: int = 3,
    ):
        self._model = model
        self._alt_model = alt_model
        self._max_retries = max_retries
        self._client: Optional[Any] = None
        self._available = False

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=120.0,
            )
            self._available = bool(api_key) and bool(api_key.strip())
        except ImportError:
            self._available = False

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResponse:
        if not self._client:
            raise RuntimeError("OpenAI provider not initialized")

        models_to_try = [self._model]
        if self._alt_model:
            models_to_try.append(self._alt_model)

        last_error = None
        for model_name in models_to_try:
            for attempt in range(self._max_retries):
                try:
                    kwargs: Dict[str, Any] = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if response_format:
                        kwargs["response_format"] = response_format

                    return await self._client.chat.completions.create(**kwargs)
                except Exception as exc:
                    last_error = exc
                    if "response_format" in str(exc).lower() and response_format:
                        response_format = None
                        continue
                    if attempt < self._max_retries - 1:
                        import asyncio

                        delay = 1.0 * (attempt + 1)
                        await asyncio.sleep(delay)
                        continue
                    break

        raise RuntimeError(f"All models failed: {last_error}")

    async def is_available(self) -> bool:
        return self._available

    def get_model_name(self) -> str:
        return self._model

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


class ProviderRegistry:
    """Registry for managing multiple AI providers with fallback."""

    def __init__(self):
        self._providers: List[AIProvider] = []
        self._default_provider: Optional[AIProvider] = None

    def register(self, provider: AIProvider, make_default: bool = False) -> None:
        self._providers.append(provider)
        if make_default or self._default_provider is None:
            self._default_provider = provider

    @property
    def default(self) -> Optional[AIProvider]:
        return self._default_provider

    async def chat_completion_with_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResponse:
        if not self._providers:
            raise RuntimeError("No AI providers registered")

        errors: List[str] = []
        for provider in self._providers:
            try:
                if await provider.is_available():
                    return await provider.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                    )
            except Exception as exc:
                errors.append(f"{provider.get_model_name()}: {exc}")
                continue

        raise RuntimeError(f"All AI providers failed: {'; '.join(errors)}")

    async def close_all(self) -> None:
        for provider in self._providers:
            try:
                await provider.close()
            except Exception:
                pass
