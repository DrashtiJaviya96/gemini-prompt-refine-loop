"""
wrapper around LiteLLM.

Why wrap it at all, instead of calling litellm.completion() everywhere?
1. One place to add retries / error handling.
2. One place to swap providers later (e.g. add an OpenAI fallback)
"""

import time
import random
import litellm
from llm_config import settings

# Errors worth retrying: rate limits and transient "server overloaded"
# responses. Auth/bad-request errors are NOT in this list on purpose --
# retrying a bad API key 5 times just wastes time, it'll never succeed.
RETRYABLE_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
    litellm.Timeout,
)


class LLMClient:
    def __init__(self, model: str = None, temperature: float = None):
        self.model = model or settings.model
        self.temperature = temperature if temperature is not None else settings.temperature

    def call(self, messages: list[dict], retries: int = 4, base_wait: float = 3.0) -> str:
        """
        messages: list of {"role": "user"|"system"|"assistant", "content": str}
        Returns the assistant's text response.

        Gemini's free tier occasionally returns 503 "high demand" errors --
        these are transient, so we retry with exponential backoff + jitter
        (jitter avoids every retry landing on the same instant if you're
        running multiple calls in parallel).
        """
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    api_key=settings.api_key,
                )
                return response.choices[0].message["content"]
            except RETRYABLE_EXCEPTIONS as e:
                last_error = e
                if attempt < retries:
                    wait = base_wait * (2 ** attempt) + random.uniform(0, 1)
                    print(f"  [retry {attempt + 1}/{retries}] {type(e).__name__}, waiting {wait:.1f}s...")
                    time.sleep(wait)
            except Exception as e:
                # Non-retryable (bad API key, invalid model, etc.) -- fail fast.
                raise RuntimeError(f"LLM call failed (non-retryable): {e}") from e

        raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_error}")