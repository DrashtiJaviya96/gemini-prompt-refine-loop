"""
Central configuration for the prompt-refinement loop.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables


@dataclass
class Settings:
    # LiteLLM model string. The "gemini/" prefix tells LiteLLM which
    # provider adapter to use under the hood.
    model: str = "gemini/gemini-3.1-flash-lite-preview"  # free-tier Gemini model

    # Same idea, but for the "judge" call that critiques/scores output.
    # Using the same model keeps this free-tier friendly; swap to a
    # bigger model if you want a stronger judge.
    judge_model: str = "gemini/gemini-3.1-flash-lite-preview"

    temperature: float = 0.7 # how random/creative vs. predictable.Scale is roughly 0 to 1 (sometimes up to 2 depending on provider).
    max_iterations: int = 4

    # Loop stops early once the judge's score reaches this (0-10 scale)
    score_threshold: float = 8.0

    api_key: str = os.getenv("GEMINI_API_KEY", "")


settings = Settings()