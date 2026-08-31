"""
Tests for the prompt loop, using a fake LLMClient so we never hit the
real Gemini API in CI. This is what shows reviewers you understand
testable design (dependency injection of the client).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompt_loop import PromptLoop, PromptLoop as _PL  # noqa
from prompt_loop import PromptLoop


class FakeClient:
    """Stands in for LLMClient. Returns scripted responses in order."""

    def __init__(self, scripted_responses):
        self.scripted_responses = list(scripted_responses)
        self.calls = []

    def call(self, messages, retries=2):
        self.calls.append(messages)
        return self.scripted_responses.pop(0)


def test_loop_stops_when_score_meets_threshold():
    # response, then judge JSON with a high score -> should stop at iteration 1
    main_client = FakeClient(["a great tagline"])
    judge_client = FakeClient(['{"score": 9, "critique": "excellent"}'])

    loop = PromptLoop(client=main_client, judge_client=judge_client)
    result = loop.run("Write a tagline")

    assert len(result.history) == 1
    assert result.history[0].score == 9
    assert result.final_response == "a great tagline"


def test_loop_rewrites_prompt_on_low_score():
    # iteration 1: low score -> rewrite -> iteration 2: high score -> stop
    main_client = FakeClient(
        ["mediocre response", "rewritten prompt text", "much better response"]
    )
    judge_client = FakeClient(
        ['{"score": 3, "critique": "too vague"}', '{"score": 9, "critique": "great"}']
    )

    loop = PromptLoop(client=main_client, judge_client=judge_client)
    result = loop.run("Write a tagline")

    assert len(result.history) == 2
    assert result.history[0].score == 3
    assert result.history[1].score == 9
    assert result.final_response == "much better response"


def test_loop_stops_at_max_iterations_even_if_score_low():
    from llm_config import settings

    original_max = settings.max_iterations
    settings.max_iterations = 2
    try:
        main_client = FakeClient(["r1", "rewritten", "r2"])
        judge_client = FakeClient(
            ['{"score": 2, "critique": "bad"}', '{"score": 4, "critique": "still bad"}']
        )
        loop = PromptLoop(client=main_client, judge_client=judge_client)
        result = loop.run("Write a tagline")

        assert len(result.history) == 2
        assert result.final_response == "r2"
    finally:
        settings.max_iterations = original_max


def test_judge_output_parsing_handles_markdown_fences():
    score, critique = PromptLoop._parse_judge_output(
        '```json\n{"score": 7.5, "critique": "decent"}\n```'
    )
    assert score == 7.5
    assert critique == "decent"


def test_judge_output_parsing_falls_back_gracefully_on_bad_json():
    score, critique = PromptLoop._parse_judge_output("not json at all")
    assert score == 0.0
    assert "Could not parse" in critique