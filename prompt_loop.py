"""
The core "prompt engineering loop":

  1. Send the current prompt to Gemini -> get a response.
  2. Send the response to a "judge" call -> get a score (0-10) + critique.
  3. If score >= threshold or we hit max_iterations, stop.
  4. Otherwise, ask the model to rewrite the ORIGINAL PROMPT based on
     the critique, and go back to step 1.

This mirrors a real prompt-engineering workflow: write, test, critique,
revise -- just automated.
"""

import json
import re
from dataclasses import dataclass, field
from llm_client import LLMClient
from llm_config import settings


@dataclass
class Iteration:
    iteration: int
    prompt: str
    response: str
    score: float
    critique: str


@dataclass
class LoopResult:
    final_prompt: str
    final_response: str
    history: list = field(default_factory=list)


# JUDGE_INSTRUCTIONS = """You are a strict evaluator. Given a TASK and a RESPONSE,
# score the response from 0 to 10 on how well it fulfills the task, and give a
# one or two sentence critique of what's missing or could be improved. Response never be perfect.

# Reply ONLY with JSON in this exact shape, no other text:
# {{"score": <number>, "critique": "<short critique>"}}

# TASK: {task}
# RESPONSE: {response}
# """

#The below old Judge Instructions, Which is used in history.json Prompt Loop. It is commented out here to see output differences between the two Judge Instructions. The new Judge Instructions is above this comment.

JUDGE_INSTRUCTIONS = """You are a strict evaluator. Given a TASK and a RESPONSE,
score the response from 0 to 10 on how well it fulfills the task, and give a
one or two sentence critique of what's missing or could be improved.

Reply ONLY with JSON in this exact shape, no other text:
{{"score": <number>, "critique": "<short critique>"}}

TASK: {task}
RESPONSE: {response}
"""



REWRITE_INSTRUCTIONS = """You are improving a prompt for an LLM based on feedback.

ORIGINAL PROMPT: {prompt}
THE RESPONSE IT PRODUCED: {response}
CRITIQUE OF THAT RESPONSE: {critique}

Rewrite the ORIGINAL PROMPT so that following it would address the critique.
Reply with ONLY the rewritten prompt text, nothing else.
"""


class PromptLoop:
    def __init__(self, client: LLMClient = None, judge_client: LLMClient = None):
        self.client = client or LLMClient(model=settings.model)
        self.judge_client = judge_client or LLMClient(model=settings.judge_model, temperature=0.0)

    def _get_response(self, prompt: str) -> str:
        return self.client.call([{"role": "user", "content": prompt}])

    def _judge(self, task: str, response: str) -> tuple[float, str]:
        judge_prompt = JUDGE_INSTRUCTIONS.format(task=task, response=response)
        raw = self.judge_client.call([{"role": "user", "content": judge_prompt}])
        return self._parse_judge_output(raw)

    @staticmethod
    def _parse_judge_output(raw: str) -> tuple[float, str]:
        # Models sometimes wrap JSON in markdown fences -- strip those first.
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            return float(data["score"]), str(data["critique"])
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fall back gracefully instead of crashing the whole loop.
            return 0.0, f"Could not parse judge output: {raw[:200]}"

    def _rewrite_prompt(self, prompt: str, response: str, critique: str) -> str:
        rewrite_prompt = REWRITE_INSTRUCTIONS.format(
            prompt=prompt, response=response, critique=critique
        )
        return self.client.call([{"role": "user", "content": rewrite_prompt}])

    def run(self, initial_prompt: str, verbose: bool = True) -> LoopResult:
        prompt = initial_prompt
        history = []

        for i in range(1, settings.max_iterations + 1):
            if verbose:
                print(f"\n[iteration {i}/{settings.max_iterations}] calling Gemini...")
            response = self._get_response(prompt)

            if verbose:
                print(f"[iteration {i}] response received, asking judge to score it...")
            score, critique = self._judge(initial_prompt, response)

            if verbose:
                # Printed the moment we know the score -- this is the
                # line that answers "why did/didn't it stop here".
                verdict = "PASS" if score >= settings.score_threshold else "retry"
                print(f"[iteration {i}] score: {score}/10 ({verdict}) -- {critique}")

            history.append(
                Iteration(
                    iteration=i,
                    prompt=prompt,
                    response=response,
                    score=score,
                    critique=critique,
                )
            )

            if score >= settings.score_threshold:
                break
            if i < settings.max_iterations:
                if verbose:
                    print(f"[iteration {i}] score below threshold ({settings.score_threshold}), rewriting prompt...")
                prompt = self._rewrite_prompt(prompt, response, critique)

        last = history[-1]
        return LoopResult(final_prompt=last.prompt, final_response=last.response, history=history)