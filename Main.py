"""
CLI entry point.
"""

import argparse
import json
from prompt_loop import PromptLoop


def main():
    parser = argparse.ArgumentParser(description="Gemini prompt-refinement loop")
    parser.add_argument("prompt", type=str, help="The initial task/prompt")
    parser.add_argument(
        "--log-file", type=str, default=None, help="Optional path to save iteration history as JSON"
    )
    args = parser.parse_args()

    loop = PromptLoop()
    result = loop.run(args.prompt)

    print("\n=== Iteration history ===")
    for step in result.history:
        print(f"\n--- Iteration {step.iteration} (score: {step.score}/10) ---")
        print(f"Prompt: {step.prompt}")
        print(f"Response: {step.response}")
        print(f"Critique: {step.critique}")

    print("\n=== Final result ===")
    print(f"Prompt: {result.final_prompt}")
    print(f"Response: {result.final_response}")

    if args.log_file:
        with open(args.log_file, "w") as f:
            json.dump([step.__dict__ for step in result.history], f, indent=2)
        print(f"\nSaved full history to {args.log_file}")


if __name__ == "__main__":
    main()