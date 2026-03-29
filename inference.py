"""
inference.py — Baseline agent that runs against the LLM Quality Reviewer environment.

This script:
  1. Calls the environment's /reset endpoint to start an episode.
  2. For each task, sends the observation to an LLM (via OpenAI client).
  3. Parses the LLM's response into a structured Action.
  4. Calls /step with the action and records the reward.
  5. Prints a final score summary for all 3 tasks.

Required environment variables:
  API_BASE_URL  — e.g. https://router.huggingface.co/v1
  MODEL_NAME    — e.g. meta-llama/Llama-3.1-8B-Instruct
  HF_TOKEN      — your HuggingFace API token

Usage:
  python inference.py
"""

import json
import os
import sys

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

# Where our FastAPI environment server is running
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# How many times to retry a failed LLM call before giving up
MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# OpenAI client (pointed at HuggingFace router)
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# System prompt — tells the LLM exactly what format to return
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert AI content reviewer. Your job is to review AI-generated text
and identify quality issues.

You MUST respond with a valid JSON object and nothing else. No explanation outside the JSON.

The JSON must have exactly these fields:
{
  "issues_found": ["list of issues — choose from: hallucination, bias, toxicity, incoherence, none"],
  "explanation": "1-3 sentence explanation of what you found",
  "severity": "one of: low, medium, high",
  "corrected_output": "a corrected version of the text, or null if not needed"
}
""".strip()


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset() -> dict:
    """Call /reset on the environment server."""
    resp = requests.post(f"{ENV_BASE_URL}/reset", timeout=10)
    resp.raise_for_status()
    return resp.json()


def env_step(action_dict: dict) -> dict:
    """Call /step with the agent's action."""
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action_dict,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM call + response parsing
# ---------------------------------------------------------------------------

def build_user_message(observation: dict) -> str:
    """Turn an observation dict into a plain-text prompt for the LLM."""
    lines = [
        f"TASK: {observation['task_description']}",
        "",
        "TEXT TO REVIEW:",
        observation["llm_output"],
    ]

    if observation.get("reference_facts"):
        lines += ["", "REFERENCE FACTS (ground truth):"]
        for fact in observation["reference_facts"]:
            lines.append(f"  - {fact}")

    if observation.get("previous_feedback"):
        lines += ["", f"PREVIOUS FEEDBACK: {observation['previous_feedback']}"]

    return "\n".join(lines)


def call_llm(observation: dict) -> dict:
    """
    Send the observation to the LLM and parse its JSON response.
    Falls back to a safe default action if parsing fails.
    """
    user_message = build_user_message(observation)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.2,       # low temperature for deterministic reviews
                max_tokens=512,
                stream=False,
            )
            raw_text = completion.choices[0].message.content or ""

            # Strip markdown code fences if the model added them
            clean = raw_text.strip().strip("```json").strip("```").strip()
            parsed = json.loads(clean)
            return parsed

        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  [Attempt {attempt}] JSON parse failed: {exc}. Retrying...")

        except Exception as exc:  # noqa: BLE001
            print(f"  [Attempt {attempt}] LLM call failed: {exc}. Retrying...")

    # All retries exhausted — return a safe fallback action
    print("  All retries failed. Using fallback action.")
    return {
        "issues_found":      ["none"],
        "explanation":       "Could not parse model response.",
        "severity":          "low",
        "corrected_output":  None,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  LLM Output Quality Reviewer — Baseline Inference")
    print("=" * 60)
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Env URL  : {ENV_BASE_URL}")
    print("=" * 60)

    # Validate that required env vars are set
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    # Start a new episode
    observation = env_reset()
    print(f"\nEpisode started. First task: {observation['task_id']}\n")

    task_scores: list[dict] = []
    done = False
    step_num = 0

    while not done:
        step_num += 1
        task_id     = observation["task_id"]
        difficulty  = "(see task_id)"

        print(f"--- Step {step_num} | Task: {task_id} ---")

        # Ask the LLM to review the current text
        action_dict = call_llm(observation)
        print(f"  Issues found : {action_dict.get('issues_found')}")
        print(f"  Severity     : {action_dict.get('severity')}")

        # Submit the action to the environment
        result = env_step(action_dict)

        reward     = result["reward"]
        done       = result["done"]
        info       = result["info"]
        observation = result["observation"]

        task_scores.append({
            "task_id":    task_id,
            "difficulty": info.get("task_difficulty", "?"),
            "score":      reward["total"],
            "feedback":   reward["feedback"],
        })

        print(f"  Score        : {reward['total']:.3f}")
        print(f"  Feedback     : {reward['feedback'][:80]}...")
        print()

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    total_score = 0.0
    for entry in task_scores:
        print(f"  [{entry['difficulty'].upper():6}] {entry['task_id']}: {entry['score']:.3f}")
        total_score += entry["score"]

    avg_score = total_score / len(task_scores) if task_scores else 0.0
    print(f"\n  Average score across all tasks: {avg_score:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()