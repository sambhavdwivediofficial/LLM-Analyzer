"""
inference.py — Baseline agent for LLM Output Quality Reviewer environment.

This script:
  1. Calls the environment's /reset endpoint to start an episode.
  2. For each task, sends the observation to an LLM (via OpenAI client).
  3. Parses the LLM's response into a structured Action.
  4. Calls /step with the action and records the reward.
  5. Prints structured [START], [STEP], [END] logs to stdout.
  6. Prints a final score summary for all 3 tasks.

Required environment variables:
  API_BASE_URL  — e.g. https://router.huggingface.co/v1
  MODEL_NAME    — e.g. meta-llama/Llama-3.1-8B-Instruct
  HF_TOKEN      — your HuggingFace API token (no default)

Usage:
  python inference.py
"""

import json
import os
import sys

import requests

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MAX_RETRIES  = 2

# ---------------------------------------------------------------------------
# System prompt — tells the LLM exactly what format to return
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI content reviewer. Your job is to review AI-generated text
and identify quality issues.

You MUST respond with a valid JSON object and nothing else. No explanation outside the JSON.

The JSON must have exactly these fields:
{
  "issues_found": ["list of issues — choose from: hallucination, bias, toxicity, incoherence, none"],
  "explanation": "1-3 sentence explanation of what you found",
  "severity": "one of: low, medium, high",
  "corrected_output": "a corrected version of the text, or null if not needed"
}"""


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset() -> dict:
    """Call /reset on the environment server to start a new episode."""
    try:
        resp = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] env_reset failed: {e}", flush=True)
        sys.exit(1)


def env_step(action_dict: dict) -> dict:
    """Call /step with the agent's action and return the result."""
    try:
        resp = requests.post(
            f"{ENV_BASE_URL}/step",
            json=action_dict,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] env_step failed: {e}", flush=True)
        sys.exit(1)


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
    Falls back to a safe default action if anything fails.
    OpenAI client is imported and initialized inside to avoid global crash.
    """
    fallback = {
        "issues_found":     ["none"],
        "explanation":      "Could not get a valid model response. Using fallback.",
        "severity":         "low",
        "corrected_output": None,
    }

    try:
        from openai import OpenAI

        token  = HF_TOKEN if HF_TOKEN else "dummy-token"
        client = OpenAI(base_url=API_BASE_URL, api_key=token)

        user_message = build_user_message(observation)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0.2,
                    max_tokens=512,
                    stream=False,
                )
                raw_text = completion.choices[0].message.content or ""
                clean    = raw_text.strip().strip("```json").strip("```").strip()
                parsed   = json.loads(clean)
                return parsed

            except (json.JSONDecodeError, KeyError) as exc:
                print(f"[WARN] Attempt {attempt}: JSON parse failed — {exc}", flush=True)

            except Exception as exc:
                print(f"[WARN] Attempt {attempt}: LLM call failed — {exc}", flush=True)

    except Exception as exc:
        print(f"[WARN] OpenAI client init failed — {exc}. Using fallback.", flush=True)

    return fallback


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:

    # Start a new episode
    observation = env_reset()
    episode_id  = observation.get("task_id", "episode_1")

    # [START] log — required by validator
    print(f"[START] task={episode_id} model={MODEL_NAME} env={ENV_BASE_URL}", flush=True)

    task_scores: list[dict] = []
    done     = False
    step_num = 0

    while not done:
        step_num  += 1
        task_id    = observation.get("task_id", "unknown")

        # Ask the LLM to review the current text
        action_dict = call_llm(observation)

        # Submit the action to the environment
        result      = env_step(action_dict)

        reward      = result["reward"]
        done        = result["done"]
        info        = result["info"]
        observation = result["observation"]
        score       = reward["total"]
        difficulty  = info.get("task_difficulty", "unknown")
        feedback    = reward.get("feedback", "")

        # [STEP] log — required by validator
        print(
            f"[STEP] step={step_num} task={task_id} difficulty={difficulty} "
            f"score={score} done={done}",
            flush=True,
        )

        # Detailed feedback for debugging
        print(f"[INFO] feedback={feedback[:120]}", flush=True)

        task_scores.append({
            "task_id":    task_id,
            "difficulty": difficulty,
            "score":      score,
        })

    # Final summary
    avg_score = (
        sum(t["score"] for t in task_scores) / len(task_scores)
        if task_scores else 0.0
    )

    # [END] log — required by validator
    print(
        f"[END] task={episode_id} score={round(avg_score, 3)} steps={step_num}",
        flush=True,
    )

    # Human-readable summary
    print("\n" + "=" * 60, flush=True)
    print("  FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    for entry in task_scores:
        print(
            f"  [{entry['difficulty'].upper():6}] {entry['task_id']}: {entry['score']:.3f}",
            flush=True,
        )
    print(f"\n  Average score: {avg_score:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()