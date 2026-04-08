"""
inference.py — Baseline agent for LLM Output Quality Reviewer environment.

Required environment variables:
  API_BASE_URL  — The API endpoint for the LLM (injected by validator)
  API_KEY       — The API key for the LLM (injected by validator)
  MODEL_NAME    — The model identifier to use for inference
  HF_TOKEN      — Your HuggingFace API key
"""

import json
import os
import sys

import requests

# ---------------------------------------------------------------------------
# Configuration — strictly using validator-injected variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "dummy-token")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MAX_RETRIES  = 2

# ---------------------------------------------------------------------------
# System prompt
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
    Send the observation to the LLM via validator-provided API_BASE_URL and API_KEY.
    Always makes a real API call through the validator proxy.
    """
    fallback = {
        "issues_found":     ["hallucination"],
        "explanation":      "Fallback response after failed LLM calls.",
        "severity":         "medium",
        "corrected_output": None,
    }

    try:
        from openai import OpenAI
        client       = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
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
                return json.loads(clean)

            except (json.JSONDecodeError, KeyError) as exc:
                print(f"[WARN] Attempt {attempt}: JSON parse failed — {exc}", flush=True)

            except Exception as exc:
                print(f"[WARN] Attempt {attempt}: LLM call failed — {exc}", flush=True)

    except Exception as exc:
        print(f"[WARN] OpenAI init failed — {exc}", flush=True)

    return fallback


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:

    observation = env_reset()
    episode_id  = observation.get("task_id", "episode_1")

    print(f"[START] task={episode_id} model={MODEL_NAME} env={ENV_BASE_URL}", flush=True)

    task_scores: list[dict] = []
    done     = False
    step_num = 0

    while not done:
        step_num   += 1
        task_id     = observation.get("task_id", "unknown")
        action_dict = call_llm(observation)
        result      = env_step(action_dict)

        reward      = result["reward"]
        done        = result["done"]
        info        = result["info"]
        observation = result["observation"]
        score       = reward["total"]
        difficulty  = info.get("task_difficulty", "unknown")
        feedback    = reward.get("feedback", "")

        print(
            f"[STEP] step={step_num} task={task_id} difficulty={difficulty} "
            f"score={score} done={done}",
            flush=True,
        )
        print(f"[INFO] feedback={feedback[:120]}", flush=True)

        task_scores.append({
            "task_id":    task_id,
            "difficulty": difficulty,
            "score":      score,
        })

    avg_score = (
        sum(t["score"] for t in task_scores) / len(task_scores)
        if task_scores else 0.0
    )

    print(
        f"[END] task={episode_id} score={round(avg_score, 3)} steps={step_num}",
        flush=True,
    )

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