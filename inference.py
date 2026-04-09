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

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "dummy-token")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MAX_RETRIES = 2

SYSTEM_PROMPT = """You are an expert AI content reviewer. Your job is to review AI-generated text and identify quality issues.

You MUST respond with a valid JSON object and nothing else. No explanation outside the JSON.

The JSON must have exactly these fields:
{
  "issues_found": ["list of issues — choose from: hallucination, bias, toxicity, incoherence, none"],
  "explanation": "1-3 sentence explanation of what you found",
  "severity": "one of: low, medium, high",
  "corrected_output": "a corrected version of the text, or null if not needed"
}"""


def env_reset() -> dict:
    """Start a new episode with the environment server."""
    try:
        resp = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[ERROR] env_reset failed: {e}", flush=True)
        sys.exit(1)


def env_step(action_dict: dict) -> dict:
    """Submit action to environment server."""
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


def build_user_message(observation: dict) -> str:
    """Build the user prompt from observation."""
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


def call_llm_direct(observation: dict) -> dict:
    """Make HTTP request to LLM API."""
    fallback = {
        "issues_found": ["hallucination"],
        "explanation": "Fallback: could not get LLM response.",
        "severity": "medium",
        "corrected_output": None,
    }

    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(observation)},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"] or ""
            clean = raw_text.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[WARN] Attempt {attempt}: parse failed — {exc}", flush=True)
        except Exception as exc:
            print(f"[WARN] Attempt {attempt}: LLM call failed — {exc}", flush=True)

    return fallback


def main() -> None:
    """Main inference loop."""
    observation = env_reset()
    
    print(f"[START] tasks=3 model={MODEL_NAME}", flush=True)

    task_scores: list[dict] = []
    done = False
    step_num = 0

    while not done:
        step_num += 1
        task_id = observation.get("task_id", "unknown")
        
        action_dict = call_llm_direct(observation)
        result = env_step(action_dict)

        reward = result["reward"]
        done = result["done"]
        info = result["info"]
        observation = result["observation"]
        score = reward["total"]
        difficulty = info.get("task_difficulty", "unknown")

        done_str = "true" if done else "false"
        print(
            f"[STEP] step={step_num} task={task_id} difficulty={difficulty} "
            f"score={score:.3f} done={done_str}",
            flush=True,
        )

        task_scores.append({
            "task_id": task_id,
            "difficulty": difficulty,
            "score": score,
        })

    avg_score = (
        sum(t["score"] for t in task_scores) / len(task_scores)
        if task_scores else 0.0
    )

    print(
        f"[END] tasks_completed={len(task_scores)} avg_score={avg_score:.3f} "
        f"steps={step_num}",
        flush=True,
    )


if __name__ == "__main__":
    main()