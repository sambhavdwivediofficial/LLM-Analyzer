"""
inference.py — Agent for LLM Output Quality Reviewer environment.

Environment variables (injected by validator):
  API_BASE_URL  — LLM API endpoint
  API_KEY       — LLM API key
  MODEL_NAME    — Model identifier
  HF_TOKEN      — HuggingFace API key (fallback for API_KEY)
  ENV_BASE_URL  — Environment server base URL (default: http://localhost:7860)
"""

import json
import os
import sys
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "dummy-token")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MAX_RETRIES  = 2
SUCCESS_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI content reviewer. Your job is to review AI-generated text and identify quality issues.

You MUST respond with a valid JSON object and nothing else. No explanation outside the JSON.

The JSON must have exactly these fields:
{
  "issues_found": ["list of issues — choose from: hallucination, bias, toxicity, incoherence, none"],
  "explanation": "1-3 sentence explanation of what you found",
  "severity": "one of: low, medium, high",
  "corrected_output": "a corrected version of the text, or null if not needed"
}"""

# ---------------------------------------------------------------------------
# Stdout logging — strict validator format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    # Sanitize action string — no newlines allowed on a single line
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset() -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_dict: dict) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/step", json=action_dict, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def build_user_message(observation: dict) -> str:
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
    fallback = {
        "issues_found":     ["hallucination"],
        "explanation":      "Fallback response: LLM call failed.",
        "severity":         "medium",
        "corrected_output": None,
    }

    url     = f"{API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(observation)},
        ],
        "temperature": 0.2,
        "max_tokens":  512,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp     = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data     = resp.json()
            raw_text = data["choices"][0]["message"]["content"] or ""
            clean    = raw_text.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[DEBUG] Attempt {attempt}: parse error — {exc}", flush=True)
        except Exception as exc:
            print(f"[DEBUG] Attempt {attempt}: LLM call failed — {exc}", flush=True)

    return fallback

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rewards:     list[float] = []
    step_num:    int         = 0
    done:        bool        = False
    observation: dict        = {}

    try:
        observation = env_reset()
    except Exception as exc:
        print(f"[DEBUG] env_reset failed: {exc}", flush=True)
        log_start(task="llm-quality-reviewer", env="llm-quality-reviewer", model=MODEL_NAME)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(1)

    episode_task = observation.get("task_id", "llm-quality-reviewer")
    log_start(task=episode_task, env="llm-quality-reviewer", model=MODEL_NAME)

    while not done:
        step_num += 1
        task_id   = observation.get("task_id", "unknown")

        action_dict = call_llm(observation)

        try:
            result = env_step(action_dict)
        except Exception as exc:
            print(f"[DEBUG] env_step failed: {exc}", flush=True)
            log_step(step=step_num, action=str(action_dict), reward=0.0, done=True, error=str(exc))
            rewards.append(0.0)
            break

        reward_obj  = result["reward"]
        done        = result["done"]
        observation = result["observation"]
        score       = float(reward_obj["total"])
        error_msg   = reward_obj.get("feedback", None)

        rewards.append(score)
        action_label = f"review(task={task_id},issues={action_dict.get('issues_found',[])})"

        log_step(
            step   = step_num,
            action = action_label,
            reward = score,
            done   = done,
            error  = None,
        )

    avg_score = sum(rewards) / len(rewards) if rewards else 0.0
    success   = avg_score >= SUCCESS_THRESHOLD

    log_end(success=success, steps=step_num, score=avg_score, rewards=rewards)


if __name__ == "__main__":
    main()