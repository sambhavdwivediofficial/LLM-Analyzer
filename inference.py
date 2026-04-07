import json
import os
import sys

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "dummy-token")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MAX_RETRIES  = 2

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


def env_reset() -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_dict: dict) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/step", json=action_dict, timeout=30)
    resp.raise_for_status()
    return resp.json()


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


def call_llm(client: OpenAI, observation: dict) -> dict:
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
            clean = raw_text.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except Exception:
            pass
    return {
        "issues_found":     ["none"],
        "explanation":      "Could not parse model response.",
        "severity":         "low",
        "corrected_output": None,
    }


def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(json.dumps({"event": "[ERROR]", "message": str(e)}))
        sys.exit(1)

    try:
        observation = env_reset()
    except Exception as e:
        print(json.dumps({"event": "[ERROR]", "message": f"env_reset failed: {str(e)}"}))
        sys.exit(1)

    episode_id = observation.get("task_id", "episode_1")

    print(json.dumps({
        "event":      "[START]",
        "episode_id": episode_id,
        "model":      MODEL_NAME,
        "env_url":    ENV_BASE_URL,
    }))

    task_scores = []
    done        = False
    step_num    = 0

    while not done:
        step_num += 1
        task_id   = observation.get("task_id", "unknown")

        try:
            action_dict = call_llm(client, observation)
        except Exception:
            action_dict = {
                "issues_found":     ["none"],
                "explanation":      "Fallback action.",
                "severity":         "low",
                "corrected_output": None,
            }

        try:
            result = env_step(action_dict)
        except Exception as e:
            print(json.dumps({"event": "[ERROR]", "message": f"env_step failed: {str(e)}"}))
            break

        reward      = result["reward"]
        done        = result["done"]
        info        = result["info"]
        observation = result["observation"]

        print(json.dumps({
            "event":      "[STEP]",
            "step":       step_num,
            "task_id":    task_id,
            "difficulty": info.get("task_difficulty", "unknown"),
            "action":     action_dict,
            "score":      reward["total"],
            "feedback":   reward["feedback"],
            "done":       done,
        }))

        task_scores.append({
            "task_id":    task_id,
            "difficulty": info.get("task_difficulty", "unknown"),
            "score":      reward["total"],
        })

    avg_score = sum(t["score"] for t in task_scores) / len(task_scores) if task_scores else 0.0

    print(json.dumps({
        "event":       "[END]",
        "total_steps": step_num,
        "task_scores": task_scores,
        "avg_score":   round(avg_score, 3),
    }))


if __name__ == "__main__":
    main()