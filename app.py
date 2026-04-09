"""
app.py — FastAPI server exposing the OpenEnv HTTP interface.

Endpoints:
  POST /reset   → start a new episode, get first observation
  POST /step    → submit an action, get reward + next observation
  GET  /state   → inspect current environment state
  GET  /health  → liveness check
  GET  /tasks   → list all tasks with grader info (required by validator)
  POST /grader  → score an action without running a full episode (required by validator)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import Action, EnvironmentState, Observation, StepResult
from env import LLMQualityReviewerEnv
from tasks import ALL_TASKS
from graders import run_grader, GRADER_REGISTRY

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Analyzer Environment API",
    description=(
        "An OpenEnv-compatible environment where an AI agent reviews "
        "LLM-generated text for hallucinations, bias, toxicity, and incoherence."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = LLMQualityReviewerEnv()


# ---------------------------------------------------------------------------
# Request model for /grader endpoint
# ---------------------------------------------------------------------------

class GraderRequest(BaseModel):
    task_id: str
    action: Action


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    """Liveness probe."""
    return {"status": "ok", "environment": "LLMQualityReviewer", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
def reset() -> Observation:
    """Start a new episode. Returns the first observation (easy task)."""
    return _env.reset()


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """Submit a review action. Returns reward + next observation."""
    try:
        return _env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    """Return current internal environment state."""
    return _env.state()


@app.get("/tasks")
def list_tasks() -> dict:
    """
    List all tasks with their grader functions.
    This endpoint is required by the OpenEnv validator to detect registered graders.
    """
    task_list = []
    for task in ALL_TASKS:
        grader_fn = GRADER_REGISTRY.get(task.task_id)
        task_list.append({
            "task_id":        task.task_id,
            "difficulty":     task.difficulty,
            "description":    task.description,
            "max_score":      1.0,
            "grader":         grader_fn.__name__ if grader_fn else None,
            "grader_module":  "graders",
            "has_grader":     grader_fn is not None,
        })
    return {
        "tasks":         task_list,
        "total":         len(task_list),
        "grader_module": "graders",
        "registry":      "GRADER_REGISTRY",
    }


@app.post("/grader")
def grade_action(request: GraderRequest) -> dict:
    """
    Score an action for a given task without running a full episode.
    Required by the OpenEnv validator to verify graders are functional.
    """
    task = next((t for t in ALL_TASKS if t.task_id == request.task_id), None)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{request.task_id}' not found. "
                   f"Valid task_ids: {[t.task_id for t in ALL_TASKS]}",
        )
    try:
        reward = run_grader(task, request.action)
        return {
            "task_id":          request.task_id,
            "total":            reward.total,
            "detection_score":  reward.detection_score,
            "explanation_score":reward.explanation_score,
            "severity_score":   reward.severity_score,
            "correction_bonus": reward.correction_bonus,
            "feedback":         reward.feedback,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc