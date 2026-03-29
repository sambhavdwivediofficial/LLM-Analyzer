"""
app.py — FastAPI server that exposes the OpenEnv HTTP interface.

Endpoints:
  POST /reset  → start a new episode, get first observation
  POST /step   → submit an action, get reward + next observation
  GET  /state  → inspect current environment state
  GET  /health → quick liveness check (used by HuggingFace ping)

The server is intentionally stateless between HTTP calls except for
a single in-memory environment instance. This is fine for evaluation;
production deployments would use a session store.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import Action, EnvironmentState, Observation, StepResult
from env import LLMQualityReviewerEnv

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

# Allow cross-origin requests so HuggingFace Spaces iframe can reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance for this process
_env = LLMQualityReviewerEnv()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    """Liveness probe — returns 200 if the server is running."""
    return {"status": "ok", "environment": "LLMQualityReviewer", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
def reset() -> Observation:
    """
    Start a new episode.
    Returns the first observation (easy hallucination task).
    """
    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """
    Submit the agent's review action for the current task.
    Returns the reward, next observation, and done flag.
    """
    try:
        result = _env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    """Return the current internal state of the environment."""
    return _env.state()