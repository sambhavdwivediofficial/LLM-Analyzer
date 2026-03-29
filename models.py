"""
models.py — Typed data models for the LLM Output Quality Reviewer environment.

These Pydantic models define exactly what the agent sees (Observation),
what it can do (Action), and what score it receives (Reward).
Keeping them in one place makes the whole codebase easy to understand.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation — what the agent receives after every step
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Everything the agent can see about the current state."""

    task_id: str = Field(
        description="Unique identifier for the current task, e.g. 'task_easy_001'."
    )
    task_description: str = Field(
        description="Plain-English instructions telling the agent what to do."
    )
    llm_output: str = Field(
        description="The AI-generated text that the agent must review and judge."
    )
    reference_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Ground-truth facts the agent can use to spot hallucinations. "
            "Empty list means no reference is provided (harder tasks)."
        ),
    )
    step_number: int = Field(
        default=0,
        description="How many steps have been taken in this episode so far."
    )
    previous_feedback: Optional[str] = Field(
        default=None,
        description="Feedback from the last action, so the agent can self-correct."
    )


# ---------------------------------------------------------------------------
# Action — what the agent submits as its review decision
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """The agent's review verdict for the current LLM output."""

    issues_found: list[str] = Field(
        description=(
            "List of issue types the agent detected. "
            "Valid values: 'hallucination', 'bias', 'toxicity', "
            "'incoherence', 'none'."
        )
    )
    explanation: str = Field(
        description="Short explanation (1-3 sentences) justifying the verdict."
    )
    severity: str = Field(
        default="low",
        description="Overall severity of the issues: 'low', 'medium', or 'high'."
    )
    corrected_output: Optional[str] = Field(
        default=None,
        description=(
            "Optional: a corrected version of the text. "
            "Providing a good correction gives bonus reward on hard tasks."
        ),
    )


# ---------------------------------------------------------------------------
# Reward — structured score returned after each step
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Detailed breakdown of the reward signal for transparency."""

    total: float = Field(
        description="Final combined score for this step, in range [0.0, 1.0]."
    )
    detection_score: float = Field(
        description="How accurately the agent identified the correct issues (0–0.6)."
    )
    explanation_score: float = Field(
        description="Quality of the written explanation (0–0.2)."
    )
    severity_score: float = Field(
        description="Whether the severity label matches the actual severity (0–0.1)."
    )
    correction_bonus: float = Field(
        default=0.0,
        description="Bonus for providing a high-quality corrected output (0–0.1)."
    )
    feedback: str = Field(
        description="Human-readable feedback explaining why this score was given."
    )


# ---------------------------------------------------------------------------
# StepResult — bundles everything the environment returns from step()
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Full return value of env.step(action)."""

    observation: Observation
    reward: Reward
    done: bool = Field(description="True when the episode has ended.")
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra diagnostics — task name, episode id, etc."
    )


# ---------------------------------------------------------------------------
# State — internal snapshot used by state()
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Complete internal state of the environment (used by state() endpoint)."""

    current_task_index: int = Field(default=0)
    current_step: int = Field(default=0)
    episode_rewards: list[float] = Field(default_factory=list)
    is_done: bool = Field(default=False)
    task_id: Optional[str] = Field(default=None)