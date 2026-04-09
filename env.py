"""
env.py — Core OpenEnv-compatible environment: LLM Output Quality Reviewer.

This environment simulates the real-world task of reviewing AI-generated text
for quality issues such as hallucinations, bias, toxicity, and incoherence.

The agent receives a piece of AI-generated text, reviews it, and submits
a structured verdict. A deterministic grader scores the verdict and returns
a reward in [0.0, 1.0].

OpenEnv interface implemented:
  - reset()  → returns initial Observation
  - step()   → returns StepResult (Observation, Reward, done, info)
  - state()  → returns EnvironmentState
"""

import uuid
from typing import Any

from models import Action, EnvironmentState, Observation, Reward, StepResult
from tasks import ALL_TASKS, Task
from graders import run_grader, GRADER_REGISTRY


MAX_STEPS_PER_EPISODE = 3


class LLMQualityReviewerEnv:
    """
    OpenEnv environment for reviewing LLM-generated text quality.

    One episode = agent reviews all 3 tasks in sequence (easy → hard).
    Each step = agent submits a review for one task, receives a reward,
    then moves to the next task until all tasks are done.
    
    Phase 2 Requirement: Validates that exactly 3 tasks have graders enabled.
    """

    def __init__(self) -> None:
        self._state = EnvironmentState()
        self._episode_id: str = ""
        self._tasks: list[Task] = ALL_TASKS
        
        # Phase 2: Validate grader registry
        self._validate_grader_registry()

    def _validate_grader_registry(self) -> None:
        """Ensure all 3 tasks are registered in the grader registry."""
        registered_tasks = set(GRADER_REGISTRY.keys())
        required_tasks = {task.task_id for task in self._tasks}
        
        if not required_tasks.issubset(registered_tasks):
            missing = required_tasks - registered_tasks
            raise RuntimeError(
                f"Grader validation failed. Missing graders for tasks: {missing}. "
                f"Registered: {list(registered_tasks)}"
            )

    def reset(self) -> Observation:
        """Reset the environment to the beginning of a new episode."""
        self._episode_id = str(uuid.uuid4())[:8]
        self._state = EnvironmentState(
            current_task_index=0,
            current_step=0,
            episode_rewards=[],
            is_done=False,
            task_id=self._tasks[0].task_id,
        )
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Process the agent's review action for the current task.

        Returns:
          StepResult containing observation, reward, done flag, and info.
        """
        if self._state.is_done:
            return self._terminal_step_result("Episode already finished.")

        current_task = self._current_task()
        reward: Reward = run_grader(current_task, action)

        self._state.episode_rewards.append(reward.total)
        self._state.current_step += 1

        next_index = self._state.current_task_index + 1
        done = next_index >= len(self._tasks)

        if done:
            self._state.is_done = True
            next_obs = self._build_terminal_observation(reward.feedback)
        else:
            self._state.current_task_index = next_index
            self._state.task_id = self._tasks[next_index].task_id
            next_obs = self._build_observation(previous_feedback=reward.feedback)

        cumulative = round(sum(self._state.episode_rewards), 3)

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info={
                "episode_id": self._episode_id,
                "task_id": current_task.task_id,
                "task_difficulty": current_task.difficulty,
                "step": self._state.current_step,
                "cumulative_reward": cumulative,
            },
        )

    def state(self) -> EnvironmentState:
        """Return a snapshot of the current environment state."""
        return self._state.model_copy()

    def _current_task(self) -> Task:
        return self._tasks[self._state.current_task_index]

    def _build_observation(self, previous_feedback: str | None = None) -> Observation:
        """Build the Observation the agent sees for the current task."""
        task = self._current_task()
        return Observation(
            task_id=task.task_id,
            task_description=task.description,
            llm_output=task.llm_output,
            reference_facts=task.reference_facts,
            step_number=self._state.current_step,
            previous_feedback=previous_feedback,
        )

    def _build_terminal_observation(self, final_feedback: str) -> Observation:
        """Build the terminal observation after all tasks are complete."""
        return Observation(
            task_id="terminal",
            task_description="Episode complete. No more tasks.",
            llm_output="",
            reference_facts=[],
            step_number=self._state.current_step,
            previous_feedback=final_feedback,
        )

    def _terminal_step_result(self, reason: str) -> StepResult:
        """Return a terminal step result when episode is finished."""
        return StepResult(
            observation=self._build_terminal_observation(reason),
            reward=Reward(
                total=0.0,
                detection_score=0.0,
                explanation_score=0.0,
                severity_score=0.0,
                correction_bonus=0.0,
                feedback=reason,
            ),
            done=True,
            info={"episode_id": self._episode_id, "note": reason},
        )