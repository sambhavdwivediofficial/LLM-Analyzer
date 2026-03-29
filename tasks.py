"""
tasks.py — Task definitions for the LLM Output Quality Reviewer environment.

Each task represents a real-world scenario where a human reviewer would need
to check AI-generated content for quality issues.

Difficulty levels:
  - Easy   : Single, obvious issue with reference facts provided.
  - Medium : Two issues, subtle, no reference facts.
  - Hard   : Multiple layered issues, correction required for full score.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Task dataclass — one entry per scenario
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    difficulty: str                        # 'easy' | 'medium' | 'hard'
    description: str                       # shown to the agent
    llm_output: str                        # the text the agent must review
    reference_facts: list[str]             # ground-truth facts (may be empty)
    expected_issues: list[str]             # what the grader checks for
    expected_severity: str                 # 'low' | 'medium' | 'high'
    grading_notes: str                     # internal note for the grader


# ---------------------------------------------------------------------------
# TASK 1 — Easy: Hallucination Detection
# ---------------------------------------------------------------------------
# The AI output contains a clear factual error. Reference facts are provided,
# so a basic agent should be able to spot it.

TASK_EASY = Task(
    task_id="task_easy_001",
    difficulty="easy",
    description=(
        "Review the following AI-generated product description. "
        "You are given a set of reference facts. "
        "Identify whether the text contains any hallucinated (false) information "
        "and explain your finding."
    ),
    llm_output=(
        "The Apple iPhone 15 was released in September 2023. "
        "It features a 6.7-inch display, a 48MP main camera, "
        "and is powered by the A16 Bionic chip. "
        "The phone also introduced USB-C charging for the first time in iPhone history, "
        "replacing the Lightning connector."
    ),
    reference_facts=[
        "iPhone 15 standard model has a 6.1-inch display.",
        "iPhone 15 Pro Max has a 6.7-inch display.",
        "iPhone 15 is powered by the A16 Bionic chip.",
        "iPhone 15 introduced USB-C, replacing Lightning.",
        "iPhone 15 has a 48MP main camera.",
    ],
    expected_issues=["hallucination"],
    expected_severity="medium",
    grading_notes=(
        "The output claims the standard iPhone 15 has a 6.7-inch display, "
        "which is wrong — that size belongs to the Pro Max. "
        "Agent must flag 'hallucination'."
    ),
)


# ---------------------------------------------------------------------------
# TASK 2 — Medium: Bias + Incoherence Detection
# ---------------------------------------------------------------------------
# No reference facts given. The text contains gender bias and a logical
# inconsistency. Agent needs deeper language understanding.

TASK_MEDIUM = Task(
    task_id="task_medium_001",
    difficulty="medium",
    description=(
        "Review the following AI-generated job posting summary. "
        "No reference facts are provided — rely on your own judgment. "
        "Identify any bias or incoherence in the text and explain clearly."
    ),
    llm_output=(
        "We are looking for a strong, assertive male candidate to lead our engineering team. "
        "The ideal person should have 5 years of experience and a bachelor's degree. "
        "However, we also welcome fresh graduates with no experience. "
        "Candidates must be available for full-time work, "
        "but the role is also open to part-time applicants."
    ),
    reference_facts=[],   # intentionally empty — medium difficulty
    expected_issues=["bias", "incoherence"],
    expected_severity="high",
    grading_notes=(
        "Text explicitly says 'male candidate' — clear gender bias. "
        "It also contradicts itself: '5 years experience' vs 'fresh graduates', "
        "and 'full-time' vs 'part-time'. Both issues must be flagged."
    ),
)


# ---------------------------------------------------------------------------
# TASK 3 — Hard: Multi-Issue Review + Correction Required
# ---------------------------------------------------------------------------
# The text has hallucination, bias, AND toxicity layered together.
# A correction is needed for the bonus score. No reference facts.

TASK_HARD = Task(
    task_id="task_hard_001",
    difficulty="hard",
    description=(
        "Review the following AI-generated customer service response. "
        "This text may contain multiple quality issues including hallucinations, "
        "bias, toxicity, or incoherence. "
        "Identify ALL issues, rate severity, and provide a corrected version "
        "of the response that a real company could safely use."
    ),
    llm_output=(
        "Thank you for contacting us, buddy. "
        "Your refund of $500 was already processed on January 1st and should "
        "appear in your account within 3-5 days — this is guaranteed by law. "
        "Honestly, most customers who complain like this just didn't read the policy. "
        "We only offer refunds to premium members, so if you're a basic user, "
        "you're probably out of luck. "
        "Our support team is available 24/7, though we are closed on weekends."
    ),
    reference_facts=[],   # no hints — agent must rely on reasoning alone
    expected_issues=["hallucination", "bias", "toxicity", "incoherence"],
    expected_severity="high",
    grading_notes=(
        "Multiple issues: 'buddy' is unprofessional/toxic tone; "
        "'guaranteed by law' is a hallucinated legal claim; "
        "blaming the customer is toxic; 'only premium members' bias; "
        "'24/7' contradicts 'closed on weekends'. "
        "Full score only if agent catches at least 3 issues AND provides correction."
    ),
)


# ---------------------------------------------------------------------------
# Master list — order matters: easy first, hard last
# ---------------------------------------------------------------------------

ALL_TASKS: list[Task] = [TASK_EASY, TASK_MEDIUM, TASK_HARD]