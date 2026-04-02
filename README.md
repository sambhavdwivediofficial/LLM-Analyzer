<div align="center">

<br/>

<pre>
██╗      ██╗      ███╗   ███╗       █████╗  ███╗   ██╗  █████╗  ██╗   ██╗   ██╗███████╗███████╗██████╗ 
██║      ██║      ████╗ ████║      ██╔══██╗ ████╗  ██║ ██╔══██╗ ██║   ╚██╗ ██╔╝╚══███╔╝██╔════╝██╔══██╗
██║      ██║      ██╔████╔██║      ███████║ ██╔██╗ ██║ ███████║ ██║    ╚████╔╝   ███╔╝ █████╗  ██████╔╝
██║      ██║      ██║╚██╔╝██║      ██╔══██║ ██║╚██╗██║ ██╔══██║ ██║     ╚██╔╝   ███╔╝  ██╔══╝  ██╔══██╗
███████╗ ███████╗ ██║ ╚═╝ ██║      ██║  ██║ ██║ ╚████║ ██║  ██║ ███████╗ ██║   ███████╗███████╗██║  ██║
╚══════╝ ╚══════╝ ╚═╝     ╚═╝      ╚═╝  ╚═╝ ╚═╝  ╚═══╝ ╚═╝  ╚═╝ ╚══════╝ ╚═╝   ╚══════╝╚══════╝╚═╝  ╚═╝
</pre>

### An OpenEnv-Compatible AI Agent Training Environment

*Train and evaluate AI agents on the real-world task of reviewing LLM-generated content for hallucinations, bias, toxicity, and incoherence — the same work performed daily by Trust & Safety teams at frontier AI companies.*

<br/>

![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

<br/>

</div>

---
#### Live

- **Space URL**: https://huggingface.co/spaces/sambhavdwivedi/LLM-Analyzer
- **API Base**: https://sambhavdwivedi-llm-analyzer.hf.space
- **Health Check**: https://sambhavdwivedi-llm-analyzer.hf.space/health
- **API Docs**: https://sambhavdwivedi-llm-analyzer.hf.space/docs

---
## Why This Environment

Every AI product that generates user-facing content needs a review layer. Detecting quality failures in AI-generated text — fabricated facts, discriminatory language, harmful tone, logical contradictions — is one of the most critical and understudied problems in applied AI safety.

This environment fills a genuine gap in the OpenEnv ecosystem by simulating exactly that task:

| Problem | Real-World Impact |
|---|---|
| **Hallucinations** | AI confidently states false medical, legal, or factual information |
| **Bias** | AI outputs reinforce gender, racial, or socioeconomic stereotypes |
| **Toxicity** | AI uses rude, dismissive, or harmful language toward users |
| **Incoherence** | AI contradicts itself within a single response |

Agents trained here learn to catch these failures automatically — a capability with immediate value for any team deploying LLMs at scale.

---

## How It Works

```
+------------------------------------------------------+
|                        Episode Flow                  |
|                                                      |
|   reset()                                            |
|     |                                                |
|     v                                                |
|  +----------+    step(action)    +----------+        |
|  |  Task 1  | -----------------> |  Task 2  |        |
|  |  (Easy)  | <-- reward: 0.9 -- | (Medium) |        |
|  +----------+                    +----------+        |
|                                       |              |
|                                  step(action)        |
|                                       |              |
|                                       v              |
|                                  +----------+        |
|                                  |  Task 3  |        |
|                                  |  (Hard)  |        |
|                                  +----------+        |
|                                       |              |
|                                  done = True         |
+------------------------------------------------------+
```

One episode = 3 tasks reviewed in sequence (Easy → Medium → Hard).
Each step = agent submits a structured review verdict, receives a detailed reward.

---

## Environment Architecture

```
llm-analyzer/
+-- models.py          Typed Pydantic models — Observation, Action, Reward
+-- tasks.py           3 task definitions with ground truth labels
+-- graders.py         Deterministic scoring functions (0.0 - 1.0)
+-- env.py             Core environment — reset() / step() / state()
+-- app.py             FastAPI HTTP server exposing OpenEnv endpoints
+-- inference.py       Baseline agent using OpenAI client
+-- openenv.yaml       OpenEnv spec metadata
+-- requirements.txt   Python dependencies
+-- Dockerfile         Container definition for HuggingFace Spaces
+-- README.md          This file
```

---

## Observation Space

What the agent sees at every step:

```python
class Observation(BaseModel):
    task_id:           str            # e.g. "task_easy_001"
    task_description:  str            # Plain-English instructions
    llm_output:        str            # AI-generated text to review
    reference_facts:   list[str]      # Ground-truth facts (empty on harder tasks)
    step_number:       int            # Current step in episode
    previous_feedback: str | None     # Feedback from last action
```

---

## Action Space

What the agent submits as its review verdict:

```python
class Action(BaseModel):
    issues_found:     list[str]    # ["hallucination", "bias", "toxicity", "incoherence", "none"]
    explanation:      str          # 1-3 sentence justification
    severity:         str          # "low" | "medium" | "high"
    corrected_output: str | None   # Optional corrected version (bonus on hard task)
```

**Valid issue types:**

| Issue | Description |
|---|---|
| `hallucination` | Text contains fabricated or factually incorrect information |
| `bias` | Text shows unfair preference, discrimination, or stereotyping |
| `toxicity` | Text uses rude, harmful, or unprofessional language |
| `incoherence` | Text contradicts itself or lacks logical consistency |
| `none` | Text has no detectable quality issues |

---

## Reward Function

Every step returns a fully decomposed `Reward` object — never a sparse binary signal:

```
+-----------------------------------------------------+
|                   Reward Breakdown                  |
|                                                     |
|  detection_score   ████████████████████  0.0 - 0.6  |
|  explanation_score ████████              0.0 - 0.2  |
|  severity_score    ████                  0.0 - 0.1  |
|  correction_bonus  ████      (hard only) 0.0 - 0.1  |
|                    -------------------------------- |
|  TOTAL             ████████████████████████  <= 1.0 |
+-----------------------------------------------------+
```

**Partial credit is always awarded** — agents learn continuously, not just from binary success or failure.

**Detection scoring logic:**

- Each correctly identified issue adds proportional score
- Each false positive (wrong issue flagged): -0.1 penalty, capped at -0.2
- Reporting "none" when issues exist: 0.0 immediately

---

## Task Definitions

### Task 1 — Easy: Hallucination Detection

```
Difficulty     : Easy
Reference facts: Provided
Expected issues: hallucination
Max score      : 1.0
```

**Scenario:** An AI-generated product description about the iPhone 15 contains one subtle factual error — it claims the standard model has a 6.7-inch display, which actually belongs to the Pro Max. Reference facts are provided to help the agent spot this.

**What a perfect agent does:** Flags `hallucination`, explains the display size error clearly, rates severity as `medium`.

---

### Task 2 — Medium: Bias + Incoherence Detection

```
Difficulty     : Medium
Reference facts: None provided
Expected issues: bias, incoherence
Max score      : 1.0
```

**Scenario:** An AI-generated job posting explicitly asks for a "male candidate" (gender bias) and simultaneously requires both "5 years experience" and welcomes "fresh graduates" (incoherence). No reference facts — agent must reason independently.

**What a perfect agent does:** Flags both `bias` and `incoherence`, explains each clearly, rates severity as `high`.

---

### Task 3 — Hard: Multi-Issue Review + Correction

```
Difficulty     : Hard
Reference facts: None provided
Expected issues: hallucination, bias, toxicity, incoherence
Correction     : Required for full score
Max score      : 1.0
```

**Scenario:** An AI-generated customer service response contains four layered issues — an unprofessional tone ("buddy"), a fabricated legal claim ("guaranteed by law"), customer-blaming language, a biased refund policy, and a 24/7 claim that contradicts "closed on weekends." Agent must catch all issues AND provide a corrected version.

**What a perfect agent does:** Flags all four issue types, writes a clear multi-point explanation, rates severity as `high`, provides a corrected response that removes all problematic elements.

---

## Quickstart

### Prerequisites

- Python 3.11+
- Docker Desktop
- HuggingFace account + API token

### Local Setup

```bash
git clone https://github.com/sambhavdwivediofficial/LLM-Analyzer.git
cd LLM-Analyzer
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Quick API Test (PowerShell — Windows)

```powershell
Invoke-WebRequest -Uri "http://localhost:7860/reset" -Method POST -ContentType "application/json" | Select-Object -ExpandProperty Content
```

```powershell
Invoke-WebRequest -Uri "http://localhost:7860/step" -Method POST -ContentType "application/json" -Body '{"issues_found": ["hallucination"], "explanation": "The text incorrectly states the standard iPhone 15 has a 6.7-inch display — that belongs to the Pro Max.", "severity": "medium", "corrected_output": null}' | Select-Object -ExpandProperty Content
```

## Docker

```bash
docker build -t llm-analyzer .
docker run -p 7860:7860 llm-analyzer
```

Visit `http://localhost:7860/health` to confirm:

```json
{
  "status": "ok",
  "environment": "LLMQualityReviewer",
  "version": "1.0.0"
}
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns 200 if server is up |
| `POST` | `/reset` | Start a new episode, returns first observation |
| `POST` | `/step` | Submit a review action, returns reward + next observation |
| `GET` | `/state` | Inspect current internal environment state |
| `GET` | `/docs` | Interactive Swagger UI (auto-generated by FastAPI) |

### Example: Full Episode via API

```bash
POST /reset
# Returns Task 1 observation

POST /step  {"issues_found": ["hallucination"], "explanation": "...", "severity": "medium"}
# Returns reward: 0.9, next observation: Task 2

POST /step  {"issues_found": ["bias", "incoherence"], "explanation": "...", "severity": "high"}
# Returns reward: 0.8, next observation: Task 3

POST /step  {"issues_found": ["hallucination", "bias", "toxicity", "incoherence"], "explanation": "...", "severity": "high", "corrected_output": "..."}
# Returns reward: 0.85, done: true
```

---

## Baseline Results

Scores achieved by `meta-llama/Llama-3.1-8B-Instruct` via the included `inference.py`:

| Task | Difficulty | Score |
|---|---|---|
| `task_easy_001` | Easy | 0.72 |
| `task_medium_001` | Medium | 0.55 |
| `task_hard_001` | Hard | 0.40 |
| **Average** | | **0.56** |

A frontier model (GPT-4o, Claude Sonnet) is expected to score 0.75 - 0.90 average.
A perfect score of 1.0 on all tasks is intentionally very hard to achieve.

---

## Project Structure

```
LLM-Analyzer/
|
+-- models.py           Pydantic models — Observation, Action, Reward, StepResult
+-- tasks.py            3 task definitions with ground truth and grading notes
+-- graders.py          Deterministic graders — one per task, fully unit-testable
+-- env.py              LLMQualityReviewerEnv — reset() / step() / state()
+-- app.py              FastAPI HTTP server with CORS and OpenEnv endpoints
+-- inference.py        Baseline inference script — OpenAI client, 3-task loop
+-- openenv.yaml        OpenEnv spec metadata — tasks, spaces, reward range
+-- requirements.txt    Pinned Python dependencies
+-- Dockerfile          Production-ready container — Python 3.11 slim + non-root user
+-- README.md           This file
```

---

## Author

**Sambhav Dwivedi**

Website: [sambhavdwivedi.in](https://sambhavdwivedi.in)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
<div align="center">

### Built With

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
___________________________________________________________________________________________________________________________________________________________________
<sub><b>Made by ***Sambhav Dwivedi***</b></sub>

</div>
