# MediGuard-Env

MediGuard-Env is an OpenEnv-compatible AI environment for healthcare billing audit and legal escalation decisions. It simulates realistic claim-review uncertainty where agents must inspect evidence, detect anomalies, and choose safe outcomes (`approve_case`, `flag_issue`, `escalate_case`) under partial observability.

## Real-World Motivation

Healthcare billing review requires balancing patient safety, fraud detection, and escalation risk. Over-aggressive flagging can delay care, while under-detection can allow over-treatment or overpricing. MediGuard-Env models this tradeoff with staged evidence reveal, noisy rewards, and trajectory-based grading.

## OpenEnv Compliance

This environment implements the full OpenEnv interface:

- `reset()` -> returns initial observation
- `step(action)` -> returns observation, reward, done, info
- `state()` -> returns full environment state

Typed models are defined using Pydantic and validated for consistency.

## Project Structure

- `api.py`
- `env/environment.py`
- `env/grader.py`
- `env/models.py`
- `env/tasks.py`
- `inference.py`
- `requirements.txt`
- `openenv.yaml`
- `Dockerfile`
- `README.md`

## Action Space

- `analyze_case`
- `investigate_cost`
- `check_guidelines`
- `request_review`
- `flag_issue`
- `approve_case`
- `escalate_case`

## Observation Space

Observation payload (key fields):

- `current_case`: progressively revealed case data
- `step_count`: current step index
- `max_steps`: episode limit
- `remaining_steps`: remaining budget
- `available_actions`: legal actions
- `progress`:
  - `analysis_done`
  - `investigation_done`
  - `guidelines_checked`
  - `review_requested`
  - `decision_taken`
- `info_level`:
  - `analysis`
  - `cost`
  - `guidelines`
  - `review`
- `confidence_level`
- `confidence`:
  - `analysis_confidence`
  - `cost_confidence`
- `last_action`

## Tasks

The environment includes three difficulty levels:

- **Easy** -> Clear anomaly with minimal ambiguity
- **Medium** -> Mixed signals with documentation gaps
- **Hard** -> High-cost but medically justified case with conflicting evidence

## Reward Logic

Dense reward in `env/environment.py` (`_calculate_reward`) includes:

- Progress rewards for first-time meaningful investigative steps
- Sequence and repetition penalties
- Final decision rewards vs hidden truth (`over_treatment`, `overpriced`, `escalation_needed`)
- Missed-issue and over-reaction penalties
- Premature decision penalties
- Small bounded reward noise to mimic real-world uncertainty
- Hard clamp to `[-1.0, 1.0]`

## Reward Properties

- Step rewards: continuous signal
- Final reward: correctness-based
- Reward range: **[-1.0, 1.0]**
- Grader output range: **[0.0, 1.0]**

## Why This Environment Matters

Unlike toy RL environments, MediGuard models real-world healthcare decision-making:

- Partial observability
- Conflicting expert signals
- Cost vs safety tradeoffs
- Delayed and incomplete information

This makes it suitable for evaluating real agent reasoning under uncertainty.

## Example Interaction

1. `reset()` -> initial limited observation
2. `analyze_case` -> reveals clinical notes
3. `investigate_cost` -> reveals itemized billing and anomalies
4. `check_guidelines` -> reveals guideline hints
5. `approve_case` / `flag_issue` -> final decision

Agents must balance safety, cost, and uncertainty to maximize reward.

## Deterministic Grader

`env/grader.py` provides `grade_episode(action_history, hidden_truth) -> float`.

- Pure deterministic function
- Uses full trajectory (not only final action)
- Scores decision correctness + reasoning quality
- Applies behavior penalties
- Returns clamped score in `[0.0, 1.0]`

## Setup Instructions

1. Create and activate virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Optional prerequisite audit:

```powershell
python setup_check.py
```

## Run API Locally

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Test endpoints:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/reset
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/step -ContentType "application/json" -Body '{"action":"analyze_case"}'
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/state
```

## Run Inference

Set environment variables (example):

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN = "<your_token>"
python inference.py
```

Expected log format:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Docker

Container runtime profile:

- base image: `python:3.10-slim`
- system packages: `build-essential`, `curl`
- pip upgraded during build
- server command: `uvicorn api:app --host 0.0.0.0 --port 7860 --workers 1`

Build and run:

```powershell
docker build -t mediguard-env .
docker run --rm -p 7860:7860 mediguard-env
```

Health check:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:7860/
```

## Hugging Face Spaces (Docker)

1. Create a new Space -> SDK: **Docker**.
2. Push repository containing `Dockerfile`, `openenv.yaml`, and source code.
3. In Space settings -> Variables and secrets:
   - `HF_TOKEN`
   - `MODEL_NAME` (optional override)
   - `API_BASE_URL` (optional override)
4. Wait for build completion.
5. Test endpoint:

```bash
curl -X POST https://<space-name>.hf.space/reset
```

## Latest Update

- Enhanced `openenv.yaml` metadata with `version`, long-form `description`, and deployment `tags` for better compatibility and clarity.
- Switched `openenv.yaml` `description` to a single-line string to eliminate YAML block scalar indentation issues.
- Finalized `requirements.txt` with the submission-safe dependency set, including `openenv-core`, `python-multipart`, and `typing-extensions`.
