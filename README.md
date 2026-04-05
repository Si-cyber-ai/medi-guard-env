# MediGuard-Env

MediGuard-Env is an OpenEnv-style environment for AI healthcare billing audit and legal escalation workflows.

This repository currently contains a production-oriented skeleton with typed models and clean state transitions, while intentionally deferring tasks, grading, and reward strategy logic.

## Current Structure

- env/environment.py
- env/models.py
- env/tasks.py
- env/grader.py
- inference.py
- setup_check.py
- api.py
- .gitignore

## What Is Implemented

### Step 1: Environment Skeleton

Implemented in env/environment.py.

1. Core class: `MediGuardEnv`
2. Public methods:
1. `reset()`
2. `step(action)`
3. `state()`
3. Internal state:
1. `current_case`
2. `step_count`
3. `max_steps` (default 6)
4. flags (`analysis_done`, `investigation_done`)
5. `action_history`
4. Action space:
1. `analyze_case`
2. `investigate_cost`
3. `check_guidelines`
4. `flag_issue`
5. `request_review`
6. `approve_case`
7. `escalate_case`
5. Episode termination:
1. Final action reached (`approve_case`, `escalate_case`, `flag_issue`)
2. Or `max_steps` reached
6. Step return contract:
1. observation (placeholder dict)
2. reward (`0.0`)
3. done (bool)
4. info (`done_reason` metadata)
7. Observation fields currently include:
1. `current_case`
2. `step_count`
3. `max_steps`
4. `remaining_steps`
5. `available_actions`
6. `progress`
7. `flags`
8. `last_action`

### Step 2: Typed Models

Implemented in env/models.py using Pydantic `BaseModel`.

1. `ObservationModel`
1. `current_case: Dict[str, Any]`
2. `step_count: int`
3. `max_steps: int`
4. `remaining_steps: int`
5. `available_actions: List[str]`
6. `progress: Dict[str, bool]`
7. `last_action: Optional[str]`
2. `ActionModel`
1. `action_type: str`
2. `reasoning: Optional[str]`
3. `RewardModel`
1. `value: StrictFloat` constrained to `[-1.0, 1.0]`
2. `explanation: str` with `min_length=1`
4. Validation included:
1. `action_type` must be non-empty after trim
2. `action_type` must be in `ACTION_SPACE`
3. `progress` must contain exactly `analysis_done` and `investigation_done`
4. `available_actions` must contain only valid actions
5. `ACTION_SPACE` is immutable (`Tuple[str, ...]`)

### Step 3: Realistic Scenario Tasks

Implemented in env/tasks.py.

1. Added 3 production-style task dictionaries with consistent schema:
1. `case_id`
2. `patient` (`age`, `condition`, `history`)
3. `prescription`
4. `billing` (`total_cost`, `itemized_costs`)
5. `notes`
6. `hidden_truth` (`is_over_treatment`, `is_overpriced`, `escalation_needed`, `uncertainty_level`, `expected_best_action`, `justification`)
2. Difficulty progression included:
1. Easy: clearly excessive treatment with one valid distractor
2. Medium: mixed appropriateness, slight overpricing, incomplete records
3. Hard: high-cost care that appears suspicious but is justified by risk factors
3. Added ambiguity and trap signals to prevent trivial policy learning.

### Step 4: Task Integration And Progressive Disclosure

Implemented in env/environment.py.

1. Environment now selects from `TASKS` at reset using random sampling for novelty, while still tracking `task_index`.
2. `hidden_truth` is stored internally (`self._hidden_truth`) and not exposed in observations.
3. Reset now starts from partial observability:
1. patient age and condition
2. minimal prescription summary
3. billing total only
4. no detailed notes or itemized costs
4. Action effects now progressively reveal data:
1. `analyze_case`: full prescription + first notes
2. `investigate_cost`: full itemized costs only after analysis; otherwise partial/confused anomaly hints
3. `check_guidelines`: more notes + guideline hint
4. `request_review`: additional review note without ending episode
5. `flag_issue` / `approve_case` / `escalate_case`: terminate episode
5. Added additional progression flags:
1. `guidelines_checked`
2. `review_requested`
6. Observation builder now returns only progressively revealed view, not full case payload.

### Step 4.1: Robust Transition Safeguards And Observation Cleanup

Implemented in env/environment.py.

1. Removed duplicate observation keying by keeping `progress` and dropping redundant `flags` from observations.
2. Added early termination guard in `step` so actions after terminal decisions return `already_terminated`.
3. Added `decision_taken` state flag for future reward/grader workflows.
4. Corrected progressive note reveal logic to reveal incrementally (`already + count`) rather than jump behavior.
5. Added `info_level` block in observation for clearer reasoning context (`analysis`, `cost`, `guidelines`, `review`).

### Step 4.2: Randomization And Uncertainty Signals

Implemented in env/environment.py.

1. Reset now uses `random.choice(TASKS)` to introduce task novelty across runs.
2. Added small bounded reward noise (`random.uniform(-0.02, 0.02)`) before clamping.
3. Added uncertainty metadata to observations:
1. `confidence_level`
2. `confidence.analysis_confidence`
3. `confidence.cost_confidence`

### Step 5: Dense Reward Logic

Implemented in env/environment.py.

1. Added modular helper `_calculate_reward(action)` and integrated it into `step`.
2. Added first-time progress rewards:
1. `analyze_case` +0.1
2. `investigate_cost` +0.2
3. `check_guidelines` +0.2
4. `request_review` +0.1
3. Added penalties:
1. repeated action: -0.05
2. investigate before analyze: -0.1
3. escalate without prior investigation: -0.3
4. episode ends with real issue undetected: -0.5
5. max steps reached without decision: -0.2
4. Added final-decision reward rules from hidden truth:
1. `flag_issue`: +0.6 if issue exists else -0.4
2. `approve_case`: +0.6 if no issue else -0.7
3. `escalate_case`: +1.0 if escalation needed else -0.8
5. Added reward clamping to `[-1.0, 1.0]`.

### Step 5.1: Reward Quality Polish

Implemented in env/environment.py.

1. Fixed missed-issue penalty trigger to apply only when a decision is actually taken.
2. Improved escalation reward nuance:
1. +1.0 when escalation is truly needed
2. -0.4 when escalation is overreaction but issues still exist
3. -0.8 when escalation is fully incorrect
3. Added good-sequence bonus (+0.2) for final decisions after both analysis and cost investigation.
4. Added premature-decision penalty (-0.3) for final decisions taken without prior analysis.

### Step 6: Deterministic Episode Grader

Implemented in env/grader.py.

1. Added pure function `grade_episode(action_history, hidden_truth) -> float`.
2. Added helper `count_repetitions(action_history)` for trajectory repetition analysis.
3. Scoring components implemented exactly as rubric-driven parts:
1. final decision correctness (up to +0.5)
2. reasoning quality from full trajectory (up to +0.3)
3. bad behavior penalties (up to -0.3)
4. missed issue penalty
5. over-reaction penalty
4. Deterministic behavior guarantees:
1. no randomness
2. no environment dependency
3. score normalized to [0.0, 1.0]

### Step 6.1: Grader Final Polish

Implemented in env/grader.py.

1. Added sequence-quality bonus (+0.05) when `analyze_case` occurs before `investigate_cost`.
2. Added premature-decision penalty (-0.2) for instant final decisions with no trajectory depth.
3. Added efficiency bonus (+0.05) for concise episodes (`len(action_history) <= 4`).
4. Updated normalization comment to explicitly state OpenEnv clamp intent.

### Step 7: Deterministic Inference Script

Implemented in inference.py.

1. Added required imports:
1. `os`
2. `OpenAI`
3. `MediGuardEnv`
4. `grade_episode`
2. Added required environment variable bindings:
1. `API_BASE_URL`
2. `MODEL_NAME`
3. `HF_TOKEN` (as API key)
3. Added strict logging functions with required output formats:
1. `[START]`
2. `[STEP]`
3. `[END]`
4. Added deterministic action policy (`choose_action`) based on progress flags.
5. Implemented synchronous `main()` loop:
1. reset environment
2. step until done
3. accumulate trajectory and rewards
4. compute final score with grader
6. Added standard entry point guard.

### Step 7.1: Inference Compliance Fixes

Implemented in inference.py.

1. Added default fallbacks for required runtime environment variables:
1. `API_BASE_URL` defaults to `https://router.huggingface.co/v1`
2. `MODEL_NAME` defaults to `Qwen/Qwen2.5-72B-Instruct`
2. Made step-level error logging spec compliant:
1. Reads `error` from `info` when available
2. Emits `null` when no error exists
3. Added edge-case output hardening:
1. ensure at least one reward value for END log
2. ensure END step count is at least 1

### Step 8: Setup Prerequisite Checker

Implemented in setup_check.py.

1. Added Python version validation (requires >= 3.10) and prints current version.
2. Added Git and Docker checks via subprocess (`--version`) with install guidance on failure.
3. Added required library checks and on-demand install flow for:
1. openai
2. openenv-core
3. huggingface_hub
4. fastapi
5. uvicorn
6. pydantic
4. Added Hugging Face login readiness check via `HF_TOKEN`.
5. Added environment variable checks for `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` with suggested defaults.
6. Added final setup summary output: `Setup Complete` or `Fix required`.

### Step 9: OpenEnv-Compatible FastAPI API

Implemented in api.py.

1. Added a lightweight FastAPI app with a single global `MediGuardEnv` instance.
2. Exposed OpenEnv-style endpoints:
1. `POST /reset`
2. `POST /step`
3. `GET /state`
4. `GET /`
3. Added `StepRequest` Pydantic model for step requests.
4. Kept JSON responses simple and deterministic.
5. Added run instruction comment for `uvicorn api:app --host 0.0.0.0 --port 8000`.

## What Is Intentionally Not Implemented Yet

1. Task generation logic beyond the initial fixed benchmark set
2. Reward strategy/business scoring logic
3. Grader integration into training/evaluation loop
4. External service integrations

## Push Readiness

A repository-level .gitignore is present and configured for Python development artifacts, virtual environments, local env files, and common editor/OS files.

## Prompt Update Log

This section is the running memory of prompt-driven changes.

1. Prompt Update 1
1. Created core environment skeleton in env/environment.py
2. Added state management, action validation, transitions, and reset/state APIs
2. Prompt Update 2
1. Created typed Pydantic models in env/models.py
2. Added action/reward/progress validation rules
3. Prompt Update 3
1. Applied model polish improvements from Prompt.md
2. Added immutable action space, available actions validator, and non-empty reward explanation
3. Created this README as an evolving implementation log
4. Prompt Update 4
1. Created env/tasks.py with 3 realistic healthcare billing audit scenarios
2. Added easy/medium/hard progression with ambiguity and trap signals
3. Updated README to track Step 3 completion
5. Prompt Update 5
1. Added `uncertainty_level` to each task hidden_truth (`low`, `medium`, `high`)
2. Added `expected_best_action` to support future reward shaping and grading
3. Updated README to keep prompt history synchronized
6. Prompt Update 6
1. Integrated env/tasks.py into environment reset flow with task cycling
2. Added partial observability and action-driven reveal mechanics
3. Added state flags for guideline checks and review requests
4. Updated README to capture Step 4 behavior and information flow
7. Prompt Update 7
1. Removed redundant observation flags and retained `progress`
2. Added strict post-termination protection in `step`
3. Added `decision_taken` tracking for decision lifecycle
4. Fixed incremental note reveal behavior
5. Added `info_level` metadata to observation
8. Prompt Update 8
1. Added `_calculate_reward(action)` helper and wired reward computation into `step`
2. Implemented dense progress rewards, sequence penalties, and final-decision rewards
3. Added missed-issue, over-reaction, and step-limit penalties
4. Added reward clamping for stable bounded output
9. Prompt Update 9
1. Refined missed-issue penalty condition to decision-based triggering only
2. Added nuanced escalation scoring for partial-vs-total errors
3. Added good-sequence bonus for disciplined decision paths
4. Added premature final-decision penalty when analysis is skipped
10. Prompt Update 10
1. Added env/grader.py with deterministic trajectory grading function
2. Implemented rubric-based scoring for correctness, reasoning quality, and penalties
3. Added normalization and repetition helper for stable deterministic scoring
11. Prompt Update 11
1. Added ordered-sequence bonus for analyze-before-investigate reasoning
2. Added premature-final-decision penalty and efficiency bonus
3. Updated clamp comment to explicitly reference OpenEnv score range
12. Prompt Update 12
1. Added root-level inference.py with deterministic synchronous execution loop
2. Implemented strict START/STEP/END logging format and deterministic action policy
3. Integrated final trajectory grading with env/grader.py
13. Prompt Update 13
1. Added default API base URL and model name fallbacks in inference.py
2. Updated STEP logging to propagate real `error` values from `info`
3. Added final log safeguards for empty rewards and zero-step edge cases
14. Prompt Update 14
1. Added root-level setup_check.py for deterministic prerequisite validation and install flow
2. Implemented Python/Git/Docker checks, package auto-install, and env/HF token checks
3. Added final setup status summary output for pass/fix reporting
15. Prompt Update 15
1. Added root-level api.py with FastAPI OpenEnv-compatible endpoints
2. Exposed /reset, /step, /state, and root health message
3. Updated README with API run and endpoint documentation
16. Prompt Update 16
1. Made task cycling explicit with `task_index` in reset
2. Ensured repeated runs rotate through easy, medium, and hard tasks deterministically
3. Updated README to reflect diversity fix for evaluation runs
17. Prompt Update 17
1. Added random task sampling in reset for novelty across runs
2. Added small bounded reward noise for uncertainty realism
3. Added observation confidence metadata and updated README accordingly

## Maintenance Rule

For each new prompt:

1. Implement requested code changes.
2. Update this README in the Prompt Update Log and relevant sections.
3. Keep implementation and documentation aligned.
