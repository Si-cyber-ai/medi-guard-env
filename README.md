# MediGuard-Env

MediGuard-Env is an OpenEnv-style environment for AI healthcare billing audit and legal escalation workflows.

This repository currently contains a production-oriented skeleton with typed models and clean state transitions, while intentionally deferring tasks, grading, and reward strategy logic.

## Current Structure

- env/environment.py
- env/models.py
- env/tasks.py
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

1. Environment now imports and cycles through `TASKS` from env/tasks.py on each reset.
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

## What Is Intentionally Not Implemented Yet

1. Task generation logic beyond the initial fixed benchmark set
2. Reward strategy/business scoring logic
3. Grader/evaluation pipeline
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

## Maintenance Rule

For each new prompt:

1. Implement requested code changes.
2. Update this README in the Prompt Update Log and relevant sections.
3. Keep implementation and documentation aligned.
