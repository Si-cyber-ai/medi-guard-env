"""Typed shared models for MediGuard-Env.

These models are intentionally lightweight and reusable across the
environment runtime, API surfaces, and inference workflows.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator


ACTION_SPACE: Tuple[str, ...] = (
    "analyze_case",
    "investigate_cost",
    "check_guidelines",
    "flag_issue",
    "request_review",
    "approve_case",
    "escalate_case",
)


class ObservationModel(BaseModel):
    """Represents the observable state exposed to the agent.

    Fields:
    - current_case: Current visible case payload (no hidden ground truth).
    - step_count: Number of actions already taken in this episode.
    - max_steps: Episode length cap.
    - remaining_steps: Steps left before forced termination.
    - available_actions: Action names the agent can choose from.
    - progress: Boolean progress flags (analysis_done, investigation_done).
    - last_action: Most recent action, or None if episode just started.
    """

    model_config = ConfigDict(extra="forbid")

    current_case: Dict[str, Any]
    step_count: int
    max_steps: int
    remaining_steps: int
    available_actions: List[str]
    progress: Dict[str, bool]
    last_action: Optional[str] = None

    @field_validator("progress")
    @classmethod
    def validate_progress_keys(cls, value: Dict[str, bool]) -> Dict[str, bool]:
        """Ensure progress contains exactly the required boolean keys."""
        required_keys = {"analysis_done", "investigation_done"}
        if set(value.keys()) != required_keys:
            keys = ", ".join(sorted(required_keys))
            raise ValueError(f"progress must contain exactly these keys: {keys}")
        return value

    @field_validator("available_actions")
    @classmethod
    def validate_available_actions(cls, value: List[str]) -> List[str]:
        """Ensure all observed available actions are valid environment actions."""
        for action in value:
            if action not in ACTION_SPACE:
                raise ValueError("Invalid action in available_actions")
        return value


class ActionModel(BaseModel):
    """Represents an action request submitted by an agent.

    Fields:
    - action_type: Action identifier; must be non-empty and in ACTION_SPACE.
    - reasoning: Optional agent rationale for future evaluation workflows.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: str
    reasoning: Optional[str] = None

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, value: str) -> str:
        """Ensure action_type is non-empty and supported by the environment."""
        action = value.strip()
        if not action:
            raise ValueError("action_type must not be empty")
        if action not in ACTION_SPACE:
            allowed = ", ".join(ACTION_SPACE)
            raise ValueError(f"action_type must be one of: {allowed}")
        return action


RewardValue = Annotated[
    StrictFloat,
    Field(
        ge=-1.0,
        le=1.0,
        description="Reward scalar constrained to [-1.0, 1.0].",
    ),
]


class RewardModel(BaseModel):
    """Represents a reward signal and short explanatory context.

    Fields:
    - value: Reward score in the closed interval [-1.0, 1.0].
    - explanation: Human-readable reason for why the reward was assigned.
    """

    model_config = ConfigDict(extra="forbid")

    value: RewardValue
    explanation: str = Field(min_length=1)
