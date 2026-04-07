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
    - confidence_level: Scalar confidence in [0.0, 1.0].
    - progress: Boolean progress flags for reasoning workflow stages.
    - info_level: Boolean map describing information completeness by domain.
    - audit_flags: Optional flattened audit indicators for agent convenience.
    - last_action: Most recent action, or None if episode just started.
    """

    model_config = ConfigDict(extra="forbid")

    current_case: Dict[str, Any]
    step_count: int
    max_steps: int
    remaining_steps: int
    available_actions: List[str]
    confidence_level: float = Field(ge=0.0, le=1.0)
    progress: Dict[str, bool]
    info_level: Dict[str, bool]
    audit_flags: Optional[List[str]] = None
    last_action: Optional[str] = None

    @field_validator("progress")
    @classmethod
    def validate_progress_keys(cls, value: Dict[str, bool]) -> Dict[str, bool]:
        """Ensure progress contains exactly the required boolean keys."""
        required_keys = {
            "analysis_done",
            "investigation_done",
            "guidelines_checked",
            "review_requested",
            "decision_taken",
        }
        if set(value.keys()) != required_keys:
            keys = ", ".join(sorted(required_keys))
            raise ValueError(f"progress must contain exactly these keys: {keys}")
        return value

    @field_validator("info_level")
    @classmethod
    def validate_info_level_keys(cls, value: Dict[str, bool]) -> Dict[str, bool]:
        """Ensure info_level contains expected completeness flags."""
        required_keys = {"analysis", "cost", "guidelines", "review"}
        if set(value.keys()) != required_keys:
            keys = ", ".join(sorted(required_keys))
            raise ValueError(f"info_level must contain exactly these keys: {keys}")
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
    - reasoning: Optional structured rationale including confidence.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: str
    reasoning: Optional["ReasoningBlock"] = None

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


class ReasoningBlock(BaseModel):
    """Structured reasoning payload for interpretable agent decisions."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


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
    - metadata: Optional decision diagnostics (e.g., premature_decision).
    """

    model_config = ConfigDict(extra="forbid")

    value: RewardValue
    explanation: str = Field(min_length=1)
    metadata: Optional[Dict[str, Any]] = None


ActionModel.model_rebuild()
