"""Core environment skeleton for MediGuard-Env.

This module defines the base environment class used to model the
AI Healthcare Billing Audit & Legal Escalation workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class MediGuardEnv:
    """Environment skeleton for case-audit decision flow.

    Notes:
    - Reward logic, grading, and task logic are intentionally omitted.
    - This class currently focuses on state transition scaffolding only.
    """

    ACTION_SPACE: List[str] = [
        "analyze_case",
        "investigate_cost",
        "check_guidelines",
        "flag_issue",
        "request_review",
        "approve_case",
        "escalate_case",
    ]

    # Actions that end the current episode immediately.
    FINAL_ACTIONS = {"approve_case", "escalate_case", "flag_issue"}

    def __init__(self, max_steps: int = 6) -> None:
        """Initialize the environment with configurable episode length."""
        self.max_steps = max_steps
        self.current_case: Dict[str, Any] = {}
        self.step_count = 0
        self.analysis_done = False
        self.investigation_done = False
        self.action_history: List[str] = []

    def reset(self) -> Dict[str, Any]:
        """Reset all environment state and return initial observation."""
        self.current_case = self._initial_case_placeholder()
        self.step_count = 0
        self.analysis_done = False
        self.investigation_done = False
        self.action_history = []
        return self._build_observation()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one action and return (observation, reward, done, info).

        Current behavior:
        - Validates action against ACTION_SPACE.
        - Increments step count.
        - Updates progress flags for analyze/investigate actions.
        - Records action in history.
        - Returns placeholder reward and info values.
        """
        self._validate_action(action)

        self.step_count += 1

        if action == "analyze_case":
            self.analysis_done = True
        elif action == "investigate_cost":
            self.investigation_done = True

        self.action_history.append(action)

        done = action in self.FINAL_ACTIONS or self.step_count >= self.max_steps
        observation = self._build_observation()
        reward = 0.0

        # Include reason metadata to make episode termination debuggable.
        if action in self.FINAL_ACTIONS:
            reason = "final_action_taken"
        elif self.step_count >= self.max_steps:
            reason = "max_steps_reached"
        else:
            reason = None

        info: Dict[str, Any] = {"done_reason": reason}

        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return a snapshot of the current internal state."""
        return {
            "current_case": dict(self.current_case),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "flags": {
                "analysis_done": self.analysis_done,
                "investigation_done": self.investigation_done,
            },
            "action_history": list(self.action_history),
        }

    def _validate_action(self, action: str) -> None:
        """Raise a clear error when an invalid action is provided."""
        if action not in self.ACTION_SPACE:
            allowed = ", ".join(self.ACTION_SPACE)
            raise ValueError(f"Invalid action '{action}'. Allowed actions: {allowed}")

    def _initial_case_placeholder(self) -> Dict[str, Any]:
        """Create an empty case placeholder for future enrichment."""
        return {
            "case_id": None,
            "patient_id": None,
            "claim_amount": None,
            "notes": [],
        }

    def _build_observation(self) -> Dict[str, Any]:
        """Build a minimal observation view from internal state."""
        return {
            "current_case": dict(self.current_case),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "remaining_steps": self.max_steps - self.step_count,
            "available_actions": list(self.ACTION_SPACE),
            "progress": {
                "analysis_done": self.analysis_done,
                "investigation_done": self.investigation_done,
            },
            "flags": {
                "analysis_done": self.analysis_done,
                "investigation_done": self.investigation_done,
            },
            "last_action": self.action_history[-1] if self.action_history else None,
        }
