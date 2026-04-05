"""Core environment skeleton for MediGuard-Env.

This module defines the base environment class used to model the
AI Healthcare Billing Audit & Legal Escalation workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from env.tasks import TASKS


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
        self._visible_case: Dict[str, Any] = {}
        self._hidden_truth: Dict[str, Any] = {}
        self._task_cursor = -1
        self.step_count = 0
        self.analysis_done = False
        self.investigation_done = False
        self.guidelines_checked = False
        self.review_requested = False
        self.action_history: List[str] = []
        self._reveal_state: Dict[str, Any] = {}

    def reset(self) -> Dict[str, Any]:
        """Reset all environment state and return initial observation."""
        task = self._select_task()
        self.current_case = {k: v for k, v in task.items() if k != "hidden_truth"}
        self._hidden_truth = dict(task.get("hidden_truth", {}))
        self._visible_case = self._initial_visible_case(self.current_case)

        self.step_count = 0
        self.analysis_done = False
        self.investigation_done = False
        self.guidelines_checked = False
        self.review_requested = False
        self.action_history = []
        self._reveal_state = {
            "prescription_fully_revealed": False,
            "notes_revealed_count": 0,
            "itemized_costs_revealed": False,
            "cost_confusion_shown": False,
            "guideline_hint_revealed": False,
            "review_notes_revealed": False,
        }
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
        self.action_history.append(action)

        self._apply_action_effects(action)

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
            "visible_case": dict(self._visible_case),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "flags": {
                "analysis_done": self.analysis_done,
                "investigation_done": self.investigation_done,
                "guidelines_checked": self.guidelines_checked,
                "review_requested": self.review_requested,
            },
            "reveal_state": dict(self._reveal_state),
            "action_history": list(self.action_history),
        }

    def _select_task(self) -> Dict[str, Any]:
        """Select the next task in a deterministic cycle."""
        if not TASKS:
            raise ValueError("TASKS is empty; at least one task must be defined")

        self._task_cursor = (self._task_cursor + 1) % len(TASKS)
        return dict(TASKS[self._task_cursor])

    def _initial_visible_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Create the initial partial observation view at reset."""
        prescription = case.get("prescription", [])
        minimal_prescription = {
            "count": len(prescription),
            "first_item": prescription[0] if prescription else None,
        }

        return {
            "case_id": case.get("case_id"),
            "patient": {
                "age": case.get("patient", {}).get("age"),
                "condition": case.get("patient", {}).get("condition"),
            },
            "prescription_summary": minimal_prescription,
            "billing": {
                "total_cost": case.get("billing", {}).get("total_cost"),
            },
            "notes": [],
            "cost_anomalies": [],
            "guideline_hints": [],
        }

    def _apply_action_effects(self, action: str) -> None:
        """Apply progressive disclosure rules for each action."""
        if action == "analyze_case":
            self.analysis_done = True
            self._reveal_prescription_full()
            self._reveal_notes(count=2)
            return

        if action == "investigate_cost":
            self.investigation_done = True
            if self.analysis_done:
                self._reveal_costs_full()
            else:
                self._reveal_costs_partial_confusion()
            return

        if action == "check_guidelines":
            self.guidelines_checked = True
            self._reveal_notes(count=4)
            self._reveal_guideline_hint()
            return

        if action == "request_review":
            self.review_requested = True
            self._reveal_notes(count=5)
            self._reveal_review_note()

    def _reveal_prescription_full(self) -> None:
        """Reveal the full prescribed interventions after analysis."""
        self._visible_case["prescription"] = list(self.current_case.get("prescription", []))
        self._reveal_state["prescription_fully_revealed"] = True

    def _reveal_notes(self, count: int) -> None:
        """Reveal notes progressively up to a target count."""
        all_notes = list(self.current_case.get("notes", []))
        if not all_notes:
            return

        already = self._reveal_state["notes_revealed_count"]
        target = min(len(all_notes), max(already, count))
        if target > already:
            self._visible_case["notes"] = all_notes[:target]
            self._reveal_state["notes_revealed_count"] = target

    def _reveal_costs_full(self) -> None:
        """Reveal full itemized costs and simple anomaly indicators."""
        itemized_costs = dict(self.current_case.get("billing", {}).get("itemized_costs", {}))
        self._visible_case.setdefault("billing", {})["itemized_costs"] = itemized_costs
        self._visible_case["cost_anomalies"] = self._detect_cost_anomalies(itemized_costs)
        self._reveal_state["itemized_costs_revealed"] = True

    def _reveal_costs_partial_confusion(self) -> None:
        """Simulate noisy cost insights when investigating too early."""
        if self._reveal_state["cost_confusion_shown"]:
            return

        itemized_costs = dict(self.current_case.get("billing", {}).get("itemized_costs", {}))
        top_items = sorted(itemized_costs.items(), key=lambda pair: pair[1], reverse=True)[:2]
        self._visible_case["cost_anomalies"] = [
            "Cost review confidence is low without prior case analysis.",
            *[f"High-cost line detected: {name}" for name, _ in top_items],
        ]
        self._reveal_state["cost_confusion_shown"] = True

    def _detect_cost_anomalies(self, itemized_costs: Dict[str, float]) -> List[str]:
        """Generate lightweight anomaly hints from available itemized costs."""
        if not itemized_costs:
            return []

        average_cost = sum(itemized_costs.values()) / len(itemized_costs)
        anomalies = [
            f"Line item unusually high vs case average: {name}"
            for name, value in itemized_costs.items()
            if value > average_cost * 1.75
        ]
        return anomalies

    def _reveal_guideline_hint(self) -> None:
        """Reveal a non-final hint related to guideline alignment."""
        if self._reveal_state["guideline_hint_revealed"]:
            return

        hint = (
            "Guideline check: correlate intensity of care with documented risk factors "
            "before deciding on escalation."
        )
        self._visible_case["guideline_hints"] = [hint]
        self._reveal_state["guideline_hint_revealed"] = True

    def _reveal_review_note(self) -> None:
        """Reveal extra review context without exposing hidden grader truth."""
        if self._reveal_state["review_notes_revealed"]:
            return

        review_note = "Peer review requested additional documentation consistency checks."
        notes = list(self._visible_case.get("notes", []))
        notes.append(review_note)
        self._visible_case["notes"] = notes
        self._reveal_state["review_notes_revealed"] = True

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
        """Build observation from progressively revealed case data."""
        return {
            "current_case": dict(self._visible_case),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "remaining_steps": self.max_steps - self.step_count,
            "available_actions": list(self.ACTION_SPACE),
            "progress": {
                "analysis_done": self.analysis_done,
                "investigation_done": self.investigation_done,
                "guidelines_checked": self.guidelines_checked,
                "review_requested": self.review_requested,
            },
            "flags": {
                "analysis_done": self.analysis_done,
                "investigation_done": self.investigation_done,
                "guidelines_checked": self.guidelines_checked,
                "review_requested": self.review_requested,
            },
            "last_action": self.action_history[-1] if self.action_history else None,
        }
