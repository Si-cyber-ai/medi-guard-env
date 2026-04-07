"""Deterministic trajectory grader for MediGuard-Env episodes.

This module provides a pure scoring function that evaluates both final decision
quality and reasoning trajectory quality from an action history.
"""

from __future__ import annotations

from typing import Any, Dict, List


def count_repetitions(action_history: List[str]) -> int:
    """Return the highest repetition count among actions in a trajectory."""
    if not action_history:
        return 0

    counts: Dict[str, int] = {}
    for action in action_history:
        counts[action] = counts.get(action, 0) + 1
    return max(counts.values())


def grade_episode(action_history: List[str], hidden_truth: Dict[str, Any]) -> float:
    """Grade an episode trajectory deterministically with score in [0.0, 1.0].

    Args:
        action_history: Ordered list of actions taken in the episode.
        hidden_truth: Ground-truth labels used for deterministic evaluation.

    Returns:
        A normalized float score between 0.0 and 1.0.
    """
    is_over_treatment = bool(hidden_truth.get("is_over_treatment", False))
    is_overpriced = bool(hidden_truth.get("is_overpriced", False))
    escalation_needed = bool(hidden_truth.get("escalation_needed", False))
    issue_exists = is_over_treatment or is_overpriced

    score = 0.0

    # No trajectory means no evidence of reasoning or decision quality.
    if not action_history:
        return score

    final_action = action_history[-1]

    # A. Final decision correctness (max +0.5)
    if final_action == "flag_issue":
        score += 0.48 if issue_exists else 0.1

    if final_action == "approve_case":
        score += 0.5 if not issue_exists else 0.0

    if final_action == "escalate_case":
        if escalation_needed:
            score += 0.5
        elif issue_exists:
            score += 0.2
        else:
            score += 0.0

    # B. Reasoning quality (max +0.3)
    if "analyze_case" in action_history:
        score += 0.1

    if "investigate_cost" in action_history:
        score += 0.1

    if "check_guidelines" in action_history or "request_review" in action_history:
        score += 0.1

    # Reward explicit conflict handling under uncertainty.
    if "check_guidelines" in action_history and "request_review" in action_history:
        score += 0.05

    # Bonus for correct reasoning order: analyze before investigate.
    if "analyze_case" in action_history and "investigate_cost" in action_history:
        if action_history.index("analyze_case") < action_history.index("investigate_cost"):
            score += 0.05

    # Optional sequence intelligence bonus for strong early strategy.
    if action_history[:2] == ["analyze_case", "investigate_cost"]:
        score += 0.03

    # Efficiency bonus for concise but meaningful trajectories.
    if len(action_history) <= 4:
        score += 0.05

    # Tiny variation for exactly 4-step trajectories.
    if len(action_history) == 4:
        score += 0.02

    # C. Bad behavior penalties (max -0.3)
    if count_repetitions(action_history) > 2:
        score -= 0.1

    if final_action in {"flag_issue", "approve_case", "escalate_case"} and "analyze_case" not in action_history:
        score -= 0.2

    if final_action == "escalate_case" and "investigate_cost" not in action_history:
        score -= 0.2

    # Penalize confident terminal decisions made without guideline check.
    if final_action in {"flag_issue", "approve_case"} and "check_guidelines" not in action_history:
        score -= 0.1

    # Penalize missing investigation when issues actually exist.
    if issue_exists and "investigate_cost" not in action_history:
        score -= 0.1

    # Penalize immediate final decisions with no reasoning trajectory.
    if final_action in {"flag_issue", "approve_case", "escalate_case"} and len(action_history) <= 1:
        score -= 0.2

    # D. Missed issue penalty
    if issue_exists and final_action == "approve_case":
        score -= 0.3

    # E. Over-reaction penalty
    if not issue_exists and final_action == "escalate_case":
        score -= 0.3

    # Clamp score to valid OpenEnv range.
    score = max(0.0, min(1.0, score))
    return float(score)
