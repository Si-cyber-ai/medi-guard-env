import os

import requests
from openai import OpenAI

from env.grader import grade_episode
from env.tasks import TASKS


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY")
ENV_API_BASE = "https://sidh2005-mediguard-env.hf.space"
_OPENAI_CLIENT = None


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_reasoning(observation, action):
    case = observation.get("current_case", {}) or {}
    notes = case.get("notes", []) or []
    anomalies = case.get("cost_anomalies", []) or []
    hints = case.get("guideline_hints", []) or []
    confidence = float(observation.get("confidence_level", 0.5))

    anomaly_state = (
        "Cost anomalies are present in the case."
        if anomalies
        else "No clear cost anomaly is visible from the current evidence."
    )
    risk_state = (
        "Risk appears high because clinical notes or guideline hints indicate meaningful concern."
        if hints or notes
        else "Risk appears low because current evidence does not show strong clinical concern."
    )

    summary = f"{anomaly_state} {risk_state} Action={action}."
    return {
        "summary": summary,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def decide_final_action_with_llm(observation):
    prompt = f"""
You are a healthcare billing auditor.

Choose ONE action:
approve_case OR flag_issue OR escalate_case

Case:
{observation}

Answer with ONLY the action name.
"""

    response = _OPENAI_CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=5,
    )

    content = (response.choices[0].message.content or "").strip().lower()

    if content in ["approve_case", "flag_issue", "escalate_case"]:
        return content

    return "investigate_cost"


def ensure_real_llm_call(client):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=2,
    )
    if not response or not response.choices:
        raise RuntimeError("LLM proxy call failed")


def choose_action(observation):
    p = observation.get("progress", {})
    last_action = observation.get("last_action")

    if last_action == "request_review":
        if not p.get("analysis_done"):
            return "analyze_case"
        elif not p.get("investigation_done"):
            return "investigate_cost"
        elif not p.get("guidelines_checked"):
            return "check_guidelines"
        else:
            return decide_final_action_with_llm(observation)

    if p.get("analysis_done") and p.get("investigation_done") and p.get("guidelines_checked"):
        return decide_final_action_with_llm(observation)

    prompt = f"""
You are an AI healthcare auditor.

Available actions:
analyze_case, investigate_cost, check_guidelines,
request_review, flag_issue, approve_case, escalate_case

Current state:
{observation}

Choose the BEST next action.

Rules:
- Don't jump to final decision too early
- Gather enough evidence
- Avoid unnecessary steps

Return ONLY the action name.
"""

    try:
        response = _OPENAI_CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=5,
        )

        action = (response.choices[0].message.content or "").strip().lower()

        if action in [
            "analyze_case",
            "investigate_cost",
            "check_guidelines",
            "request_review",
            "flag_issue",
            "approve_case",
            "escalate_case",
        ]:
            return action

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        raise

    # fallback (safe but NOT perfect)
    return "request_review"


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    global _OPENAI_CLIENT
    _OPENAI_CLIENT = client

    # Ensure at least one real proxy call before any environment API request.
    ensure_real_llm_call(client)

    for task_idx in range(len(TASKS)):
        hidden_truth = TASKS[task_idx]["hidden_truth"]
        action_history = []
        rewards = []
        step_count = 0
        score = 0.0
        success = False

        log_start(task=f"mediguard_task_{task_idx + 1}", env="mediguard_env", model=MODEL_NAME)

        try:
            reset_response = requests.post(f"{ENV_API_BASE}/reset", timeout=30)
            reset_response.raise_for_status()
            observation = reset_response.json()["observation"]
            done = False

            while not done:
                step_count += 1

                action = choose_action(observation)
                reasoning = build_reasoning(observation, action)

                step_response = requests.post(
                    f"{ENV_API_BASE}/step",
                    json={"action": action, "reasoning": reasoning},
                    timeout=30,
                )
                step_response.raise_for_status()
                payload = step_response.json()
                observation = payload["observation"]
                reward = payload["reward"]
                done = payload["done"]
                info = payload["info"]

                action_history.append(action)
                rewards.append(reward)

                error = info.get("done_reason") if isinstance(info, dict) else None
                log_step(step_count, action, reward, done, error)

            # FINAL SCORE USING DETERMINISTIC GRADER
            score = grade_episode(action_history, hidden_truth)
            score = min(1.0, max(0.0, score))

            success = score > 0.7

        except Exception:
            score = 0.0
            success = False

        finally:
            if not rewards:
                rewards = [0.0]

            steps = max(step_count, 1)
            log_end(success, steps, score, rewards)


if __name__ == "__main__":
    main()