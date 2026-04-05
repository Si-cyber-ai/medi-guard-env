import os

from openai import OpenAI

from env.environment import MediGuardEnv
from env.grader import grade_episode


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")


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


def choose_action(observation):
    progress = observation["progress"]

    if not progress["analysis_done"]:
        return "analyze_case"

    if not progress["investigation_done"]:
        return "investigate_cost"

    if not progress["guidelines_checked"]:
        return "check_guidelines"

    # simple heuristic
    return "flag_issue"


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = MediGuardEnv()

    action_history = []
    rewards = []
    step_count = 0
    score = 0.0
    success = False

    log_start(task="mediguard_task", env="mediguard_env", model=MODEL_NAME)

    try:
        observation = env.reset()
        done = False

        while not done:
            step_count += 1

            action = choose_action(observation)

            observation, reward, done, info = env.step(action)

            action_history.append(action)
            rewards.append(reward)

            error = info.get("error") if isinstance(info, dict) else None
            log_step(step_count, action, reward, done, error)

        # FINAL SCORE USING GRADER
        score = grade_episode(action_history, env._hidden_truth)

        success = score > 0.5

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
