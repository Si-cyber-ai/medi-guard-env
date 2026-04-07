from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from env.environment import MediGuardEnv
from env.models import ACTION_SPACE, ReasoningBlock


app = FastAPI()

# Single shared environment instance for a lightweight OpenEnv-compatible API.
env = MediGuardEnv()


class StepRequest(BaseModel):
    action: str
    reasoning: Optional[ReasoningBlock] = None


@app.post("/reset")
def reset():
    global env
    env = MediGuardEnv()  # fresh environment per episode
    observation = env.reset()
    return {
        "observation": observation
    }


@app.post("/step")
def step(request: StepRequest):
    try:
        # Validate action early
        if request.action not in ACTION_SPACE:
            return {
                "observation": env._build_observation(),
                "reward": -0.1,
                "done": False,
                "info": {"error": "Invalid action"}
            }

        observation, reward, done, info = env.step(request.action)

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }

    except Exception as e:
        return {
            "observation": env._build_observation(),
            "reward": 0.0,
            "done": True,
            "info": {"error": str(e)}
        }


@app.get("/state")
def state():
    return env.state()


@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "MediGuard-Env"
    }


# Run with:
# uvicorn api:app --host 0.0.0.0 --port 8000
