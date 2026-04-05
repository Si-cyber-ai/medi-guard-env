from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from env.environment import MediGuardEnv
from env.models import ActionModel


app = FastAPI()

# Single shared environment instance for a lightweight OpenEnv-compatible API.
env = MediGuardEnv()


class StepRequest(BaseModel):
    action: str


@app.post("/reset")
def reset():
    observation = env.reset()
    return {
        "observation": observation
    }


@app.post("/step")
def step(request: StepRequest):
    try:
        observation, reward, done, info = env.step(request.action)

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }

    except Exception as e:
        return {
            "error": str(e)
        }


@app.get("/state")
def state():
    return env.state()


@app.get("/")
def root():
    return {"message": "MediGuard-Env API running"}


# Run with:
# uvicorn api:app --host 0.0.0.0 --port 8000
