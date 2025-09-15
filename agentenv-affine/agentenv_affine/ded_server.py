"""
FastAPI Server for Affine DED
"""
from typing import List

from fastapi import FastAPI

from .ded_environment import ded_env_server
from .ded_model import CreateQuery, ResetQuery, StepQuery, StepResponse

app = FastAPI()


@app.get("/", response_model=str)
def generate_ok():
    return "ok"


@app.get("/list_envs", response_model=List[int])
async def list_envs():
    return list(ded_env_server.env.keys())


@app.post("/create", response_model=int)
async def create(create_query: CreateQuery):
    env = await ded_env_server.create(create_query.id)
    return env


@app.get("/observation", response_model=str)
async def observation(env_idx: int):
    return await ded_env_server.observation(env_idx)


@app.post("/step", response_model=StepResponse)
async def step(step_query: StepQuery):
    observation, reward, done, _ = await ded_env_server.step(
        step_query.env_idx, step_query.action
    )
    return StepResponse(observation=observation, reward=reward, done=done)


@app.post("/reset", response_model=str)
async def reset(reset_query: ResetQuery):
    return await ded_env_server.reset(reset_query.env_idx, reset_query.id) 