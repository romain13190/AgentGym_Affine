"""
FastAPI Server for Affine DED
"""
from typing import List

from fastapi import FastAPI

from .ded_environment import ded_env_server

app = FastAPI()


@app.get("/", response_model=str)
def generate_ok():
    return "ok"


@app.get("/list_envs", response_model=List[int])
async def list_envs():
    return list(ded_env_server.env.keys())


@app.post("/create", response_model=dict)
async def create():
    env = await ded_env_server.create()
    return {"id": env}


@app.get("/observation", response_model=str)
async def observation(id: int):
    return await ded_env_server.observation(id)


@app.post("/step")
async def step(payload: dict):
    id_ = int(payload.get("id"))
    action = payload.get("action", "")
    observation, reward, done, _ = await ded_env_server.step(id_, action)
    return {"observation": observation, "reward": reward, "done": done}


@app.post("/reset", response_model=str)
async def reset(payload: dict):
    id_ = int(payload.get("id"))
    return await ded_env_server.reset(id_, None) 