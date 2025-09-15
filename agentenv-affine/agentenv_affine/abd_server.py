from typing import List

from fastapi import FastAPI

from .abd_environment import abd_env_server

app = FastAPI()

@app.get("/", response_model=str)
def ok():
    return "ok"

@app.get("/list_envs", response_model=List[int])
def list_envs():
    return list(abd_env_server.envs.keys())

@app.post("/create", response_model=int)
async def create():
    return await abd_env_server.create()

@app.get("/observation", response_model=str)
async def observation(env_idx: int):
    return await abd_env_server.observation(env_idx)

@app.post("/step")
async def step(payload: dict):
    env_idx = int(payload["env_idx"]) ; action = payload["action"]
    obs, reward, done, info = await abd_env_server.step(env_idx, action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.post("/reset", response_model=str)
async def reset(payload: dict):
    env_idx = int(payload["env_idx"]) 
    return await abd_env_server.reset(env_idx) 