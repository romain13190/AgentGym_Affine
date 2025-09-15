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

@app.post("/create", response_model=dict)
async def create():
    env = await abd_env_server.create()
    return {"id": env}

@app.get("/observation", response_model=str)
async def observation(id: int):
    return await abd_env_server.observation(id)

@app.post("/step")
async def step(payload: dict):
    id_ = int(payload["id"]) ; action = payload["action"]
    obs, reward, done, info = await abd_env_server.step(id_, action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.post("/reset", response_model=str)
async def reset(payload: dict):
    id_ = int(payload["id"]) 
    return await abd_env_server.reset(id_) 