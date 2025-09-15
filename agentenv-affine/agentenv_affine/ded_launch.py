import uvicorn
 
if __name__ == "__main__":
    uvicorn.run("agentenv_affine.ded_server:app", host="0.0.0.0", port=8010, reload=False) 