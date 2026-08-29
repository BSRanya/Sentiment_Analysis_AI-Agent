from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from agent import Agent

app = FastAPI(title="Sentiment Analysis Agent API ")

# Allow frontend origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message : str

class ChatResponse(BaseModel):
    response : str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request : ChatRequest):
    """
    Send a user message to the agent and get the final answer .
    Creates a fresh agent per request (stateless)"""

    agent=Agent()
    try:
        answer = await agent.run(request.message)
        return ChatResponse(response = answer)
    except Exception as e :
        raise HTTPException (status_code=500 , detail=str(e))
    finally:
        await agent.close()