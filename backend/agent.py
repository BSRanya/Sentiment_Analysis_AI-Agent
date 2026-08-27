#the ReAct loop 
import asyncio
import json
import logging 
from typing import Any , Dict , List
import httpx

from config import GROQ_API_KEY , MODEL_NAME ,MAX_ITERATIONS ,TOOL_TIMEOUT ,SYSTEM_PROMPT

from tools import TOOLS , TOOL_FUNCTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class agent :
    def __init__(self):
        self.messages:List[Dict[str , Any]] = [{"role":"system", "content" : SYSTEM_PROMPT}]
        self.client=httpx.AsyncClient(
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=30.0,
        )
    async def call_llm(self) -> Dict[str , Any]:
        """Send current messages to Groq and return the assistant message."""
        payload = {
            "model":MODEL_NAME,
            "messages":self.messages,
            "tools":TOOLS,
            "tool_choice":"auto", #let LLM decide 
            "temperature":0.2,
        }
        try:
            resp = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,)
            resp.raise_for_status()
            data= resp.json()
            return data["choice"][0]["message"]
        except httpx.HTTPStatusError as e :
            logger.error(f"LLM API error: {e.response.status_code} {e.response.text}")
            raise
        except Exception as e :
            logger.error(f"LLM call failed : {e}")
            raise
    
    async def execute_tool(self, tool_call : Dict[str , Any]) -> str :
        """Execute a tool call and return the result string """
        func_name = tool_call["function"]["name"]
        args= json.loads(tool_call["function"]["arguments"])
        tool_func = TOOL_FUNCTIONS.get(func_name)
        if not tool_func:
            return f"Error: Unknown tool '{func_name}'"
        #Execute with timeout (works for both sync and async tools)

        try:
            async with asyncio.timeout(TOOL_TIMEOUT):
                #Run sync function in thread to avoid blocking event loop
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**args)