import asyncio
import json
import logging
from typing import Any, Dict, List
import httpx

from config import GROQ_API_KEY, MODEL_NAME, MAX_ITERATIONS, TOOL_TIMEOUT, SYSTEM_PROMPT
from tools import TOOLS, TOOL_FUNCTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=30.0,
        )
        # ponytail: track consecutive identical tool calls to break stuck loops
        self._last_calls: List[tuple] = []

    async def call_llm(self) -> Dict[str, Any]:
        payload = {
            "model": MODEL_NAME,
            "messages": self.messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        try:
            resp = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API error: {e.response.status_code} {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def execute_tool(self, tool_call: Dict[str, Any]) -> str:
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        tool_func = TOOL_FUNCTIONS.get(func_name)
        if not tool_func:
            return f"Error: Unknown tool '{func_name}'"
        try:
            async with asyncio.timeout(TOOL_TIMEOUT):
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**args)
                else:
                    result = await asyncio.to_thread(tool_func, **args)
                if not isinstance(result, str):
                    result = json.dumps(result)
                return result
        except asyncio.TimeoutError:
            return f"Error: Tool '{func_name}' timed out after {TOOL_TIMEOUT}s"
        except Exception as e:
            logger.exception(f"Tool {func_name} failed: {e}")
            return f"Error: Tool execution failed: {str(e)}"

    def _is_stuck(self) -> bool:
        # ponytail: if last 3 consecutive tool calls are identical (name+args), break
        if len(self._last_calls) < 3:
            return False
        return len(set(self._last_calls[-3:])) == 1

    async def run(self, user_query: str) -> str:
        self.messages.append({"role": "user", "content": user_query})

        for iteration in range(MAX_ITERATIONS):
            logger.info(f"--- Iteration {iteration + 1} ---")
            assistant_msg = await self.call_llm()
            self.messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                return assistant_msg.get("content", "No response.")

            for tool_call in tool_calls:
                tool_call_id = tool_call["id"]
                func_name = tool_call["function"]["name"]
                args_str = tool_call["function"]["arguments"]
                logger.info(f"Calling tool: {func_name} with args: {args_str}")

                result = await self.execute_tool(tool_call)
                logger.info(f"Tool result: {result[:200]}...")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                })

                # ponytail: track calls, break if stuck
                self._last_calls.append((func_name, args_str))
                if len(self._last_calls) > 10:
                    self._last_calls.pop(0)
                if self._is_stuck():
                    logger.warning("Agent stuck in loop — same tool call repeated 3x. Breaking.")
                    return "I seem to be stuck in a loop. Please try a different question."

        return "Max iterations reached without final answer."

    async def close(self):
        await self.client.aclose()
