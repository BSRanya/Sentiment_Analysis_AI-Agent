import os 
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY= os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

MODEL_NAME="llama-3.1-70b-versatile"
 # How many tool call loops
MAX_ITERATIONS =3
# seconds per tool execution
TOOL_TIMEOUT= 10

SYSTEM_PROMPT = """You are a very smart, clever , detectible and helpful assistant with access to tools. 
When the user asks about the sentiment of a text, always use the 'sentiment_analysis' tool to get the result. 
Do not try to guess the sentiment yourself."""

