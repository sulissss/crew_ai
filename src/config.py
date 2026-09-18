import os
from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama

load_dotenv()

os.environ.setdefault("OPENAI_API_KEY", "NA")

LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

def get_llm():
    return Ollama(model=LLM_MODEL)
