import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm(model_name: str):
    """Return LLM object based on model name"""
    models = {
        "gemma2-9b-it": "gemma2-9b-it",
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    }
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not supported")
    
    return ChatGroq(model_name=models[model_name.lower()], groq_api_key=GROQ_API_KEY)

def query_llm(llm, prompt: str):
    """Invoke LLM with user query"""
    return llm.invoke(prompt).content
