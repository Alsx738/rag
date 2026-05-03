import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_openai_client():
    """
    Returns a configured OpenAI client instance using the API key from environment variables.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Missing 'OPENAI_API_KEY' in .env file.")
    return OpenAI(api_key=openai_api_key)

# Global client instance (used for embeddings and raw API calls)
client = get_openai_client()

# Model names from environment variables (with sensible defaults)
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL      = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")

# Global LangChain LLM instance (used by agents and chains)
chat_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
