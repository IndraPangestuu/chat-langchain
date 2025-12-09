import os
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings_model() -> Embeddings:
    """Get embeddings model with support for OpenAI-compatible APIs."""
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    
    if openai_api_base:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=200,
            openai_api_base=openai_api_base,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)
