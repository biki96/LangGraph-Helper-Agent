"""Embeddings service."""

from langchain_core.embeddings import Embeddings


def create_google_embeddings(
    api_key: str,
    model: str = "models/text-embedding-004",
) -> Embeddings:
    """Create Google Gemini embeddings.

    Args:
        api_key: Google API key.
        model: Model name.

    Returns:
        Embeddings instance.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
