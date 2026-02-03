"""LLM service."""

from langchain_core.language_models import BaseChatModel


def create_google_llm(
    api_key: str,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create Google Gemini chat model.

    Args:
        api_key: Google API key.
        model: Model name.
        temperature: Sampling temperature.

    Returns:
        Chat model instance.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )
