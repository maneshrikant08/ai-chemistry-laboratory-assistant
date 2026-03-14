from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config import config


def get_llm(provider: str | None = None, model_name: str | None = None):
    try:
        provider = (provider or config.DEFAULT_PROVIDER).lower()
        model_name = model_name or config.DEFAULT_MODEL

        if provider == "openai":
            if not config.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI models")
            return ChatOpenAI(api_key=config.OPENAI_API_KEY, model=model_name)

        if provider == "groq":
            if not config.GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is required for Groq models")
            return ChatGroq(api_key=config.GROQ_API_KEY, model=model_name)

        if provider == "gemini":
            if not config.GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY is required for Gemini models")
            return ChatGoogleGenerativeAI(
                google_api_key=config.GEMINI_API_KEY, model=model_name
            )

        raise RuntimeError(f"Unsupported provider: {provider}")
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize LLM provider '{provider}': {exc}") from exc
