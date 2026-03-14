from langchain_openai import OpenAIEmbeddings
from config import config


def get_embeddings(provider: str | None = None, model_name: str | None = None):
    try:
        provider = (provider or config.EMBED_PROVIDER).lower()
        model_name = model_name or config.EMBED_MODEL

        if provider != "openai":
            provider = "openai"

        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings")

        return OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model=model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize embeddings provider '{provider}': {exc}"
        ) from exc
