from openai import OpenAI

from settings import get_settings


def create_chat_client() -> tuple[OpenAI, str]:
    settings = get_settings()
    resolved_api_key = settings.openrouter_api_key.strip()
    if not resolved_api_key:
        raise RuntimeError("Missing API key. Set OPENROUTER_API_KEY in the .env file.")

    return (
        OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=resolved_api_key,
        ),
        settings.openrouter_model,
    )
