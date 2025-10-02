from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str | None = None

    CHAT_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    CHROMA_DIR: str = ".chroma"
    COLLECTION_NAME: str = "rag_modular"

    DEFAULT_SYSTEM_PROMPT: str = (
        "Você é um assistente técnico que responde de forma objetiva e cita fontes quando houver."
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def get_settings() -> Settings:
    return Settings()
