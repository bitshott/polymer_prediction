from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[0]
ENV_DIR = BASE_DIR / ".env"

class Settings(BaseSettings):
    MODELS_BASE_PATH: str = Field(..., env='MODELS_BASE_PATH')
    DEVICE: str = Field(..., env='DEVICE')

    model_config = SettingsConfigDict(env_file=ENV_DIR)
