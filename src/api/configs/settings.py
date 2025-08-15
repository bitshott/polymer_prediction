from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[0]
ENV_DIR = BASE_DIR / ".env"

class Settings(BaseSettings):
    models_path: str = Field(..., env='MODELS_PATH')
    device: str = Field(..., env='DEVICE')

    model_config = SettingsConfigDict(env_file=ENV_DIR)
