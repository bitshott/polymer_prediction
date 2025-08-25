from dataclasses import dataclass
from typing import Literal, Any
from pathlib import Path
from exceptions import RegistryError


@dataclass(frozen=True)
class ModelID():
    name: str
    version: str 
    tag: str | None = "production"

@dataclass(frozen=True)
class ModelSpec():
    id: ModelID
    format: Literal["xgb"]
    path: Path
    target: str

class ModelRegistry():
    def __init__(self, providers: list[Any]):
        self.providers = providers
    
    def resolve(self, model_id: ModelID):
        for provider in self.providers:
            try:
                return provider.resolve(model_id)
            except RegistryError:
                continue
        raise RegistryError(f'Model not found via any provider: {model_id.name}')