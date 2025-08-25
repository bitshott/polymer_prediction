from registry import ModelID, ModelSpec
from pathlib import Path
import json

class FilesystemProvider():
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.formats = {"xgb": "model.json"}

    def resolve(self, model_id: ModelID) -> ModelSpec: 
        if not model_id.version:
            alias = json.load((self.base_path / model_id.name / "aliases.json").read_text())
            model_version = alias["production"] 
        else:
            model_version = model_id.version

        metadata = json.loads((self.base_path / model_id.name / model_version / "metadata.json").read_text())
        format = metadata["format"]
        
        model_path = self.base_path / model_id.name / model_version / self.formats.get(format)
        
        
        return ModelSpec(
            id=ModelID(name=model_id.name, version=model_version, tag=model_id.tag),
            format=format,
            path=model_path,
            target=model_id.name.lower()
        )
