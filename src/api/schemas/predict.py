from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1)

class PredictResponse(BaseModel):
    tg: float = Field(..., gt=0)
    ffv: float = Field(..., gt=0)
    tc: float = Field(..., gt=0)
    density: float = Field(..., gt=0)
    rg: float = Field(..., gt=0)