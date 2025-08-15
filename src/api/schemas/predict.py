from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1)

class PredictResponse(BaseModel):
    Tg: float = Field(..., gt=0)
    FFV: float = Field(..., gt=0)
    Tc: float = Field(..., gt=0)
    Density: float = Field(..., gt=0)
    Rg: float = Field(..., gt=0)