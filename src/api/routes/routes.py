from fastapi import APIRouter
from api.routes.schemas.predict import PredictRequest, PredictResponse


api_v1 = APIRouter(prefix='/api/v1')

@api_v1.post('/predict', response_model=PredictResponse)
async def make_prediction(payload: PredictRequest):
    
    return {
        PredictResponse(
            tg=tg,
            ffv=ffv,
            tc=tc,
            density=density,
            rg=rg
        )
    }   