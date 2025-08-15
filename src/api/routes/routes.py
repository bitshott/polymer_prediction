from fastapi import APIRouter
from api.routes.schemas.predict import PredictRequest, PredictResponse


api_v1 = APIRouter(prefix='/api/v1')

@api_v1.post('/predict', response_model=PredictResponse)
def make_prediction(payload: PredictRequest):
    pass
    # return {
    #     PredictResponse(
    #         Tg
    #     )
    # }   