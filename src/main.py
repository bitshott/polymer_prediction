from fastapi import FastAPI
from api.routes import routes
from contextlib import asynccontextmanager
from api.routes.configs.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(routes.api_v1)