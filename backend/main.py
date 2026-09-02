from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from database.config import settings
from database.connection import engine, Base
import models 
from middlewares.timing import ProcessTimeMiddleware
from routes.predictions import router as prediction_routers
from routes.customers import router as customers_router
from routes.auth import router as auth_router
from routes.analytics import router as analytics_router

@asynccontextmanager
async def lifespan(app:FastAPI):
    print(" connectnig to Database an dcreating tables.")
    Base.metadata.create_all(bind = engine)
    print("All database tables ready in database.")
    yield
    print("server shutting down....")

app = FastAPI(
    title= settings.PROJECT_NAME,
    openapi_url = f"{settings.API_PREFIX}/openapi.json",
    docs_url="/docs",
    lifespan= lifespan
) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ProcessTimeMiddleware)

app.include_router(prediction_routers, prefix=settings.API_PREFIX)
app.include_router(customers_router, prefix=settings.API_PREFIX)
app.include_router(auth_router, prefix=settings.API_PREFIX)
app.include_router(analytics_router,prefix=settings.API_PREFIX)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
