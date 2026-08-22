import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.utils.logger import logger
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading LegalGraphRAG configuration and initializing...")
        config = LegalGraphRAGConfig.from_env_file(".env")

        # Override some properties based on env directly if needed
        model_name = os.getenv("model_name")
        if model_name:
            config.model.model_name = model_name
            config.model.api_key = os.getenv(f"{model_name}_api_key", os.getenv("api_key"))
            config.model.base_url = os.getenv(f"{model_name}_base_url", os.getenv("base_url"))

        graph_db_path = os.getenv("graph_db_path")
        if graph_db_path:
            config.graph.graph_db_path = graph_db_path

        rag_system = LegalGraphRAG(config=config)

        # Store in app state
        app.state.rag_system = rag_system

        logger.info("LegalGraphRAG initialized successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        app.state.rag_system = None

    yield

    # Cleanup on shutdown
    logger.info("Shutting down LegalGraphRAG API...")


app = FastAPI(
    title="LegalGraphRAG API",
    description="API for Vietnamese Civil Law analysis using Graph RAG",
    lifespan=lifespan,
)

# Fix CORS - DO NOT USE ["*"] IN PRODUCTION
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(
    ","
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount the web directories
os.makedirs("web", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount API routes
app.include_router(router)


@app.get("/")
async def root():
    # Redirect to the better, more specific UI (web)
    return RedirectResponse(url="/web/index.html")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    is_prod = os.getenv("ENVIRONMENT") == "production" or os.getenv("RENDER") == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=not is_prod)
