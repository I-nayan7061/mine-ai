"""FastAPI Entrypoint Application.
SIH26025 - NexGen | Mine Subsidence Monitoring & Early Warning System

Serves:
- REST API at /api/*
- Real-Time WebSocket at /api/ws/telemetry
- Interactive Web Dashboard at / and /dashboard
- OpenAPI Documentation at /docs
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .routes import router as api_router
from src.config import app_config
from src.db import db_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for database initialization and cleanup."""
    # Startup
    try:
        await db_manager.init_db()
    except Exception as e:
        print(f"Warning: Database initialization error: {e}")
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Mine Subsidence AI Early Warning System",
    description="Real-Time Vibration, Tilt & Displacement Telemetry Analysis with SHAP Explainability (SIH26025 - NexGen)",
    version="1.1.0",
    lifespan=lifespan
)

# CORS configuration
origins = app_config.get("api.cors_origins", ["*"])
# If wildcard is used, allow_credentials must be False according to CORS specification
has_wildcard = "*" in origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=not has_wildcard,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

# Include API router
app.include_router(api_router)

# Mount Dashboard Static Files
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard_view():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    host = app_config.get("api.host", "0.0.0.0")
    port = int(os.environ.get("PORT", app_config.get("api.port", 8000)))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
