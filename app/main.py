"""
Main FastAPI Application Entrypoint for ChronoStore.
Initializes the dual-layer feature store, middleware, routers, and lifecycle events.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.api.routes_health import router as health_router
from app.core.config import settings
from app.core.logging import logger
from app.services.asof_engine import DataLeakageDetectedError
from app.services.registry import registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager to handle startup and shutdown sequences."""
    logger.info("Initializing ChronoStore Feature Store v%s...", settings.VERSION)
    # Ensure offline storage directory exists
    settings.get_offline_dir()
    logger.info("Offline store path verified: %s", settings.OFFLINE_STORE_DIR.resolve())
    logger.info("Online store initialized (TTL: %s).", settings.ONLINE_STORE_DEFAULT_TTL_SECONDS)
    yield
    logger.info("Shutting down ChronoStore Feature Store...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "ChronoStore is a production-grade dual-layer Feature Store solving "
        "training-serving skew and temporal future-data leakage using ultra-low latency "
        "online caching and leak-free Point-in-Time (ASOF) historical joins."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measures request execution time and appends header."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.3f}"
    return response


@app.exception_handler(DataLeakageDetectedError)
async def handle_leakage_exception(request: Request, exc: DataLeakageDetectedError):
    """Custom exception handler for strict temporal leakage violations."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "TemporalDataLeakageViolation",
            "message": str(exc),
            "detail": "Point-in-time join encountered an observation matching future feature states.",
        },
    )


# Root health and metrics endpoints for orchestrators / Kubernetes probes
app.include_router(health_router)

# Versioned API routes under /api/v1
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["System"])
def root_info():
    """Root landing endpoint with system status and API documentation links."""
    return {
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "documentation": "/docs",
        "api_v1_prefix": "/api/v1",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

