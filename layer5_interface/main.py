"""
Layer 5 - Interface
FastAPI Application Entry Point

PURPOSE:
    Creates the FastAPI application.
    Configures middleware and CORS.
    Mounts the API router.
    Serves the platform as HTTP API.

RUNNING LOCALLY:
    uvicorn layer5_interface.main:app --reload
    
    Then open:
    http://localhost:8000/docs
    → Auto-generated API documentation
    → Test all endpoints in browser

RUNNING IN DOCKER:
    docker build -t abutech-platform .
    docker run -p 8000:8000 abutech-platform

API DOCUMENTATION:
    /docs    → Swagger UI (interactive)
    /redoc   → ReDoc (readable)
    Both auto-generated from Pydantic models.
    BofA integration team uses these.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from layer5_interface.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "%(name)s %(message)s"
    )
)
logger = logging.getLogger(__name__)


# ============================================================
# APPLICATION LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown.
    Runs once when platform starts.
    """
    logger.info("=" * 50)
    logger.info("AbuTech AI Security Platform")
    logger.info("Starting up...")
    logger.info("=" * 50)

    # Pre-load models at startup
    # So first request is not slow
    try:
        from layer2_ml.anomaly\
            .isolation_forest_detector\
            import IsolationForestDetector
        IsolationForestDetector()
        logger.info("Isolation Forest loaded")
    except Exception as e:
        logger.warning(
            f"Could not preload IF model: {e}"
        )

    try:
        from layer2_ml.classification\
            .pii_classifier import PIIClassifier
        PIIClassifier()
        logger.info("PII Classifier loaded")
    except Exception as e:
        logger.warning(
            f"Could not preload PII classifier: {e}"
        )

    logger.info("Platform ready")
    logger.info("API docs: http://localhost:8000/docs")

    yield  # Application runs here

    logger.info("Platform shutting down...")


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AbuTech AI Security Platform",
    description=(
        "Enterprise AI Security Platform with "
        "LSTM + Attention, Knowledge Graph, "
        "GNN Threat Detection, and LLM Agents.\n\n"
        "**Compliance**: SR 11-7 OCC, GDPR, "
        "HIPAA, PCI-DSS, CCPA\n\n"
        "**University of Arizona** | "
        "Abuhari Consulting Services LLC"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Health",
            "description": (
                "Platform health and status endpoints. "
                "Used by Docker and Kubernetes."
            )
        },
        {
            "name": "Ingestion",
            "description": (
                "Event ingestion endpoints. "
                "Normalize, score, and enrich "
                "security events from any source."
            )
        },
        {
            "name": "Investigation",
            "description": (
                "LLM-powered investigation endpoints. "
                "Trigger Layer 4 agents for "
                "full incident analysis."
            )
        },
        {
            "name": "Knowledge Graph",
            "description": (
                "Query the security knowledge graph. "
                "Nodes, edges, and threat relationships."
            )
        },
        {
            "name": "Dashboard",
            "description": (
                "Data endpoints for Streamlit dashboard. "
                "Real-time feed and statistics."
            )
        }
    ]
)

# ============================================================
# MIDDLEWARE
# ============================================================

# CORS — allows Streamlit to call the API
# In production: restrict to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Request timing middleware
# Logs how long each request takes
# Performance monitoring for SR 11-7
@app.middleware("http")
async def add_process_time(
    request: Request,
    call_next
):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    response.headers[
        "X-Process-Time"
    ] = str(process_time)

    if process_time > 5.0:
        logger.warning(
            f"SLOW REQUEST: "
            f"{request.method} {request.url.path} "
            f"took {process_time:.2f}s"
        )

    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):
    logger.error(
        f"Unhandled exception: {exc}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path)
        }
    )


# ============================================================
# MOUNT ROUTES
# ============================================================

# All routes prefixed with /api/v1
# Versioning allows future /api/v2 without breaking
app.include_router(
    router,
    prefix="/api/v1"
)


# Root endpoint
@app.get("/", tags=["Health"])
async def root():
    return {
        "platform": "AbuTech AI Security Platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "university": "University of Arizona",
        "company": "Abuhari Consulting Services LLC"
    }