"""
OnsetLab Playground - Backend API

FastAPI server that wraps the OnsetLab SDK for web-based interaction.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .routes import chat_router, session_router, export_router, mcp_router
from .services.session_store import session_store

# Static file directories
STATIC_DIR = Path(__file__).parent.parent / "static"
SITE_DIR = STATIC_DIR / "site"
PLAYGROUND_DIR = STATIC_DIR / "playground"

# Create app
app = FastAPI(
    title="OnsetLab Playground",
    description="Try OnsetLab agents in your browser",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include API routers
app.include_router(chat_router)
app.include_router(session_router)
app.include_router(export_router)
app.include_router(mcp_router)

# Mount static asset directories
if PLAYGROUND_DIR.exists():
    app.mount("/playground/assets", StaticFiles(directory=PLAYGROUND_DIR / "assets"), name="playground-assets")

if SITE_DIR.exists():
    app.mount("/assets", StaticFiles(directory=SITE_DIR / "assets"), name="site-assets")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    stats = session_store.stats()
    return {
        "status": "healthy",
        "sessions": stats,
    }


# Playground SPA routes
@app.get("/playground")
@app.get("/playground/{path:path}")
async def serve_playground(path: str = ""):
    """Serve the Playground SPA."""
    # Serve static files (JS, CSS, etc.)
    file_path = PLAYGROUND_DIR / path
    if path and file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    # SPA fallback
    index = PLAYGROUND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("<h1>Playground not built</h1><p>Run: cd web/frontend && npm run build</p>")


# Landing page catch-all (must be last)
@app.get("/")
@app.get("/{path:path}")
async def serve_site(path: str = ""):
    """Serve the Landing Page SPA."""
    # Serve static files
    file_path = SITE_DIR / path
    if path and file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    # SPA fallback
    index = SITE_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"name": "OnsetLab Playground API", "version": "1.0.0", "docs": "/docs"}
