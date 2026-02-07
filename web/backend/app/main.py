"""
OnsetLab Playground - Backend API

FastAPI server that wraps the OnsetLab SDK for web-based interaction.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import chat_router, session_router, export_router, mcp_router
from .services.session_store import session_store

# Create app
app = FastAPI(
    title="OnsetLab Playground",
    description="Try OnsetLab agents in your browser",
    version="1.0.0",
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

# Include routers
app.include_router(chat_router)
app.include_router(session_router)
app.include_router(export_router)
app.include_router(mcp_router)


@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "OnsetLab Playground API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    stats = session_store.stats()
    return {
        "status": "healthy",
        "sessions": stats,
    }


# For development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
