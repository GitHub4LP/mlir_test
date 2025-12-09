"""
MLIR Blueprint Editor - FastAPI Backend

This module provides the main FastAPI application for the MLIR Blueprint Editor.
It handles project management, dialect parsing, and MLIR execution.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import projects, dialects, execution, graph, build, types

app = FastAPI(
    title="MLIR Blueprint Editor API",
    description="Backend API for the MLIR Blueprint visual editor",
    version="0.1.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(dialects.router, prefix="/api/dialects", tags=["dialects"])
app.include_router(execution.router, prefix="/api/execution", tags=["execution"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])
app.include_router(build.router, prefix="/api/build", tags=["build"])
app.include_router(types.router, prefix="/api/types", tags=["types"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}
