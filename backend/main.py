"""MLIR 蓝图编辑器后端入口"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import projects, dialects, execution, graph, build, types

app = FastAPI(
    title="MLIR Blueprint Editor API",
    description="MLIR 蓝图可视化编辑器后端 API",
    version="0.1.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
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
