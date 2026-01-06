"""MLIR 蓝图编辑器后端入口"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import projects, dialects, execution, graph, build, types

logger = logging.getLogger(__name__)

# mlir_data 目录路径
MLIR_DATA_DIR = Path(__file__).parent.parent / "mlir_data"


def ensure_mlir_data():
    """
    确保 mlir_data 目录存在且包含方言数据。
    如果不存在或为空，自动生成。
    """
    dialects_dir = MLIR_DATA_DIR / "dialects"
    
    # 检查是否需要生成
    needs_generation = False
    if not dialects_dir.exists():
        needs_generation = True
        logger.info("mlir_data/dialects 目录不存在，将自动生成")
    elif not any(dialects_dir.glob("*.json")):
        needs_generation = True
        logger.info("mlir_data/dialects 目录为空，将自动生成")
    
    if needs_generation:
        logger.info("正在生成方言数据...")
        try:
            from backend.mlir_utils.generator import generate_all
            generate_all(MLIR_DATA_DIR)
            logger.info("方言数据生成完成")
        except Exception as e:
            logger.error(f"生成方言数据失败: {e}")
            raise RuntimeError(f"无法生成 mlir_data: {e}") from e
    else:
        dialect_count = len(list(dialects_dir.glob("*.json")))
        logger.info(f"mlir_data 已存在，包含 {dialect_count} 个方言")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时：确保 mlir_data 存在
    ensure_mlir_data()
    yield
    # 关闭时：清理资源（如有需要）


app = FastAPI(
    title="MLIR Blueprint Editor API",
    description="MLIR 蓝图可视化编辑器后端 API",
    version="0.1.0",
    lifespan=lifespan,
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
