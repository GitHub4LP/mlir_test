"""
图执行 API

接收前端图 JSON，转换为 MLIR，执行并返回结果。
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.graph_builder import GraphBuilder, Graph, GraphNode, GraphEdge
from backend.api.execution import _execute_compile, _execute_jit

router = APIRouter()


class GraphNodeModel(BaseModel):
    """图节点模型"""
    id: str
    op_name: str                        # fullName: "arith.addi"
    result_types: list[str]             # 具体类型: ["i32"]
    attributes: dict[str, str] = {}     # 属性值
    region_graphs: list["GraphModel"] = []


class GraphEdgeModel(BaseModel):
    """图边模型"""
    source_node: str
    source_output: int
    target_node: str
    target_input: int


class GraphModel(BaseModel):
    """图模型"""
    nodes: list[GraphNodeModel]
    edges: list[GraphEdgeModel]
    block_arg_types: list[str] | None = None


class ExecuteGraphRequest(BaseModel):
    """执行图请求"""
    graph: GraphModel
    func_name: str = "main"
    mode: str = "jit"  # "jit" | "compile"


class ExecuteGraphResponse(BaseModel):
    """执行图响应"""
    success: bool
    mlir_code: str
    output: str
    error: str | None = None


def _convert_graph(model: GraphModel) -> Graph:
    """将 Pydantic 模型转换为内部数据结构"""
    nodes = [
        GraphNode(
            id=n.id,
            op_name=n.op_name,
            result_types=n.result_types,
            attributes=n.attributes,
            region_graphs=[_convert_graph(rg) for rg in n.region_graphs],
        )
        for n in model.nodes
    ]
    
    edges = [
        GraphEdge(
            source_node=e.source_node,
            source_output=e.source_output,
            target_node=e.target_node,
            target_input=e.target_input,
        )
        for e in model.edges
    ]
    
    return Graph(
        nodes=nodes,
        edges=edges,
        block_arg_types=model.block_arg_types,
    )


@router.post("/execute", response_model=ExecuteGraphResponse)
async def execute_graph(request: ExecuteGraphRequest):
    """
    执行图
    
    1. 将图 JSON 转换为 MLIR Module
    2. 验证 Module
    3. 执行并返回结果
    """
    try:
        # 转换图
        graph = _convert_graph(request.graph)
        
        # 构建 MLIR
        builder = GraphBuilder()
        module = builder.build(graph, func_name=request.func_name)
        
        # 验证
        if not module.operation.verify():
            return ExecuteGraphResponse(
                success=False,
                mlir_code=str(module),
                output="",
                error="MLIR verification failed",
            )
        
        mlir_code = str(module)
        
        # 执行
        if request.mode == "jit":
            result = await _execute_jit(mlir_code)
        else:
            result = await _execute_compile(mlir_code)
        
        return ExecuteGraphResponse(
            success=result.success,
            mlir_code=mlir_code,
            output=result.output,
            error=result.error,
        )
        
    except Exception as e:
        return ExecuteGraphResponse(
            success=False,
            mlir_code="",
            output="",
            error=str(e),
        )


@router.post("/build", response_model=dict)
async def build_graph(request: ExecuteGraphRequest):
    """
    仅构建 MLIR，不执行
    
    用于调试和预览生成的 MLIR 代码。
    """
    try:
        graph = _convert_graph(request.graph)
        builder = GraphBuilder()
        module = builder.build(graph, func_name=request.func_name)
        
        verified = module.operation.verify()
        
        return {
            "success": verified,
            "mlir_code": str(module),
            "verified": verified,
        }
        
    except Exception as e:
        return {
            "success": False,
            "mlir_code": "",
            "verified": False,
            "error": str(e),
        }


# 允许递归模型
GraphNodeModel.model_rebuild()
GraphModel.model_rebuild()
