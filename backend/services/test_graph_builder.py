"""
测试图到 MLIR 转换
"""

import pytest
from .op_registry import get_registry
from .graph_builder import GraphBuilder, Graph, GraphNode, GraphEdge


def test_op_registry():
    """测试操作注册表"""
    registry = get_registry()
    
    # 检查常用操作（按需加载）
    assert "arith.addi" in registry
    assert "arith.constant" in registry
    assert "scf.for" in registry
    assert "func.func" in registry
    
    # 检查操作信息
    addi = registry.get("arith.addi")
    assert addi is not None
    assert addi.full_name == "arith.addi"
    assert "lhs" in addi.params
    assert "rhs" in addi.params


def test_simple_add():
    """测试简单加法：两个参数相加返回"""
    graph = Graph(
        nodes=[
            GraphNode(
                id="entry",
                op_name="function-entry",
                result_types=["i32", "i32"],
                attributes={},
                region_graphs=[],
            ),
            GraphNode(
                id="add",
                op_name="arith.addi",
                result_types=["i32"],
                attributes={},
                region_graphs=[],
            ),
            GraphNode(
                id="return",
                op_name="function-return",
                result_types=["i32"],
                attributes={},
                region_graphs=[],
            ),
        ],
        edges=[
            # entry:0 -> add:0 (lhs)
            GraphEdge(source_node="entry", source_output=0, target_node="add", target_input=0),
            # entry:1 -> add:1 (rhs)
            GraphEdge(source_node="entry", source_output=1, target_node="add", target_input=1),
            # add:0 -> return:0
            GraphEdge(source_node="add", source_output=0, target_node="return", target_input=0),
        ],
    )
    
    builder = GraphBuilder()
    module = builder.build(graph, func_name="test_add")
    
    # 验证模块
    assert module.operation.verify()
    
    # 检查生成的 MLIR
    mlir_str = str(module)
    assert "func.func @test_add" in mlir_str
    assert "arith.addi" in mlir_str
    assert "return" in mlir_str


def test_constant_and_add():
    """测试常量和加法"""
    graph = Graph(
        nodes=[
            GraphNode(
                id="entry",
                op_name="function-entry",
                result_types=["i32"],
                attributes={},
                region_graphs=[],
            ),
            GraphNode(
                id="const",
                op_name="arith.constant",
                result_types=["i32"],
                attributes={"value": "42 : i32"},
                region_graphs=[],
            ),
            GraphNode(
                id="add",
                op_name="arith.addi",
                result_types=["i32"],
                attributes={},
                region_graphs=[],
            ),
            GraphNode(
                id="return",
                op_name="function-return",
                result_types=["i32"],
                attributes={},
                region_graphs=[],
            ),
        ],
        edges=[
            GraphEdge(source_node="entry", source_output=0, target_node="add", target_input=0),
            GraphEdge(source_node="const", source_output=0, target_node="add", target_input=1),
            GraphEdge(source_node="add", source_output=0, target_node="return", target_input=0),
        ],
    )
    
    builder = GraphBuilder()
    module = builder.build(graph, func_name="test_const")
    
    assert module.operation.verify()
    
    mlir_str = str(module)
    assert "arith.constant" in mlir_str
    assert "42" in mlir_str


def test_json_format_types():
    """测试 JSON 格式类型（如 I32）自动转换为 MLIR 格式（如 i32）"""
    graph = Graph(
        nodes=[
            GraphNode(
                id="entry",
                op_name="function-entry",
                result_types=["I32"],  # JSON 格式
                attributes={},
                region_graphs=[],
            ),
            GraphNode(
                id="const",
                op_name="arith.constant",
                result_types=["I32"],  # JSON 格式
                attributes={"value": "20 : I32"},  # JSON 格式类型在属性中
                region_graphs=[],
            ),
            GraphNode(
                id="add",
                op_name="arith.addi",
                result_types=["I32"],  # JSON 格式
                attributes={},
                region_graphs=[],
            ),
            GraphNode(
                id="return",
                op_name="function-return",
                result_types=["I32"],  # JSON 格式
                attributes={},
                region_graphs=[],
            ),
        ],
        edges=[
            GraphEdge(source_node="entry", source_output=0, target_node="add", target_input=0),
            GraphEdge(source_node="const", source_output=0, target_node="add", target_input=1),
            GraphEdge(source_node="add", source_output=0, target_node="return", target_input=0),
        ],
    )
    
    builder = GraphBuilder()
    module = builder.build(graph, func_name="test_json_types")
    
    # 验证模块（类型应该被正确转换）
    assert module.operation.verify()
    
    mlir_str = str(module)
    # 生成的 MLIR 应该使用小写类型
    assert "i32" in mlir_str
    assert "20" in mlir_str


if __name__ == "__main__":
    test_op_registry()
    print("✓ test_op_registry passed")
    
    test_simple_add()
    print("✓ test_simple_add passed")
    
    test_constant_and_add()
    print("✓ test_constant_and_add passed")
    
    test_json_format_types()
    print("✓ test_json_format_types passed")
    
    print("\nAll tests passed!")
