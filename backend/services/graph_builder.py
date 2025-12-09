"""
图到 MLIR 转换器

将前端图 JSON 转换为 MLIR Module，
利用 Python bindings 的内在映射，不使用字符串拼接。
"""

from dataclasses import dataclass

from mlir import ir

from .op_registry import get_registry
from .type_registry import json_type_to_mlir
from .enum_registry import get_enum_registry


@dataclass
class GraphNode:
    """图节点"""
    id: str
    op_name: str                    # fullName: "arith.addi"
    result_types: list[str]         # 具体类型: ["i32"]
    attributes: dict[str, str]      # 属性值
    region_graphs: list["Graph"]    # 嵌套 region 的子图


@dataclass
class GraphEdge:
    """图边"""
    source_node: str
    source_output: int      # 输出索引
    target_node: str
    target_input: int       # 输入索引


@dataclass
class Graph:
    """图结构"""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    block_arg_types: list[str] = None  # region 入口 block 的参数类型


class GraphBuilder:
    """图到 MLIR 转换器"""
    
    def __init__(self) -> None:
        self.registry = get_registry()
        self.enum_registry = get_enum_registry()
        self.ctx: ir.Context | None = None
        self.ssa_map: dict[str, ir.Value] = {}  # "node_id:output_idx" → Value
    
    def build(self, graph: Graph, func_name: str = "main") -> ir.Module:
        """将图转换为 MLIR Module"""
        with ir.Context() as ctx:
            self.ctx = ctx
            with ir.Location.unknown():
                module = ir.Module.create()
                
                # 收集函数签名
                entry_node = self._find_entry_node(graph)
                return_node = self._find_return_node(graph)
                
                param_types = self._parse_types(entry_node.result_types) if entry_node else []
                return_types = self._parse_types(return_node.result_types) if return_node else []
                
                func_type = ir.FunctionType.get(param_types, return_types)
                
                with ir.InsertionPoint(module.body):
                    func_op = ir.Operation.create(
                        "func.func",
                        results=[],
                        operands=[],
                        attributes={
                            "sym_name": ir.StringAttr.get(func_name),
                            "function_type": ir.TypeAttr.get(func_type),
                        },
                        regions=1,
                    )
                    
                    # 创建入口 block
                    block = func_op.regions[0].blocks.append(*param_types)
                    
                    # 注册入口参数到 SSA 映射
                    if entry_node:
                        for i, arg in enumerate(block.arguments):
                            self.ssa_map[f"{entry_node.id}:{i}"] = arg
                    
                    # 构建函数体
                    with ir.InsertionPoint(block):
                        self._build_graph(graph, exclude_nodes={
                            entry_node.id if entry_node else None,
                            return_node.id if return_node else None,
                        } - {None})
                        
                        # 构建 return
                        if return_node:
                            return_values = self._collect_inputs(return_node, graph)
                            ir.Operation.create(
                                "func.return",
                                results=[],
                                operands=return_values,
                                attributes={},
                            )
                        else:
                            ir.Operation.create(
                                "func.return",
                                results=[],
                                operands=[],
                                attributes={},
                            )
                
                return module
    
    def _build_graph(self, graph: Graph, exclude_nodes: set[str] | None = None) -> None:
        """构建图中的操作"""
        exclude = exclude_nodes or set()
        sorted_nodes = self._topological_sort(graph, exclude)
        
        for node in sorted_nodes:
            self._build_node(node, graph)
    
    def _build_node(self, node: GraphNode, graph: Graph) -> None:
        """构建单个节点"""
        # 获取操作数
        operands = self._collect_inputs(node, graph)
        
        # 解析结果类型
        result_types = self._parse_types(node.result_types)
        
        # 解析属性（需要方言名来处理枚举）
        dialect_name = node.op_name.split(".")[0] if "." in node.op_name else None
        attributes = self._parse_attributes(node.attributes, dialect_name)
        
        # 创建操作
        op = ir.Operation.create(
            node.op_name,
            results=result_types,
            operands=operands,
            attributes=attributes,
            regions=len(node.region_graphs),
        )
        
        # 处理 regions
        for i, region_graph in enumerate(node.region_graphs):
            self._build_region(op.regions[i], region_graph)
        
        # 注册结果到 SSA 映射
        for i, result in enumerate(op.results):
            self.ssa_map[f"{node.id}:{i}"] = result
    
    def _build_region(self, region: ir.Region, graph: Graph) -> None:
        """构建 region"""
        block_arg_types = self._parse_types(graph.block_arg_types or [])
        block = region.blocks.append(*block_arg_types)
        
        # 注册 block 参数（需要特殊处理，这里简化）
        with ir.InsertionPoint(block):
            self._build_graph(graph)
    
    def _collect_inputs(self, node: GraphNode, graph: Graph) -> list[ir.Value]:
        """收集节点的输入值"""
        # 按 target_input 索引排序的边
        input_edges = sorted(
            [e for e in graph.edges if e.target_node == node.id],
            key=lambda e: e.target_input,
        )
        
        values = []
        for edge in input_edges:
            key = f"{edge.source_node}:{edge.source_output}"
            if key in self.ssa_map:
                values.append(self.ssa_map[key])
        
        return values
    
    def _parse_types(self, type_strs: list[str]) -> list[ir.Type]:
        """
        解析类型字符串列表
        
        自动处理 JSON 格式（如 "I32"）和 MLIR 格式（如 "i32"）
        """
        result = []
        for t in type_strs:
            # 使用类型注册表转换 JSON 格式到 MLIR 格式
            mlir_type_str = json_type_to_mlir(t)
            result.append(ir.Type.parse(mlir_type_str))
        return result
    
    def _parse_attributes(
        self, attrs: dict[str, any], dialect_name: str | None = None
    ) -> dict[str, ir.Attribute]:
        """
        解析属性
        
        处理三种情况：
        1. 枚举对象：{"str": "oeq", "symbol": "OEQ", "value": 1} → 整数属性
        2. 枚举字符串（兼容）：symbol（如 "OEQ"）→ 整数属性
        3. 普通属性：可能包含 JSON 格式类型（如 "20 : I32"）
        """
        result = {}
        i64 = ir.IntegerType.get_signless(64)
        
        for k, v in attrs.items():
            # 情况1: 枚举对象（完整格式）
            if isinstance(v, dict) and "value" in v and "symbol" in v:
                result[k] = ir.IntegerAttr.get(i64, v["value"])
                continue
            
            # 情况2: 枚举字符串（兼容旧格式）
            if dialect_name and isinstance(v, str):
                enum_int = self.enum_registry.get_enum_int_value(dialect_name, v)
                if enum_int is not None:
                    result[k] = ir.IntegerAttr.get(i64, enum_int)
                    continue
            
            # 情况3: 普通属性
            if isinstance(v, str):
                converted = self._convert_attr_types(v)
                result[k] = ir.Attribute.parse(converted)
            else:
                # 其他类型直接转字符串解析
                result[k] = ir.Attribute.parse(str(v))
        
        return result
    
    def _convert_attr_types(self, attr_str: str) -> str:
        """
        转换属性字符串中的 JSON 格式类型为 MLIR 格式
        
        例如: "20 : I32" -> "20 : i32"
        """
        # 处理 TypedAttr 格式: "value : type"
        if " : " in attr_str:
            parts = attr_str.rsplit(" : ", 1)
            if len(parts) == 2:
                value, type_str = parts
                mlir_type = json_type_to_mlir(type_str.strip())
                return f"{value} : {mlir_type}"
        return attr_str
    
    def _topological_sort(self, graph: Graph, exclude: set[str]) -> list[GraphNode]:
        """拓扑排序"""
        # 构建入度表
        in_degree: dict[str, int] = {}
        adjacency: dict[str, list[str]] = {}
        node_map: dict[str, GraphNode] = {}
        
        for node in graph.nodes:
            if node.id in exclude:
                continue
            in_degree[node.id] = 0
            adjacency[node.id] = []
            node_map[node.id] = node
        
        for edge in graph.edges:
            if edge.source_node in exclude or edge.target_node in exclude:
                continue
            if edge.target_node in in_degree:
                in_degree[edge.target_node] += 1
            if edge.source_node in adjacency:
                adjacency[edge.source_node].append(edge.target_node)
        
        # Kahn 算法
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            nid = queue.pop(0)
            result.append(node_map[nid])
            
            for neighbor in adjacency.get(nid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _find_entry_node(self, graph: Graph) -> GraphNode | None:
        """查找入口节点"""
        for node in graph.nodes:
            if node.op_name == "function-entry":
                return node
        return None
    
    def _find_return_node(self, graph: Graph) -> GraphNode | None:
        """查找返回节点"""
        for node in graph.nodes:
            if node.op_name == "function-return":
                return node
        return None
