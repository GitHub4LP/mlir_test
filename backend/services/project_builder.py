"""
项目级 MLIR 构建器

将整个项目（多函数）转换为 MLIR Module。
支持函数调用、依赖分析、拓扑排序。

设计原则：
- 项目级构建，而非单函数构建
- 统一处理所有节点类型
- 自动依赖分析和排序
- 利用 MLIR Python bindings 的内在映射
"""

from dataclasses import dataclass, field
from typing import Any

from mlir import ir
from mlir.dialects import func, arith

from .type_registry import json_type_to_mlir


@dataclass
class PortConfig:
    """端口配置"""
    id: str
    name: str
    kind: str  # 'input' | 'output'
    typeConstraint: str
    concreteType: str | None = None


@dataclass
class GraphNode:
    """图节点 - 统一模型"""
    id: str
    type: str  # 'operation' | 'function-entry' | 'function-return' | 'function-call'
    data: dict[str, Any]


@dataclass
class GraphEdge:
    """图边"""
    id: str
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


@dataclass
class FunctionGraph:
    """函数图"""
    nodes: list[GraphNode]
    edges: list[GraphEdge]


@dataclass
class FunctionDef:
    """函数定义"""
    id: str
    name: str
    parameters: list[dict[str, str]]  # [{name, type}]
    returnTypes: list[dict[str, str]]  # [{name, type}]
    graph: FunctionGraph
    isMain: bool = False


@dataclass
class Project:
    """项目定义"""
    name: str
    path: str
    mainFunction: FunctionDef
    customFunctions: list[FunctionDef]
    dialects: list[str] = field(default_factory=list)


class ProjectBuilder:
    """项目级 MLIR 构建器"""
    
    def __init__(self) -> None:
        self.ctx: ir.Context | None = None
        self.module: ir.Module | None = None
        # 函数映射：function_id -> FuncOp
        self.func_ops: dict[str, Any] = {}
        # SSA 值映射："{node_id}:{output_handle}" -> Value
        self.ssa_map: dict[str, ir.Value] = {}
    
    def build(self, project: Project) -> ir.Module:
        """构建整个项目为 MLIR Module"""
        with ir.Context() as ctx:
            self.ctx = ctx
            with ir.Location.unknown():
                self.module = ir.Module.create()
                
                # 1. 收集所有函数
                all_functions = self._collect_functions(project)
                
                # 2. 分析依赖并拓扑排序
                sorted_functions = self._topological_sort_functions(all_functions)
                
                # 3. 逐个生成函数
                with ir.InsertionPoint(self.module.body):
                    for func_def in sorted_functions:
                        self._build_function(func_def)
                
                return self.module
    
    def _collect_functions(self, project: Project) -> dict[str, FunctionDef]:
        """收集所有函数定义"""
        functions = {project.mainFunction.id: project.mainFunction}
        for f in project.customFunctions:
            functions[f.id] = f
        return functions
    
    def _topological_sort_functions(
        self, functions: dict[str, FunctionDef]
    ) -> list[FunctionDef]:
        """
        拓扑排序函数，确保被调用者先生成
        
        分析每个函数图中的 function-call 节点，构建调用图
        """
        # 构建调用图
        call_graph: dict[str, set[str]] = {fid: set() for fid in functions}
        
        for fid, func_def in functions.items():
            for node in func_def.graph.nodes:
                if node.type == 'function-call':
                    callee_id = node.data.get('functionId')
                    if callee_id and callee_id in functions:
                        call_graph[fid].add(callee_id)
        
        # Kahn 算法拓扑排序
        in_degree = {fid: 0 for fid in functions}
        for fid, callees in call_graph.items():
            for callee in callees:
                in_degree[fid] += 1  # fid 依赖 callee
        
        # 反转：被依赖者先输出
        reverse_graph: dict[str, set[str]] = {fid: set() for fid in functions}
        for fid, callees in call_graph.items():
            for callee in callees:
                reverse_graph[callee].add(fid)
        
        # 重新计算入度（基于反转图）
        in_degree = {fid: len(call_graph[fid]) for fid in functions}
        
        queue = [fid for fid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            fid = queue.pop(0)
            result.append(functions[fid])
            
            for dependent in reverse_graph[fid]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # 如果有循环依赖，剩余的函数按原顺序添加
        remaining = [f for fid, f in functions.items() if f not in result]
        result.extend(remaining)
        
        return result
    
    def _build_function(self, func_def: FunctionDef) -> None:
        """构建单个函数"""
        # 清空当前函数的 SSA 映射
        self.ssa_map.clear()
        
        # 解析参数和返回类型
        param_types = self._parse_types([p['type'] for p in func_def.parameters])
        return_types = self._parse_types([r['type'] for r in func_def.returnTypes])
        
        # 创建函数类型和函数操作
        func_type = ir.FunctionType.get(param_types, return_types)
        func_op = func.FuncOp(func_def.name, func_type)
        
        # 保存函数引用（供 function-call 使用）
        self.func_ops[func_def.id] = func_op
        
        # 创建入口 block
        block = func_op.add_entry_block()
        
        # 查找 entry 和 return 节点
        entry_node = self._find_node_by_type(func_def.graph, 'function-entry')
        return_node = self._find_node_by_type(func_def.graph, 'function-return')
        
        # 注册入口参数到 SSA 映射
        if entry_node:
            outputs = entry_node.data.get('outputs', [])
            for i, (arg, port) in enumerate(zip(block.arguments, outputs)):
                self.ssa_map[f"{entry_node.id}:{port['id']}"] = arg
        
        # 构建函数体
        with ir.InsertionPoint(block):
            # 拓扑排序节点
            exclude = {entry_node.id if entry_node else None, 
                      return_node.id if return_node else None} - {None}
            sorted_nodes = self._topological_sort_nodes(func_def.graph, exclude)
            
            # 构建每个节点
            for node in sorted_nodes:
                self._build_node(node, func_def.graph)
            
            # 构建 return
            if return_node:
                return_values = self._collect_node_inputs(return_node, func_def.graph)
                func.ReturnOp(return_values)
            else:
                func.ReturnOp([])
    
    def _build_node(self, node: GraphNode, graph: FunctionGraph) -> None:
        """构建单个节点 - 统一处理所有类型"""
        if node.type == 'operation':
            self._build_operation_node(node, graph)
        elif node.type == 'function-call':
            self._build_function_call_node(node, graph)
        # function-entry 和 function-return 在 _build_function 中特殊处理
    
    def _build_operation_node(self, node: GraphNode, graph: FunctionGraph) -> None:
        """构建 MLIR 操作节点"""
        data = node.data
        
        # 新格式：直接使用 fullName
        op_name = data.get('fullName', '')
        
        # 收集输入
        operands = self._collect_node_inputs(node, graph)
        
        # 解析结果类型 - 直接从 outputTypes 获取
        output_types = data.get('outputTypes', {})
        result_types = self._parse_types(list(output_types.values()))
        
        # 解析属性
        attributes = self._parse_attributes(data.get('attributes', {}), output_types)
        
        # 创建操作
        op = ir.Operation.create(
            op_name,
            results=result_types,
            operands=operands,
            attributes=attributes,
        )
        
        # 注册结果到 SSA 映射 - 使用 outputTypes 的键作为结果名称
        for i, (result, result_name) in enumerate(zip(op.results, output_types.keys())):
            handle = f"output-{result_name}"
            self.ssa_map[f"{node.id}:{handle}"] = result
    
    def _build_function_call_node(self, node: GraphNode, graph: FunctionGraph) -> None:
        """构建函数调用节点"""
        data = node.data
        callee_id = data.get('functionId', '')
        
        # 获取被调用函数
        callee_func = self.func_ops.get(callee_id)
        if not callee_func:
            raise ValueError(f"Function '{callee_id}' not found")
        
        # 收集输入参数
        arguments = self._collect_node_inputs(node, graph)
        
        # 创建 func.call
        call_op = func.CallOp(callee_func, arguments)
        
        # 注册结果到 SSA 映射
        outputs = data.get('outputs', [])
        for i, (result, port) in enumerate(zip(call_op.results, outputs)):
            self.ssa_map[f"{node.id}:{port['id']}"] = result
    
    def _collect_node_inputs(self, node: GraphNode, graph: FunctionGraph) -> list[ir.Value]:
        """收集节点的输入值"""
        # 找到所有指向此节点的边
        input_edges = [e for e in graph.edges if e.target == node.id]
        
        # 过滤执行边
        data_edges = [e for e in input_edges if not e.targetHandle.startswith('exec-')]
        
        # 按输入端口排序
        # 对于 operation 节点，需要按 operation.arguments 中 operand 的顺序
        # 对于 function-call 节点，需要按 inputs 的顺序
        # 对于 function-return 节点，需要按 inputs 的顺序
        
        if node.type == 'operation':
            return self._collect_operation_inputs(node, data_edges)
        elif node.type == 'function-call':
            return self._collect_function_call_inputs(node, data_edges)
        elif node.type == 'function-return':
            return self._collect_function_return_inputs(node, data_edges)
        
        return []
    
    def _collect_operation_inputs(
        self, node: GraphNode, edges: list[GraphEdge]
    ) -> list[ir.Value]:
        """收集操作节点的输入
        
        使用 inputTypes 的键顺序来确定操作数顺序。
        Python 3.7+ 的 dict 保持插入顺序，前端保存时会保持正确顺序。
        """
        input_types = node.data.get('inputTypes', {})
        
        # 构建 handle -> edge 映射
        edge_map = {e.targetHandle: e for e in edges}
        
        values = []
        for input_name in input_types.keys():
            handle = f"input-{input_name}"
            edge = edge_map.get(handle)
            if edge:
                key = f"{edge.source}:{edge.sourceHandle}"
                if key in self.ssa_map:
                    values.append(self.ssa_map[key])
        
        return values
    
    def _collect_function_call_inputs(
        self, node: GraphNode, edges: list[GraphEdge]
    ) -> list[ir.Value]:
        """收集函数调用节点的输入"""
        inputs = node.data.get('inputs', [])
        
        # 构建 handle -> edge 映射
        edge_map = {e.targetHandle: e for e in edges}
        
        values = []
        for port in inputs:
            edge = edge_map.get(port['id'])
            if edge:
                key = f"{edge.source}:{edge.sourceHandle}"
                if key in self.ssa_map:
                    values.append(self.ssa_map[key])
        
        return values
    
    def _collect_function_return_inputs(
        self, node: GraphNode, edges: list[GraphEdge]
    ) -> list[ir.Value]:
        """收集函数返回节点的输入"""
        inputs = node.data.get('inputs', [])
        
        # 构建 handle -> edge 映射
        edge_map = {e.targetHandle: e for e in edges}
        
        values = []
        for port in inputs:
            edge = edge_map.get(port['id'])
            if edge:
                key = f"{edge.source}:{edge.sourceHandle}"
                if key in self.ssa_map:
                    values.append(self.ssa_map[key])
        
        return values
    
    def _parse_types(self, type_strs: list[str]) -> list[ir.Type]:
        """解析类型字符串列表"""
        result = []
        for t in type_strs:
            mlir_type_str = json_type_to_mlir(t)
            result.append(ir.Type.parse(mlir_type_str))
        return result
    
    def _parse_attributes(
        self, attrs: dict[str, Any], output_types: dict[str, str]
    ) -> dict[str, ir.Attribute]:
        """解析属性"""
        result = {}
        for k, v in attrs.items():
            if v is None or v == '':
                continue
            
            # 处理 TypedAttrInterface（如 arith.constant 的 value）
            attr_str = str(v)
            if not ':' in attr_str and output_types:
                # 需要添加类型后缀
                first_type = next(iter(output_types.values()), None)
                if first_type:
                    mlir_type = json_type_to_mlir(first_type)
                    attr_str = f"{v} : {mlir_type}"
            else:
                # 转换属性中的类型
                attr_str = self._convert_attr_types(attr_str)
            
            result[k] = ir.Attribute.parse(attr_str)
        
        return result
    
    def _convert_attr_types(self, attr_str: str) -> str:
        """转换属性字符串中的类型"""
        if ' : ' in attr_str:
            parts = attr_str.rsplit(' : ', 1)
            if len(parts) == 2:
                value, type_str = parts
                mlir_type = json_type_to_mlir(type_str.strip())
                return f"{value} : {mlir_type}"
        return attr_str
    
    def _find_node_by_type(self, graph: FunctionGraph, node_type: str) -> GraphNode | None:
        """查找指定类型的节点"""
        for node in graph.nodes:
            if node.type == node_type:
                return node
        return None
    
    def _topological_sort_nodes(
        self, graph: FunctionGraph, exclude: set[str]
    ) -> list[GraphNode]:
        """拓扑排序节点"""
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
            # 跳过执行边
            if edge.sourceHandle.startswith('exec-') or edge.targetHandle.startswith('exec-'):
                continue
            if edge.source in exclude or edge.target in exclude:
                continue
            if edge.target in in_degree:
                in_degree[edge.target] += 1
            if edge.source in adjacency:
                adjacency[edge.source].append(edge.target)
        
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


def build_project_from_dict(project_dict: dict) -> ir.Module:
    """从字典构建项目"""
    # 转换为数据类
    def parse_graph(g: dict) -> FunctionGraph:
        nodes = [GraphNode(id=n['id'], type=n['type'], data=n['data']) for n in g['nodes']]
        edges = [GraphEdge(**e) for e in g['edges']]
        return FunctionGraph(nodes=nodes, edges=edges)
    
    def parse_function(f: dict) -> FunctionDef:
        return FunctionDef(
            id=f['id'],
            name=f['name'],
            parameters=f.get('parameters', []),
            returnTypes=f.get('returnTypes', []),
            graph=parse_graph(f['graph']),
            isMain=f.get('isMain', False),
        )
    
    project = Project(
        name=project_dict['name'],
        path=project_dict['path'],
        mainFunction=parse_function(project_dict['mainFunction']),
        customFunctions=[parse_function(f) for f in project_dict.get('customFunctions', [])],
        dialects=project_dict.get('dialects', []),
    )
    
    builder = ProjectBuilder()
    return builder.build(project)
