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
from mlir.dialects import func

from .attribute_converter import convert_attributes
from .type_registry import json_type_to_mlir
from backend.api.types import (
    _build_all_constraint_defs, 
    _expand_rule_to_types, 
)
from backend.api.constraint_utils import get_buildable_types
from backend.api.dialects import load_dialect


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
    parameters: list[dict[str, str]]  # [{name, constraint}]
    returnTypes: list[dict[str, str]]  # [{name, constraint}]
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


@dataclass
class PendingFunction:
    """待构建的函数实例"""
    func_def: FunctionDef
    input_types: list[str]
    output_types: list[str]
    graph_with_types: FunctionGraph  # 类型传播后的图副本


class ProjectBuilder:
    """项目级 MLIR 构建器
    
    设计原则：
    - 两阶段构建：先收集所有函数实例，再统一构建
    - 用 (func_id, input_types, output_types) 作为唯一 key
    - 每个函数有独立的 SSA 作用域
    """
    
    def __init__(self) -> None:
        self.ctx: ir.Context | None = None
        self.module: ir.Module | None = None
        # SSA 值映射："{node_id}:{output_handle}" -> Value（每个函数独立）
        self.ssa_map: dict[str, ir.Value] = {}
        # 约束定义缓存（用于获取约束的具体类型）
        self._constraint_defs: dict[str, Any] | None = None
        self._buildable_types: set[str] | None = None
        # 统一函数映射：(func_id, input_types, output_types) -> FuncOp
        self.built_functions: dict[tuple[str, tuple[str, ...], tuple[str, ...]], Any] = {}
        # 所有函数定义（用于递归查找）
        self.all_functions: dict[str, FunctionDef] = {}
        # 操作定义缓存：fullName -> OperationDef
        self._operation_defs: dict[str, Any] = {}
        # 待构建函数列表（阶段1收集，阶段2构建）
        self._pending_functions: list[PendingFunction] = []
        self._visited_keys: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    
    def build(self, project: Project) -> ir.Module:
        """构建整个项目为 MLIR Module
        
        两阶段构建：
        1. 收集阶段：从 main 开始 DFS，收集所有需要构建的函数实例
        2. 构建阶段：逆序构建（先构建被依赖的函数）
        """
        with ir.Context() as ctx:
            self.ctx = ctx
            with ir.Location.unknown():
                self.module = ir.Module.create()
                
                # 收集所有函数定义
                self.all_functions = self._collect_functions(project)
                
                # 阶段1：收集所有需要构建的函数实例
                self._pending_functions = []
                self._visited_keys = set()
                self._collect_pending_functions(project.mainFunction)
                
                print(f"[BUILD] Collected {len(self._pending_functions)} function instances")
                
                # 阶段2：正序构建（DFS 后序保证被依赖的函数先添加）
                for pending in self._pending_functions:
                    self._build_single_function(pending)
                
                return self.module
    
    def _collect_pending_functions(self, func_def: FunctionDef) -> None:
        """阶段1：从函数签名解析类型，开始收集"""
        buildable = self._get_buildable_types_set()
        
        # 解析参数类型
        input_types: list[str] = []
        for p in func_def.parameters:
            constraint = p['constraint']
            if constraint in buildable:
                input_types.append(constraint)
            else:
                concrete_types = self._get_concrete_types(constraint)
                if not concrete_types:
                    raise ValueError(f"Cannot resolve constraint '{constraint}' for parameter {p['name']}")
                input_types.append(concrete_types[0])
        
        # 解析返回类型
        output_types: list[str] = []
        for r in func_def.returnTypes:
            constraint = r['constraint']
            if constraint in buildable:
                output_types.append(constraint)
            else:
                concrete_types = self._get_concrete_types(constraint)
                if not concrete_types:
                    raise ValueError(f"Cannot resolve constraint '{constraint}' for return {r['name']}")
                output_types.append(concrete_types[0])
        
        self._collect_pending_functions_with_types(func_def, input_types, output_types)
    
    def _collect_pending_functions_with_types(
        self,
        func_def: FunctionDef,
        input_types: list[str],
        output_types: list[str]
    ) -> None:
        """阶段1：收集函数实例（DFS）"""
        type_key = (func_def.id, tuple(input_types), tuple(output_types))
        
        # 已访问，跳过
        if type_key in self._visited_keys:
            return
        self._visited_keys.add(type_key)
        
        func_name = func_def.name if func_def.isMain else self._generate_specialized_name(
            func_def.name, input_types, output_types
        )
        print(f"[COLLECT] func={func_name} inputs={input_types} outputs={output_types}")
        
        # 类型传播，得到带类型的图副本
        graph_with_types = self._propagate_types_in_function_graph(
            func_def.graph,
            self._find_node_by_type(func_def.graph, 'function-entry'),
            input_types,
            output_types,
            func_def
        )
        
        # 递归处理所有 function-call 节点
        for node in graph_with_types.nodes:
            if node.type == 'function-call':
                callee_id = node.data.get('functionId', '')
                callee_def = self.all_functions.get(callee_id)
                if not callee_def:
                    raise ValueError(f"Function '{callee_id}' not found")
                
                # 推断调用点类型
                call_types = self._infer_call_site_types(node, graph_with_types)
                if call_types:
                    self._collect_pending_functions_with_types(
                        callee_def,
                        call_types['input_types'],
                        call_types['output_types']
                    )
        
        # 添加到待构建列表（后添加的先构建，所以 main 最后构建）
        self._pending_functions.append(PendingFunction(
            func_def=func_def,
            input_types=input_types,
            output_types=output_types,
            graph_with_types=graph_with_types
        ))
    
    def _build_single_function(self, pending: PendingFunction) -> None:
        """阶段2：构建单个函数（独立的 SSA 作用域）"""
        func_def = pending.func_def
        input_types = pending.input_types
        output_types = pending.output_types
        graph_with_types = pending.graph_with_types
        
        type_key = (func_def.id, tuple(input_types), tuple(output_types))
        
        # 生成函数名
        if func_def.isMain or func_def.id == 'main':
            func_name = func_def.name
        else:
            func_name = self._generate_specialized_name(func_def.name, input_types, output_types)
        
        print(f"[BUILD] func={func_name}")
        
        # 解析类型
        param_types = self._parse_types(input_types)
        return_types = self._parse_types(output_types)
        
        # 在 module.body 级别创建函数
        with ir.InsertionPoint(self.module.body):
            func_type = ir.FunctionType.get(param_types, return_types)
            func_op = func.FuncOp(func_name, func_type)
        
        # 保存函数引用
        self.built_functions[type_key] = func_op
        
        # 清空 SSA 映射（每个函数独立）
        self.ssa_map.clear()
        
        # 创建入口 block
        block = func_op.add_entry_block()
        
        # 找到 entry 和 return 节点
        entry_node = self._find_node_by_type(graph_with_types, 'function-entry')
        return_node = self._find_node_by_type(graph_with_types, 'function-return')
        
        # 注册入口参数到 SSA 映射
        if entry_node:
            outputs = entry_node.data.get('outputs', [])
            if not outputs:
                outputs = [
                    {'id': f'data-out-{p["name"]}', 'name': p['name']}
                    for p in func_def.parameters
                ]
            for i, (arg, port) in enumerate(zip(block.arguments, outputs)):
                self.ssa_map[f"{entry_node.id}:{port['id']}"] = arg
        
        # 构建函数体
        with ir.InsertionPoint(block):
            # 拓扑排序节点
            exclude = {entry_node.id if entry_node else None,
                      return_node.id if return_node else None} - {None}
            sorted_nodes = self._topological_sort_nodes(graph_with_types, exclude)
            
            # 构建每个节点
            for node in sorted_nodes:
                self._build_node(node, graph_with_types)
            
            # 构建 return
            if return_node:
                return_values = self._collect_node_inputs(return_node, graph_with_types, func_def)
                func.ReturnOp(return_values)
            else:
                func.ReturnOp([])
    
    def _collect_functions(self, project: Project) -> dict[str, FunctionDef]:
        """收集所有函数定义"""
        functions = {project.mainFunction.id: project.mainFunction}
        for f in project.customFunctions:
            functions[f.id] = f
        return functions
    
    def _get_constraint_defs(self) -> dict[str, Any]:
        """获取约束定义映射（延迟加载）"""
        if self._constraint_defs is None:
            defs = _build_all_constraint_defs()
            self._constraint_defs = {d.name: d for d in defs}
        return self._constraint_defs
    
    def _get_buildable_types_set(self) -> set[str]:
        """获取 BuildableType 集合（延迟加载）"""
        if self._buildable_types is None:
            self._buildable_types = set(get_buildable_types())
        return self._buildable_types
    
    def _get_concrete_types(self, constraint: str) -> list[str]:
        """获取约束的具体类型列表"""
        buildable = self._get_buildable_types_set()
        
        # 如果是 BuildableType，直接返回
        if constraint in buildable:
            return [constraint]
        
        # 获取约束定义
        defs_map = self._get_constraint_defs()
        def_obj = defs_map.get(constraint)
        
        if not def_obj or not def_obj.rule:
            return []
        
        # 展开规则到具体类型
        types = _expand_rule_to_types(def_obj.rule, defs_map, buildable)
        return sorted(list(types))
    
    def _get_operation_def(self, full_name: str) -> Any | None:
        """获取操作定义（延迟加载）"""
        if full_name not in self._operation_defs:
            try:
                dialect_name = full_name.split('.')[0]
                dialect_info = load_dialect(dialect_name)
                for op in dialect_info.operations:
                    if op.fullName == full_name:
                        self._operation_defs[full_name] = op
                        break
                else:
                    self._operation_defs[full_name] = None
            except Exception:
                self._operation_defs[full_name] = None
        return self._operation_defs.get(full_name)
    
    def _get_fixed_type(self, constraint: str) -> str | None:
        """检查约束是否是单一类型（类似前端的 getFixedType）"""
        buildable = self._get_buildable_types_set()
        if constraint in buildable:
            return constraint
        
        concrete_types = self._get_concrete_types(constraint)
        if len(concrete_types) == 1:
            return concrete_types[0]
        
        return None
    
    def _infer_call_site_types(
        self, call_node: GraphNode, graph: FunctionGraph
    ) -> dict[str, list[str]] | None:
        """
        从调用点推断输入和输出类型
        
        优先级：
        1. inputTypes/outputTypes（有效集合，string[]）
           - 单一元素：直接使用
           - 多元素：结合连接节点推断
        2. 从连接的节点推断
        3. 端口原始约束（fallback）
        
        返回: {
            'input_types': [参数类型列表],
            'output_types': [返回值类型列表]
        }
        """
        data = call_node.data
        inputs = data.get('inputs', [])
        outputs = data.get('outputs', [])
        
        # 调试输出
        print(f"[DEBUG _infer_call_site_types] call_node.id={call_node.id}")
        print(f"  data keys: {list(data.keys())}")
        print(f"  inputs: {inputs}")
        print(f"  outputs: {outputs}")
        print(f"  functionId: {data.get('functionId')}")
        
        # 如果 inputs 或 outputs 为空，尝试从 FunctionDef 重建
        if not inputs or not outputs:
            function_id = data.get('functionId')
            if function_id:
                callee_def = self.all_functions.get(function_id)
                if callee_def:
                    print("  [DEBUG] Rebuilding inputs/outputs from FunctionDef")
                    # 从 FunctionDef 重建 inputs
                    if not inputs:
                        inputs = [
                            {
                                'id': f'data-in-{p["name"]}',
                                'name': p['name'],
                                'typeConstraint': p['constraint']
                            }
                            for p in callee_def.parameters
                        ]
                        print(f"  [DEBUG] Rebuilt inputs: {inputs}")
                    # 从 FunctionDef 重建 outputs
                    if not outputs:
                        outputs = [
                            {
                                'id': f'data-out-{r["name"]}',
                                'name': r['name'],
                                'typeConstraint': r['constraint']
                            }
                            for r in callee_def.returnTypes
                        ]
                        print(f"  [DEBUG] Rebuilt outputs: {outputs}")
        
        # inputTypes/outputTypes 现在是 string[]（有效集合）
        input_types_data = data.get('inputTypes', {})
        output_types_data = data.get('outputTypes', {})
        buildable = self._get_buildable_types_set()
        
        def get_type_from_effective_set(effective_set: list[str] | str | None) -> str | None:
            """从有效集合中获取具体类型"""
            if effective_set is None:
                return None
            # 兼容旧格式（string）
            if isinstance(effective_set, str):
                return effective_set if effective_set in buildable else None
            # 新格式（string[]）
            if isinstance(effective_set, list) and len(effective_set) > 0:
                # 过滤出 BuildableType
                buildable_types = [t for t in effective_set if t in buildable]
                if len(buildable_types) == 1:
                    return buildable_types[0]
                elif len(buildable_types) > 1:
                    # 多个类型，返回第一个（后续可以结合上下文优化）
                    return buildable_types[0]
            return None
        
        # 推断输入类型
        input_types: list[str] = []
        for port in inputs:
            port_name = port['name']
            inferred_type = None
            
            # 优先级1: 从 inputTypes 获取（有效集合）
            if port_name in input_types_data:
                inferred_type = get_type_from_effective_set(input_types_data[port_name])
            
            # 优先级2: 从连接的源节点推断
            if not inferred_type:
                source_type = self._infer_port_type_from_edge(
                    call_node.id, port['id'], graph, is_input=True
                )
                if source_type and source_type in buildable:
                    inferred_type = source_type
            
            # 优先级3: 使用端口原始约束（fallback）
            if not inferred_type:
                inferred_type = port.get('typeConstraint', 'AnyType')
            
            input_types.append(inferred_type)
        
        # 推断输出类型
        output_types: list[str] = []
        for port in outputs:
            port_name = port['name']
            inferred_type = None
            
            # 优先级1: 从 outputTypes 获取（有效集合）
            if port_name in output_types_data:
                inferred_type = get_type_from_effective_set(output_types_data[port_name])
            
            # 优先级2: 从连接的后续节点推断
            if not inferred_type:
                target_type = self._infer_port_type_from_edge(
                    call_node.id, port['id'], graph, is_input=False
                )
                if target_type and target_type in buildable:
                    inferred_type = target_type
            
            # 优先级3: 使用端口原始约束（fallback）
            if not inferred_type:
                inferred_type = port.get('typeConstraint', 'AnyType')
            
            output_types.append(inferred_type)
        
        if len(input_types) != len(inputs) or len(output_types) != len(outputs):
            return None
        
        return {
            'input_types': input_types,
            'output_types': output_types
        }
    
    def _infer_port_type_from_edge(
        self, node_id: str, port_id: str, graph: FunctionGraph, is_input: bool
    ) -> str | None:
        """从连接的边推断端口类型
        
        注意：outputTypes/inputTypes 现在是 string[]（有效集合）
        """
        buildable = self._get_buildable_types_set()
        
        def get_type_from_effective_set(effective_set: list[str] | str | None) -> str | None:
            """从有效集合中获取具体类型"""
            if effective_set is None:
                return None
            # 兼容旧格式（string）
            if isinstance(effective_set, str):
                return effective_set if effective_set in buildable else None
            # 新格式（string[]）
            if isinstance(effective_set, list) and len(effective_set) > 0:
                # 过滤出 BuildableType
                buildable_types = [t for t in effective_set if t in buildable]
                if len(buildable_types) >= 1:
                    return buildable_types[0]
            return None
        
        # 查找连接到该端口的边
        if is_input:
            # 输入端口：查找源节点
            edges = [e for e in graph.edges if e.target == node_id and e.targetHandle == port_id]
            if edges:
                edge = edges[0]
                # 从源节点的 outputTypes 获取
                source_node = self._find_node_by_id(graph, edge.source)
                if source_node:
                    output_types = source_node.data.get('outputTypes', {})
                    # 从 sourceHandle 提取端口名
                    port_name = edge.sourceHandle.replace('data-out-', '')
                    type_from_output = output_types.get(port_name)
                    
                    # 从有效集合获取类型
                    inferred = get_type_from_effective_set(type_from_output)
                    if inferred:
                        return inferred
        else:
            # 输出端口：查找目标节点
            edges = [e for e in graph.edges if e.source == node_id and e.sourceHandle == port_id]
            if edges:
                edge = edges[0]
                # 从目标节点的 inputTypes 获取
                target_node = self._find_node_by_id(graph, edge.target)
                if target_node:
                    input_types = target_node.data.get('inputTypes', {})
                    # 从 targetHandle 提取端口名
                    port_name = edge.targetHandle.replace('data-in-', '')
                    type_from_input = input_types.get(port_name)
                    
                    # 从有效集合获取类型
                    inferred = get_type_from_effective_set(type_from_input)
                    if inferred:
                        return inferred
        
        return None
    
    def _find_node_by_id(self, graph: FunctionGraph, node_id: str) -> GraphNode | None:
        """根据 ID 查找节点"""
        for node in graph.nodes:
            if node.id == node_id:
                return node
        return None
    def _propagate_types_in_function_graph(
        self,
        graph: FunctionGraph,
        entry_node: GraphNode | None,
        input_types: list[str],
        output_types: list[str],
        func_def: FunctionDef
    ) -> FunctionGraph:
        """
        在函数图内部传播类型（返回副本）
        
        方案1：在前端结果基础上，加入外部调用的源，继续传播收窄
        方案2：如果方案1失败（交集为空），完全重新传播
        """
        import copy
        
        # 创建图的副本（深拷贝节点）
        graph_copy = FunctionGraph(
            nodes=[GraphNode(
                id=node.id,
                type=node.type,
                data=copy.deepcopy(node.data)
            ) for node in graph.nodes],
            edges=graph.edges  # 边不需要深拷贝
        )
        
        if not entry_node:
            return graph_copy
        
        # 在副本中找到 entry 和 return 节点
        entry_node_copy = self._find_node_by_type(graph_copy, 'function-entry')
        return_node_copy = self._find_node_by_type(graph_copy, 'function-return')
        
        # 重建 Entry 节点的 outputs（如果为空）
        if entry_node_copy:
            entry_data = entry_node_copy.data
            entry_outputs = entry_data.get('outputs', [])
            if not entry_outputs:
                entry_outputs = [
                    {'id': f'data-out-{p["name"]}', 'name': p['name']}
                    for p in func_def.parameters
                ]
                entry_data['outputs'] = entry_outputs
            
            # 为每个输出端口设置具体类型
            for i, port in enumerate(entry_outputs):
                if i < len(input_types):
                    port_name = port['name']
                    if 'outputTypes' not in entry_data:
                        entry_data['outputTypes'] = {}
                    entry_data['outputTypes'][port_name] = input_types[i]
        
        # 重建 Return 节点的 inputs（如果为空）
        if return_node_copy:
            return_data = return_node_copy.data
            return_inputs = return_data.get('inputs', [])
            if not return_inputs:
                return_inputs = [
                    {'id': f'data-in-{r["name"]}', 'name': r['name']}
                    for r in func_def.returnTypes
                ]
                return_data['inputs'] = return_inputs
        
        # 方案1：在前端结果基础上继续传播
        propagation_graph = self._build_propagation_graph(graph_copy, func_def)
        type_sources = self._extract_type_sources(graph_copy.nodes, func_def, use_frontend_results=True)
        
        if type_sources:
            print(f"[PROP] func={func_def.name} mode=frontend+external sources={len(type_sources)}")
            propagated_types = self._propagate_types_bfs(propagation_graph, type_sources, graph_copy)
            
            # 计算约束收窄并更新节点
            narrowed = self._compute_narrowed_constraints(
                propagation_graph, propagated_types, type_sources, graph_copy
            )
            self._apply_narrowed_constraints(narrowed, graph_copy)
            if narrowed:
                print(f"[PROP] narrowed {len(narrowed)} ports")
            
            # 检查是否有端口无法解析为 BuildableType
            buildable = self._get_buildable_types_set()
            needs_fallback = False
            
            for node in graph_copy.nodes:
                if node.type == 'operation':
                    output_types_dict = node.data.get('outputTypes', {})
                    for port_name, type_name in output_types_dict.items():
                        if type_name not in buildable:
                            concrete_types = self._get_concrete_types(type_name)
                            if not concrete_types or len(concrete_types) > 1:
                                needs_fallback = True
                                break
                    if needs_fallback:
                        break
            
            if needs_fallback:
                # 方案2：完全重新传播
                print(f"[PROP] func={func_def.name} fallback to full propagation")
                type_sources = self._extract_type_sources(graph_copy.nodes, func_def, use_frontend_results=False)
                if type_sources:
                    propagated_types = self._propagate_types_bfs(propagation_graph, type_sources, graph_copy)
                    narrowed = self._compute_narrowed_constraints(
                        propagation_graph, propagated_types, type_sources, graph_copy
                    )
                    self._apply_narrowed_constraints(narrowed, graph_copy)
                    if narrowed:
                        print(f"[PROP] narrowed {len(narrowed)} ports")
        else:
            # 如果没有类型源，尝试方案2
            print(f"[PROP] func={func_def.name} mode=full sources=0, trying fallback")
            type_sources = self._extract_type_sources(graph_copy.nodes, func_def, use_frontend_results=False)
            if type_sources:
                print(f"[PROP] func={func_def.name} mode=full sources={len(type_sources)}")
                propagated_types = self._propagate_types_bfs(propagation_graph, type_sources, graph_copy)
                narrowed = self._compute_narrowed_constraints(
                    propagation_graph, propagated_types, type_sources, graph_copy
                )
                self._apply_narrowed_constraints(narrowed, graph_copy)
                if narrowed:
                    print(f"[PROP] narrowed {len(narrowed)} ports")
        
        return graph_copy
    
    def _build_propagation_graph(
        self, graph: FunctionGraph, func_def: FunctionDef
    ) -> dict[str, set[str]]:
        """
        构建传播图（包括 trait 边和双向边）
        
        传播图描述类型如何从一个端口流向另一个端口：
        1. 操作节点内传播：由操作的 Trait 决定（如 SameOperandsAndResultType）
        2. 函数级别传播：由函数的 Traits 决定（如 SameType）
        3. 节点间传播：由连线决定（双向）
        """
        propagation_graph: dict[str, set[str]] = {}
        
        def add_edge(from_port: str, to_port: str) -> None:
            if from_port not in propagation_graph:
                propagation_graph[from_port] = set()
            propagation_graph[from_port].add(to_port)
        
        def add_bidirectional_edge(port_a: str, port_b: str) -> None:
            add_edge(port_a, port_b)
            add_edge(port_b, port_a)
        
        # 1. 从操作 Traits 构建节点内传播边
        for node in graph.nodes:
            if node.type != 'operation':
                continue
            
            full_name = node.data.get('fullName', '')
            if not full_name:
                continue
            
            op_def = self._get_operation_def(full_name)
            if not op_def:
                continue
            
            # SameOperandsAndResultType：所有数据端口类型相同
            if 'SameOperandsAndResultType' in op_def.traits:
                ports: list[str] = []
                
                # 优先从 inputTypes/outputTypes 的 keys 获取端口名
                input_types = node.data.get('inputTypes', {})
                output_types = node.data.get('outputTypes', {})
                
                # 收集所有输入端口
                if input_types:
                    for port_name in input_types.keys():
                        ports.append(f"{node.id}:data-in-{port_name}")
                else:
                    # fallback: 从操作定义重建
                    for arg in op_def.arguments:
                        if arg.kind == 'operand':
                            if arg.isVariadic:
                                variadic_counts = node.data.get('variadicCounts', {})
                                count = variadic_counts.get(arg.name, 1)
                                for i in range(count):
                                    ports.append(f"{node.id}:data-in-{arg.name}_{i}")
                            else:
                                ports.append(f"{node.id}:data-in-{arg.name}")
                
                # 收集所有输出端口
                if output_types:
                    for port_name in output_types.keys():
                        ports.append(f"{node.id}:data-out-{port_name}")
                else:
                    # fallback: 从操作定义重建
                    for result in op_def.results:
                        if result.isVariadic:
                            variadic_counts = node.data.get('variadicCounts', {})
                            count = variadic_counts.get(result.name, 1)
                            for i in range(count):
                                ports.append(f"{node.id}:data-out-{result.name}_{i}")
                        else:
                            ports.append(f"{node.id}:data-out-{result.name}")
                
                # 任意两个端口之间双向传播
                for i in range(len(ports)):
                    for j in range(i + 1, len(ports)):
                        add_bidirectional_edge(ports[i], ports[j])
        
        # 2. 从函数级别 Traits 构建传播边
        if hasattr(func_def, 'traits') and func_def.traits:
            entry_node = self._find_node_by_type(graph, 'function-entry')
            return_node = self._find_node_by_type(graph, 'function-return')
            
            for trait in func_def.traits:
                if trait.get('kind') == 'SameType':
                    ports: list[str] = []
                    
                    for port_name in trait.get('ports', []):
                        if port_name.startswith('return:'):
                            return_name = port_name[7:]
                            if return_node:
                                ports.append(f"{return_node.id}:data-in-{return_name}")
                        else:
                            if entry_node:
                                ports.append(f"{entry_node.id}:data-out-{port_name}")
                    
                    # 任意两个端口之间双向传播
                    for i in range(len(ports)):
                        for j in range(i + 1, len(ports)):
                            add_bidirectional_edge(ports[i], ports[j])
        
        # 3. 从连线构建节点间传播边（双向）
        for edge in graph.edges:
            # 跳过执行边
            if edge.sourceHandle.startswith('exec-') or edge.targetHandle.startswith('exec-'):
                continue
            
            source_port = f"{edge.source}:{edge.sourceHandle}"
            target_port = f"{edge.target}:{edge.targetHandle}"
            add_bidirectional_edge(source_port, target_port)
        
        return propagation_graph
    
    def _extract_type_sources(
        self, nodes: list[GraphNode], func_def: FunctionDef, use_frontend_results: bool = True
    ) -> list[tuple[str, str]]:
        """
        提取类型源
        
        类型源包括：
        1. 用户显式选择的类型（pinnedTypes）
        2. 单一具体类型的约束（自动解析，如 I32 → [I32]）
        3. Entry 节点的 outputTypes（外部调用传入的具体类型）
        
        注意：operation 节点的 inputTypes/outputTypes 是初始约束，不是类型源！
        """
        sources: list[tuple[str, str]] = []
        added_ports = set()
        
        def add_source(port_key: str, type_name: str) -> None:
            if port_key not in added_ports:
                added_ports.add(port_key)
                sources.append((port_key, type_name))
        
        for node in nodes:
            if node.type == 'operation':
                data = node.data
                pinned_types = data.get('pinnedTypes', {})
                full_name = data.get('fullName', '')
                
                # 1. 用户显式选择的类型
                for handle_id, type_name in pinned_types.items():
                    if type_name:
                        port_key = f"{node.id}:{handle_id}"
                        add_source(port_key, type_name)
                
                # 2. 自动解析单一元素约束（从操作定义获取原始约束）
                if full_name:
                    op_def = self._get_operation_def(full_name)
                    if op_def:
                        # 输入端口
                        for arg in op_def.arguments:
                            if arg.kind == 'operand':
                                port_key = f"{node.id}:data-in-{arg.name}"
                                if port_key not in added_ports:
                                    fixed_type = self._get_fixed_type(arg.typeConstraint)
                                    if fixed_type:
                                        add_source(port_key, fixed_type)
                        # 输出端口
                        for result in op_def.results:
                            port_key = f"{node.id}:data-out-{result.name}"
                            if port_key not in added_ports:
                                fixed_type = self._get_fixed_type(result.typeConstraint)
                                if fixed_type:
                                    add_source(port_key, fixed_type)
            
            elif node.type == 'function-entry':
                data = node.data
                pinned_types = data.get('pinnedTypes', {})
                
                # 1. 用户显式选择的类型
                for handle_id, type_name in pinned_types.items():
                    if type_name:
                        port_key = f"{node.id}:{handle_id}"
                        add_source(port_key, type_name)
                
                # 2. Entry 节点的 outputTypes（外部调用传入的具体类型）
                # 这是后端设置的，包含调用点的具体类型
                output_types = data.get('outputTypes', {})
                for port_name, type_name in output_types.items():
                    if type_name:
                        port_key = f"{node.id}:data-out-{port_name}"
                        if port_key not in added_ports:
                            add_source(port_key, type_name)
                
                # 3. 自动解析单一元素约束（从 FunctionDef 获取）
                for param in func_def.parameters:
                    port_key = f"{node.id}:data-out-{param['name']}"
                    if port_key not in added_ports:
                        fixed_type = self._get_fixed_type(param['constraint'])
                        if fixed_type:
                            add_source(port_key, fixed_type)
            
            elif node.type == 'function-return':
                data = node.data
                pinned_types = data.get('pinnedTypes', {})
                
                # 1. 用户显式选择的类型
                for handle_id, type_name in pinned_types.items():
                    if type_name:
                        port_key = f"{node.id}:{handle_id}"
                        add_source(port_key, type_name)
                
                # 2. 自动解析单一元素约束（从 FunctionDef 获取）
                for ret in func_def.returnTypes:
                    port_key = f"{node.id}:data-in-{ret['name']}"
                    if port_key not in added_ports:
                        fixed_type = self._get_fixed_type(ret['constraint'])
                        if fixed_type:
                            add_source(port_key, fixed_type)
            
            elif node.type == 'function-call':
                data = node.data
                pinned_types = data.get('pinnedTypes', {})
                
                # 只提取用户显式选择的类型（pinnedTypes）
                # inputTypes/outputTypes 是初始约束，不是类型源！
                for handle_id, type_name in pinned_types.items():
                    if type_name:
                        port_key = f"{node.id}:{handle_id}"
                        add_source(port_key, type_name)
        
        return sources
    
    def _propagate_types_bfs(
        self,
        propagation_graph: dict[str, set[str]],
        type_sources: list[tuple[str, str]],
        graph: FunctionGraph
    ) -> dict[str, str]:
        """
        BFS 传播类型（沿着传播图双向传播）
        """
        types: dict[str, str] = {}
        queue: list[str] = []
        node_map = {node.id: node for node in graph.nodes}
        
        # 初始化：将所有源加入队列
        for port_key, type_name in type_sources:
            types[port_key] = type_name
            queue.append(port_key)
        
        # BFS 传播
        while queue:
            current_port = queue.pop(0)
            current_type = types.get(current_port)
            if not current_type:
                continue
            
            # 获取所有可以传播到的邻居
            neighbors = propagation_graph.get(current_port, set())
            for neighbor in neighbors:
                # 如果邻居还没有类型，传播过去
                if neighbor not in types:
                    types[neighbor] = current_type
                    queue.append(neighbor)
        
        # 更新节点的 inputTypes/outputTypes
        for port_key, type_name in types.items():
            node_id, handle_id = port_key.split(':', 1)
            node = node_map.get(node_id)
            if not node:
                continue
            
            data = node.data.copy()
            
            # 判断是输入还是输出端口
            if handle_id.startswith('data-in-'):
                port_name = handle_id.replace('data-in-', '')
                if 'inputTypes' not in data:
                    data['inputTypes'] = {}
                data['inputTypes'][port_name] = type_name
            elif handle_id.startswith('data-out-'):
                port_name = handle_id.replace('data-out-', '')
                if 'outputTypes' not in data:
                    data['outputTypes'] = {}
                data['outputTypes'][port_name] = type_name
            else:
                # 可能是通过 handle_id 直接指定的（如 pinnedTypes）
                # 尝试从 inputs/outputs 中找到对应的端口
                inputs = data.get('inputs', [])
                outputs = data.get('outputs', [])
                
                for port in inputs:
                    if port.get('id') == handle_id:
                        port_name = port.get('name', '')
                        if port_name:
                            if 'inputTypes' not in data:
                                data['inputTypes'] = {}
                            data['inputTypes'][port_name] = type_name
                        break
                
                for port in outputs:
                    if port.get('id') == handle_id:
                        port_name = port.get('name', '')
                        if port_name:
                            if 'outputTypes' not in data:
                                data['outputTypes'] = {}
                            data['outputTypes'][port_name] = type_name
                        break
            
            node.data = data
        
        return types
    
    def _apply_narrowed_constraints(
        self,
        narrowed: dict[str, str],
        graph: FunctionGraph
    ) -> None:
        """应用收窄后的约束到节点"""
        node_map = {node.id: node for node in graph.nodes}
        
        for port_key, narrowed_constraint in narrowed.items():
            node_id, handle_id = port_key.split(':', 1)
            node = node_map.get(node_id)
            if not node:
                continue
            
            data = node.data.copy()
            
            if handle_id.startswith('data-in-'):
                port_name = handle_id.replace('data-in-', '')
                if 'inputTypes' not in data:
                    data['inputTypes'] = {}
                data['inputTypes'][port_name] = narrowed_constraint
            elif handle_id.startswith('data-out-'):
                port_name = handle_id.replace('data-out-', '')
                if 'outputTypes' not in data:
                    data['outputTypes'] = {}
                data['outputTypes'][port_name] = narrowed_constraint
            
            node.data = data
    
    def _compute_narrowed_constraints(
        self,
        propagation_graph: dict[str, set[str]],
        propagated_types: dict[str, str],
        type_sources: list[tuple[str, str]],
        graph: FunctionGraph
    ) -> dict[str, str]:
        """
        计算约束收窄（类似前端的 computeNarrowedConstraints）
        
        使用 outputTypes/inputTypes 作为当前约束，与邻居的有效类型求交集
        """
        narrowed = {}
        source_set = {port_key for port_key, _ in type_sources}
        
        # 辅助函数：从 start 出发，找到所有能到达的源（排除 exclude）
        def find_reachable_sources(start: str, exclude: str) -> set[str]:
            reachable = set()
            visited = set()
            queue = [start]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                if current in source_set and current != exclude:
                    reachable.add(current)
                
                neighbors = propagation_graph.get(current, set())
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            return reachable
        
        # 构建端口约束映射（使用 outputTypes/inputTypes 作为当前约束）
        # 注意：前端发送的 inputTypes/outputTypes 现在是 string[]（有效集合）
        port_constraints: dict[str, str] = {}
        
        def get_constraint_from_effective_set(effective_set: list[str] | str | None) -> str | None:
            """从有效集合中获取约束（用于传播计算）"""
            if effective_set is None:
                return None
            # 兼容旧格式（string）
            if isinstance(effective_set, str):
                return effective_set
            # 新格式（string[]）
            if isinstance(effective_set, list) and len(effective_set) > 0:
                # 如果只有一个元素，直接返回
                if len(effective_set) == 1:
                    return effective_set[0]
                # 多个元素，返回第一个（后续会通过交集计算收窄）
                return effective_set[0]
            return None
        
        for node in graph.nodes:
            data = node.data
            node_id = node.id
            
            # 收集所有端口的当前约束
            input_types = data.get('inputTypes', {})
            for port_name, type_value in input_types.items():
                port_key = f"{node_id}:data-in-{port_name}"
                constraint = get_constraint_from_effective_set(type_value)
                if constraint:
                    port_constraints[port_key] = constraint
            
            output_types = data.get('outputTypes', {})
            for port_name, type_value in output_types.items():
                port_key = f"{node_id}:data-out-{port_name}"
                constraint = get_constraint_from_effective_set(type_value)
                if constraint:
                    port_constraints[port_key] = constraint
            
            # 如果没有 outputTypes/inputTypes，尝试从操作定义获取原始约束
            if node.type == 'operation' and not input_types and not output_types:
                full_name = data.get('fullName', '')
                if full_name:
                    op_def = self._get_operation_def(full_name)
                    if op_def:
                        for arg in op_def.arguments:
                            if arg.kind == 'operand':
                                port_key = f"{node_id}:data-in-{arg.name}"
                                if port_key not in port_constraints:
                                    port_constraints[port_key] = arg.typeConstraint
                        
                        for result in op_def.results:
                            port_key = f"{node_id}:data-out-{result.name}"
                            if port_key not in port_constraints:
                                port_constraints[port_key] = result.typeConstraint
        
        # 遍历每个端口，计算约束收窄
        for port_key, current_constraint in port_constraints.items():
            neighbors = propagation_graph.get(port_key, set())
            if not neighbors:
                continue
            
            # 获取当前约束的具体类型列表
            current_types = set(self._get_concrete_types(current_constraint))
            if not current_types:
                continue
            
            intersection = current_types.copy()
            
            # 与所有邻居的有效类型求交集
            for neighbor_key in neighbors:
                neighbor_constraint = port_constraints.get(neighbor_key)
                if not neighbor_constraint:
                    continue
                
                # 邻居能到达的其他源（排除自己）
                other_sources = find_reachable_sources(neighbor_key, port_key)
                
                # 邻居有效类型：如果能到达其他源，用传播结果；否则用当前约束
                neighbor_effective = (
                    propagated_types.get(neighbor_key) if other_sources
                    else neighbor_constraint
                )
                neighbor_types = set(self._get_concrete_types(neighbor_effective))
                
                # 求交集
                intersection = intersection & neighbor_types
            
            # 检查是否发生收窄
            if intersection and len(intersection) < len(current_types):
                # 如果交集唯一，选择该类型；否则选择第一个
                if len(intersection) == 1:
                    narrowed_type = list(intersection)[0]
                else:
                    narrowed_type = sorted(list(intersection))[0]
                
                # 检查是否是 BuildableType，如果是，直接使用；否则尝试找到对应的约束名
                buildable = self._get_buildable_types_set()
                if narrowed_type in buildable:
                    narrowed[port_key] = narrowed_type
                else:
                    # 如果交集类型不是 BuildableType，保持原约束（这种情况不应该发生）
                    pass
        
        return narrowed
    
    def _infer_operation_output_types(self, node: GraphNode) -> None:
        """
        推断操作节点的输出类型
        
        1. 如果操作有 SameOperandsAndResultType trait，输出类型与输入类型相同
        2. 否则，使用 outputTypes 中的类型约束（如果已经是 BuildableType）
        
        注意：inputTypes/outputTypes 现在是 string[]（有效集合）
        """
        data = node.data.copy()
        full_name = data.get('fullName', '')
        input_types = data.get('inputTypes', {})
        output_types = data.get('outputTypes', {})
        
        if not full_name:
            return
        
        op_def = self._get_operation_def(full_name)
        buildable = self._get_buildable_types_set()
        
        def get_type_from_effective_set(effective_set: list[str] | str | None) -> str | None:
            """从有效集合中获取具体类型"""
            if effective_set is None:
                return None
            # 兼容旧格式（string）
            if isinstance(effective_set, str):
                return effective_set if effective_set in buildable else None
            # 新格式（string[]）
            if isinstance(effective_set, list) and len(effective_set) > 0:
                # 过滤出 BuildableType
                buildable_types = [t for t in effective_set if t in buildable]
                if len(buildable_types) >= 1:
                    return buildable_types[0]
            return None
        
        # 检查操作是否有 SameOperandsAndResultType trait
        if op_def and 'SameOperandsAndResultType' in op_def.traits:
            # 从有效集合中提取具体类型
            input_concrete_types = [get_type_from_effective_set(v) for v in input_types.values()]
            input_concrete_types = [t for t in input_concrete_types if t is not None]
            
            if len(input_concrete_types) > 0:
                first_input_type = input_concrete_types[0]
                all_same = all(t == first_input_type for t in input_concrete_types)
                
                if all_same and first_input_type in buildable:
                    # 更新所有输出端口为相同类型（保持 string 格式，后端内部使用）
                    for result_name in output_types.keys():
                        output_types[result_name] = first_input_type
                    data['outputTypes'] = output_types
                    node.data = data
                    return
        
        # 如果没有 trait 或输入类型不一致，确保 outputTypes 是 BuildableType
        # 如果 outputTypes 中的类型不是 BuildableType，尝试从约束解析
        updated = False
        for result_name, type_value in output_types.items():
            concrete_type = get_type_from_effective_set(type_value)
            if not concrete_type or concrete_type not in buildable:
                # 尝试从操作定义获取约束并解析
                if op_def:
                    for result in op_def.results:
                        if result.name == result_name:
                            fixed_type = self._get_fixed_type(result.typeConstraint)
                            if fixed_type:
                                output_types[result_name] = fixed_type
                                updated = True
                            break
        
        if updated:
            data['outputTypes'] = output_types
            node.data = data
    
    def _build_node(self, node: GraphNode, graph: FunctionGraph) -> None:
        """构建单个节点 - 统一处理所有类型"""
        if node.type == 'operation':
            self._build_operation_node(node, graph)
        elif node.type == 'function-call':
            self._build_function_call_node(node, graph)
        # function-entry 和 function-return 在 _build_function 中特殊处理
    
    def _build_operation_node(self, node: GraphNode, graph: FunctionGraph) -> None:
        """构建 MLIR 操作节点
        
        注意：outputTypes 现在是 string[]（有效集合）
        """
        data = node.data
        
        # 新格式：直接使用 fullName
        op_name = data.get('fullName', '')
        
        # 收集输入
        operands = self._collect_node_inputs(node, graph)
        
        # 解析结果类型 - 从 outputTypes 获取，如果是约束则先解析为具体类型
        output_types = data.get('outputTypes', {})
        buildable = self._get_buildable_types_set()
        resolved_output_types = []
        
        def get_type_from_effective_set(effective_set: list[str] | str | None) -> str | None:
            """从有效集合中获取具体类型"""
            if effective_set is None:
                return None
            # 兼容旧格式（string）
            if isinstance(effective_set, str):
                return effective_set if effective_set in buildable else None
            # 新格式（string[]）
            if isinstance(effective_set, list) and len(effective_set) > 0:
                # 过滤出 BuildableType
                buildable_types = [t for t in effective_set if t in buildable]
                if len(buildable_types) >= 1:
                    return buildable_types[0]
            return None
        
        for port_name, type_value in output_types.items():
            # 从有效集合获取类型
            type_name = get_type_from_effective_set(type_value)
            
            if type_name and type_name in buildable:
                resolved_output_types.append(type_name)
            else:
                # 如果是约束或无法解析，尝试展开
                constraint = type_value if isinstance(type_value, str) else (type_value[0] if isinstance(type_value, list) and len(type_value) > 0 else 'AnyType')
                concrete_types = self._get_concrete_types(constraint)
                if not concrete_types:
                    raise ValueError(f"Cannot resolve constraint '{constraint}' for {op_name}.{port_name}")
                if len(concrete_types) > 1:
                    # 多个类型，选择第一个（后续可以改进）
                    print(f"[BUILD] {op_name}.{port_name}: {constraint} -> {concrete_types[0]} (from {len(concrete_types)} options)")
                resolved_output_types.append(concrete_types[0])
        
        result_types = self._parse_types(resolved_output_types)
        
        # 解析属性（使用解析后的具体类型）
        resolved_output_types_dict = {
            port_name: resolved_type
            for port_name, resolved_type in zip(output_types.keys(), resolved_output_types)
        }
        attributes = convert_attributes(data.get('attributes', {}), resolved_output_types_dict)
        
        # 创建操作
        op = ir.Operation.create(
            op_name,
            results=result_types,
            operands=operands,
            attributes=attributes,
        )
        
        # 注册结果到 SSA 映射 - 使用 outputTypes 的键作为结果名称
        for i, (result, result_name) in enumerate(zip(op.results, output_types.keys())):
            handle = f"data-out-{result_name}"
            self.ssa_map[f"{node.id}:{handle}"] = result
    
    def _build_function_call_node(self, node: GraphNode, graph: FunctionGraph) -> None:
        """构建函数调用节点
        
        两阶段构建：被调用函数已在阶段1收集、阶段2构建完成，直接查找即可。
        """
        data = node.data
        callee_id = data.get('functionId', '')
        
        # 从调用点推断类型
        call_types = self._infer_call_site_types(node, graph)
        if not call_types:
            raise ValueError(f"Cannot infer call site types for '{callee_id}'")
        
        input_types = call_types['input_types']
        output_types = call_types['output_types']
        type_key = (callee_id, tuple(input_types), tuple(output_types))
        
        # 获取已构建的函数（两阶段构建保证已存在）
        callee_func = self.built_functions.get(type_key)
        if not callee_func:
            raise ValueError(f"Function '{callee_id}' not found for types {input_types}/{output_types}")
        
        # 收集输入参数
        arguments = self._collect_node_inputs(node, graph)
        
        # 创建 func.call
        call_op = func.CallOp(callee_func, arguments)
        
        # 注册结果到 SSA 映射（使用 outputTypes 的 keys）
        output_types_dict = data.get('outputTypes', {})
        for i, (result, port_name) in enumerate(zip(call_op.results, output_types_dict.keys())):
            handle = f"data-out-{port_name}"
            self.ssa_map[f"{node.id}:{handle}"] = result
    
    def _escape_type_name(self, type_name: str) -> str:
        """转义类型名中的非法字符
        
        规则：
        - 单下划线 _ 用于替换非法字符
        - 当前 BuildableType 都是安全的（I32, F32等），直接返回
        - 未来复杂类型（tensor<...>等）需要编码时再扩展
        """
        # 当前 BuildableType 都是安全的，直接返回
        # 未来扩展：对 <, >, ,, x 等字符进行编码
        return type_name
    
    def _generate_specialized_name(
        self, base_name: str, input_types: list[str], output_types: list[str]
    ) -> str:
        """生成特化函数名
        
        格式：{base}___{inputs}___{outputs}
        - ___ 三下划线：大分割（基名/输入/输出）
        - __ 两下划线：小分割（同一段内多个类型）
        - _ 单下划线：转义非法字符（当前未使用）
        """
        # 转义每个类型名
        escaped_inputs = [self._escape_type_name(t) for t in input_types]
        escaped_outputs = [self._escape_type_name(t) for t in output_types]
        
        # 用 __ 连接同一段内的多个类型
        inputs_segment = "__".join(escaped_inputs)
        outputs_segment = "__".join(escaped_outputs)
        
        # 用 ___ 连接三个大段
        return f"{base_name}___{inputs_segment}___{outputs_segment}"
    
    def _collect_node_inputs(self, node: GraphNode, graph: FunctionGraph, func_def: FunctionDef | None = None) -> list[ir.Value]:
        input_edges = [e for e in graph.edges if e.target == node.id]
        data_edges = [e for e in input_edges if not e.targetHandle.startswith('exec-')]
        
        if node.type == 'operation':
            return self._collect_operation_inputs(node, data_edges)
        elif node.type == 'function-call':
            return self._collect_function_call_inputs(node, data_edges)
        elif node.type == 'function-return':
            return self._collect_function_return_inputs(node, data_edges, func_def)
        
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
            handle = f"data-in-{input_name}"
            edge = edge_map.get(handle)
            if edge:
                key = f"{edge.source}:{edge.sourceHandle}"
                if key in self.ssa_map:
                    values.append(self.ssa_map[key])
        
        return values
    
    def _collect_function_call_inputs(
        self, node: GraphNode, edges: list[GraphEdge]
    ) -> list[ir.Value]:
        """收集函数调用节点的输入
        
        使用 inputTypes 的键顺序来确定操作数顺序。
        """
        input_types = node.data.get('inputTypes', {})
        
        # 构建 handle -> edge 映射
        edge_map = {e.targetHandle: e for e in edges}
        
        values = []
        for port_name in input_types.keys():
            handle = f"data-in-{port_name}"
            edge = edge_map.get(handle)
            if edge:
                key = f"{edge.source}:{edge.sourceHandle}"
                if key in self.ssa_map:
                    values.append(self.ssa_map[key])
        
        return values
    
    def _collect_function_return_inputs(
        self, node: GraphNode, edges: list[GraphEdge], func_def: FunctionDef | None = None
    ) -> list[ir.Value]:
        """收集函数返回节点的输入"""
        # 优先从节点数据读取（运行时已重建），否则从 FunctionDef 读取
        inputs = node.data.get('inputs', [])
        if not inputs and func_def:
            # 从 FunctionDef.returnTypes 重建端口列表
            inputs = [
                {'id': f'data-in-{r.get("name", f"result_{i}")}', 'name': r.get('name', f'result_{i}')}
                for i, r in enumerate(func_def.returnTypes)
            ]
        
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
