/**
 * Graph Converter Service
 * 
 * 将前端图格式转换为后端 API 格式。
 * 
 * 前端图：
 * - 节点包含完整的 OperationDef
 * - 类型存储为 JSON 格式（如 "I32"），后端负责转换为 MLIR 格式
 * - 属性存储为 MLIR 属性字符串（如 "10 : i32"）
 * - 边使用 handle 字符串标识端口
 * 
 * 后端 API：
 * - 节点只需 op_name (fullName)、result_types、attributes
 * - 边使用数字索引标识端口
 * - 后端 type_registry.py 负责将 JSON 格式类型转换为 MLIR 格式
 */

import type { GraphNode, GraphEdge, BlueprintNodeData, OperationDef, FunctionEntryData, FunctionReturnData } from '../types';
import { PortRef, PortKind } from './port';

/**
 * 后端图节点模型
 */
export interface BackendGraphNode {
  id: string;
  op_name: string;                    // fullName: "arith.addi"
  result_types: string[];             // JSON 格式类型: ["I32"]，后端转换为 MLIR 格式
  attributes: Record<string, string>; // MLIR 属性字符串
  region_graphs: BackendGraph[];      // 嵌套区域图
}

/**
 * 后端图边模型
 */
export interface BackendGraphEdge {
  source_node: string;
  source_output: number;
  target_node: string;
  target_input: number;
}

/**
 * 后端图模型
 */
export interface BackendGraph {
  nodes: BackendGraphNode[];
  edges: BackendGraphEdge[];
  block_arg_types?: string[];
}

/**
 * 从 handle 字符串解析端口索引
 * 
 * Handle 格式：
 * - "data-out-{name}" -> 输出端口
 * - "data-in-{name}" -> 输入端口
 */
function parseOutputIndex(handle: string, operation: OperationDef): number {
  const parsed = PortRef.parseHandleId(handle);
  if (!parsed || parsed.kind !== PortKind.DataOut) {
    throw new Error(`Invalid output handle: ${handle}`);
  }
  
  const name = parsed.name;
  const index = operation.results.findIndex(r => r.name === name);
  
  if (index === -1) {
    throw new Error(`Output '${name}' not found in operation '${operation.fullName}'`);
  }
  
  return index;
}

function parseInputIndex(handle: string, operation: OperationDef): number {
  const parsed = PortRef.parseHandleId(handle);
  if (!parsed || parsed.kind !== PortKind.DataIn) {
    throw new Error(`Invalid input handle: ${handle}`);
  }
  
  const name = parsed.name;
  const operands = operation.arguments.filter(a => a.kind === 'operand');
  const index = operands.findIndex(o => o.name === name);
  
  if (index === -1) {
    throw new Error(`Input '${name}' not found in operation '${operation.fullName}'`);
  }
  
  return index;
}

/**
 * 检查是否为执行边（非数据边）
 */
function isExecEdge(handle: string): boolean {
  return handle.startsWith('exec-');
}

/**
 * 将属性值转换为 MLIR 属性字符串
 * 
 * 对于 TypedAttrInterface（如 arith.constant 的 value），需要结合类型信息
 * 例如：value=10, type=i32 -> "10 : i32"
 */
function formatAttribute(
  value: unknown,
  typeConstraint: string,
  resultType: string | undefined
): string {
  // 如果值已经是字符串且包含类型信息，直接返回
  if (typeof value === 'string' && value.includes(':')) {
    return value;
  }
  
  // TypedAttrInterface 需要类型后缀
  if (typeConstraint.toLowerCase().includes('typedattrinterface') || 
      typeConstraint.toLowerCase() === 'typedattr') {
    if (resultType) {
      return `${value} : ${resultType}`;
    }
  }
  
  // 其他属性直接转为字符串
  return String(value);
}

/**
 * 转换单个操作节点
 */
function convertOperationNode(node: GraphNode): BackendGraphNode {
  const data = node.data as BlueprintNodeData;
  const operation = data.operation;
  
  // 收集结果类型（按顺序）
  const resultTypes: string[] = [];
  for (const result of operation.results) {
    const resultType = data.outputTypes?.[result.name];
    if (!resultType) {
      throw new Error(`Missing type for result '${result.name}' in node '${node.id}'`);
    }
    resultTypes.push(resultType);
  }
  
  // 第一个结果类型（用于 TypedAttrInterface 属性）
  const primaryResultType = resultTypes[0];
  
  // 收集属性并转换为 MLIR 格式
  const attributes: Record<string, string> = {};
  for (const arg of operation.arguments) {
    if (arg.kind === 'attribute') {
      const value = data.attributes[arg.name];
      if (value !== undefined && value !== null && value !== '') {
        attributes[arg.name] = formatAttribute(
          value,
          arg.typeConstraint,
          primaryResultType
        );
      }
    }
  }
  
  return {
    id: node.id,
    op_name: operation.fullName,
    result_types: resultTypes,
    attributes,
    region_graphs: [], // TODO: 支持区域
  };
}

/**
 * 转换函数入口节点
 */
function convertFunctionEntryNode(node: GraphNode): BackendGraphNode {
  const data = node.data as FunctionEntryData;
  
  // 函数参数类型（后端会自动处理 JSON 格式到 MLIR 格式的转换）
  const resultTypes = data.outputs.map(output => output.typeConstraint);
  
  return {
    id: node.id,
    op_name: 'function-entry',
    result_types: resultTypes,
    attributes: {},
    region_graphs: [],
  };
}

/**
 * 转换函数返回节点
 */
function convertFunctionReturnNode(node: GraphNode): BackendGraphNode {
  const data = node.data as FunctionReturnData;
  
  // 返回值类型（后端会自动处理 JSON 格式到 MLIR 格式的转换）
  const resultTypes = data.inputs.map(input => input.typeConstraint);
  
  return {
    id: node.id,
    op_name: 'function-return',
    result_types: resultTypes,
    attributes: {},
    region_graphs: [],
  };
}

/**
 * 解析源节点的输出索引
 */
function getSourceOutputIndex(sourceNode: GraphNode, sourceHandle: string): number {
  if (sourceNode.type === 'operation') {
    const data = sourceNode.data as BlueprintNodeData;
    return parseOutputIndex(sourceHandle, data.operation);
  } else if (sourceNode.type === 'function-entry') {
    // function-entry 的输出格式是 output.id
    const data = sourceNode.data as FunctionEntryData;
    const index = data.outputs.findIndex(o => o.id === sourceHandle);
    return index >= 0 ? index : 0;
  }
  return 0;
}

/**
 * 解析目标节点的输入索引
 */
function getTargetInputIndex(targetNode: GraphNode, targetHandle: string): number {
  if (targetNode.type === 'operation') {
    const data = targetNode.data as BlueprintNodeData;
    return parseInputIndex(targetHandle, data.operation);
  } else if (targetNode.type === 'function-return') {
    // function-return 的输入格式是 input.id
    const data = targetNode.data as FunctionReturnData;
    const index = data.inputs.findIndex(i => i.id === targetHandle);
    return index >= 0 ? index : 0;
  }
  return 0;
}

/**
 * 转换图边
 */
function convertEdge(
  edge: GraphEdge,
  nodeMap: Map<string, GraphNode>
): BackendGraphEdge | null {
  // 跳过执行边
  if (isExecEdge(edge.sourceHandle) || isExecEdge(edge.targetHandle)) {
    return null;
  }
  
  const sourceNode = nodeMap.get(edge.source);
  const targetNode = nodeMap.get(edge.target);
  
  if (!sourceNode || !targetNode) {
    console.warn(`Edge references missing node: ${edge.source} -> ${edge.target}`);
    return null;
  }
  
  // 支持的边类型：
  // - operation -> operation
  // - operation -> function-return
  // - function-entry -> operation
  const validSourceTypes = ['operation', 'function-entry'];
  const validTargetTypes = ['operation', 'function-return'];
  
  if (!validSourceTypes.includes(sourceNode.type) || !validTargetTypes.includes(targetNode.type)) {
    return null;
  }
  
  return {
    source_node: edge.source,
    source_output: getSourceOutputIndex(sourceNode, edge.sourceHandle),
    target_node: edge.target,
    target_input: getTargetInputIndex(targetNode, edge.targetHandle),
  };
}

/**
 * 将前端图转换为后端 API 格式
 * 
 * 转换所有节点类型：
 * - operation: MLIR 操作节点
 * - function-entry: 函数入口（提供参数类型）
 * - function-return: 函数返回（提供返回类型）
 */
export function convertToBackendGraph(
  nodes: GraphNode[],
  edges: GraphEdge[]
): BackendGraph {
  // 构建节点映射
  const nodeMap = new Map<string, GraphNode>();
  for (const node of nodes) {
    nodeMap.set(node.id, node);
  }
  
  // 转换所有节点
  const backendNodes: BackendGraphNode[] = [];
  for (const node of nodes) {
    if (node.type === 'operation') {
      backendNodes.push(convertOperationNode(node));
    } else if (node.type === 'function-entry') {
      backendNodes.push(convertFunctionEntryNode(node));
    } else if (node.type === 'function-return') {
      backendNodes.push(convertFunctionReturnNode(node));
    }
    // function-call 暂不支持
  }
  
  // 转换边
  const backendEdges: BackendGraphEdge[] = [];
  for (const edge of edges) {
    const converted = convertEdge(edge, nodeMap);
    if (converted) {
      backendEdges.push(converted);
    }
  }
  
  return {
    nodes: backendNodes,
    edges: backendEdges,
  };
}

/**
 * 执行图请求
 */
export interface ExecuteGraphRequest {
  graph: BackendGraph;
  func_name: string;
}

/**
 * 执行图响应
 */
export interface ExecuteGraphResponse {
  success: boolean;
  mlir_code: string;
  output: string;
  error?: string;
}

/**
 * 调用后端执行图（JIT 模式）
 */
export async function executeGraph(
  nodes: GraphNode[],
  edges: GraphEdge[],
  funcName: string = 'main'
): Promise<ExecuteGraphResponse> {
  const graph = convertToBackendGraph(nodes, edges);
  
  const response = await fetch('/api/graph/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      graph,
      func_name: funcName,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * 调用后端构建图（仅生成 MLIR，不执行）
 */
export async function buildGraph(
  nodes: GraphNode[],
  edges: GraphEdge[],
  funcName: string = 'main'
): Promise<{ success: boolean; mlir_code: string; verified: boolean; error?: string }> {
  const graph = convertToBackendGraph(nodes, edges);
  
  const response = await fetch('/api/graph/build', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      graph,
      func_name: funcName,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }
  
  return response.json();
}
