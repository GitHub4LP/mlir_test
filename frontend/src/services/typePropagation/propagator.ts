/**
 * 类型传播器
 * 
 * 从用户选择的类型源，沿着 Trait 和连线传播到其他端口。
 * 支持操作节点 Traits、函数级 Traits、跨函数边界传播。
 * 
 * 核心概念：
 * - 类型约束 = 类型集合：每个约束名对应一个具体类型的集合
 * - 传播 = 求交集：多个约束相遇时，计算它们的交集
 * - pinnedTypes = 用户意图：持久化存储，作为传播源
 * - inputTypes/outputTypes = 有效集合：传播结果，存储具体类型数组
 * 
 * 设计原则：
 * - 使用框架无关的 EditorNode/EditorEdge 类型
 * - 不依赖任何渲染框架（React Flow、Vue Flow 等）
 * 
 * 模块拆分：
 * - graph.ts: 传播图构建
 * - algorithm.ts: BFS 传播算法
 * - extractor.ts: 数据提取（类型源、端口约束）
 * - applicator.ts: 结果应用
 * - propagator.ts: 高级封装（本文件）
 */

import type { EditorNode, EditorEdge } from '../../editor/types';
import type { PropagationResult, VariableId } from './types';
import type { BlueprintNodeData, FunctionDef, FunctionCallData, PortState } from '../../types';
import { PortRef } from '../port';

// 从拆分模块导入（本文件使用）
import { buildPropagationGraph } from './graph';
import { propagateTypes } from './algorithm';
import { extractTypeSources, extractPortConstraints, computeOptionsExcludingSelf } from './extractor';

/**
 * 方言过滤配置
 * 
 * 用于在计算 options 时按方言过滤约束名
 */
export interface DialectFilterConfig {
  /** 获取函数的可达方言集（递归计算） */
  getReachableDialects: (functionId: string) => string[];
  /** 按方言过滤约束名 */
  filterConstraintsByDialects: (constraints: string[], dialects: string[]) => string[];
}

/**
 * 获取端口允许的方言列表
 * 
 * 根据节点类型确定端口可以使用哪些方言的约束：
 * - Operation: 只允许该操作所属方言
 * - Entry/Return: 允许当前函数的可达方言集
 * - Call: 允许被调用函数的可达方言集
 * 
 * @param varId - 端口变量 ID
 * @param nodes - 当前函数图的节点
 * @param currentFunction - 当前函数定义
 * @param getReachableDialects - 获取函数可达方言集的函数
 */
export function getAllowedDialectsForPort(
  varId: VariableId,
  nodes: EditorNode[],
  currentFunction: FunctionDef | undefined,
  getReachableDialects: (functionId: string) => string[]
): string[] {
  const portRef = PortRef.parse(varId);
  if (!portRef) return [];
  
  const node = nodes.find(n => n.id === portRef.nodeId);
  if (!node) return [];
  
  switch (node.type) {
    case 'operation': {
      const data = node.data as BlueprintNodeData;
      return data.operation?.dialect ? [data.operation.dialect] : [];
    }
    
    case 'function-entry':
    case 'function-return': {
      if (!currentFunction) return [];
      return getReachableDialects(currentFunction.id);
    }
    
    case 'function-call': {
      const data = node.data as FunctionCallData;
      return getReachableDialects(data.functionId);
    }
    
    default:
      return [];
  }
}

// 重新导出，保持 API 兼容
export { buildPropagationGraph } from './graph';
export { propagateTypes } from './algorithm';
export { extractTypeSources, extractPortConstraints, computeOptionsExcludingSelf } from './extractor';
export { applyPropagationResult } from './applicator';

/**
 * 计算所有端口的显示类型
 * 
 * 优先级：
 * 1. 传播结果（如果有）
 * 2. 原始类型约束
 * 
 * @param nodes - 当前函数图的节点（EditorNode 类型）
 * @param edges - 当前函数图的边（EditorEdge 类型）
 * @param currentFunction - 当前函数定义（用于获取函数级别 Traits）
 * @param getConstraintElements - 获取约束映射到的类型集合元素
 * @param pickConstraintName - 选择约束名称
 */
export function computeDisplayTypes(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null
): Map<VariableId, string> {
  // 1. 构建传播图（包含函数级别 Traits）
  const graph = buildPropagationGraph(nodes, edges, currentFunction);

  // 2. 提取端口约束
  const portConstraints = extractPortConstraints(nodes, currentFunction);

  // 3. 提取类型源
  const sources = extractTypeSources(nodes);

  // 4. 传播类型
  const result = propagateTypes(graph, sources, portConstraints, getConstraintElements);

  // 5. 从有效集合计算显示类型
  const displayTypes = new Map<VariableId, string>();
  for (const [varId, effectiveSet] of result.effectiveSets) {
    const displayType = computeDisplayTypeFromSet(effectiveSet, null, null, pickConstraintName);
    displayTypes.set(varId, displayType);
  }

  return displayTypes;
}

/**
 * 从有效集合计算显示类型
 * 
 * 规则：
 * - 单一元素：直接显示该元素
 * - 多元素：选择一个约束名显示
 * 
 * @param effectiveSet - 有效集合（具体类型数组）
 * @param nodeDialect - 节点方言（用于选择约束名）
 * @param pinnedName - 用户 pin 的约束名
 * @param pickConstraintName - 选择约束名的函数
 */
export function computeDisplayTypeFromSet(
  effectiveSet: string[],
  nodeDialect: string | null,
  pinnedName: string | null,
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null
): string {
  if (effectiveSet.length === 0) {
    return 'AnyType';  // 空集合，显示 AnyType
  }
  
  if (effectiveSet.length === 1) {
    return effectiveSet[0];  // 单一元素，直接显示
  }
  
  // 多元素，选择约束名
  const constraintName = pickConstraintName(effectiveSet, nodeDialect, pinnedName);
  return constraintName || effectiveSet[0];  // 如果找不到约束名，显示第一个元素
}

/**
 * 计算端口的 UI 状态
 * 
 * @param effectiveSet - 有效集合（用于计算 displayType）
 * @param originalConstraint - 原始约束
 * @param pinnedType - 用户 pin 的类型
 * @param nodeDialect - 节点方言
 * @param pickConstraintName - 选择约束名的函数
 * @param findSubsetConstraints - 找出所有元素集合是有效集合子集的约束名
 * @param optionsSet - 可选集的元素集合（用于计算 options，排除自己后的有效集合）
 */
export function computePortState(
  effectiveSet: string[],
  originalConstraint: string,
  pinnedType: string | undefined,
  nodeDialect: string | null,
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null,
  findSubsetConstraints: (E: string[]) => string[],
  optionsSet: string[]
): PortState {
  // 1. 计算显示类型（基于 effectiveSet）
  const displayType = computeDisplayTypeFromSet(effectiveSet, nodeDialect, pinnedType || null, pickConstraintName);
  
  // 2. 计算可选集（基于 optionsSet，排除自己后的有效集合）
  const options = optionsSet.length > 0 ? findSubsetConstraints(optionsSet) : [];
  
  // 3. canEdit = options.length > 1
  const canEdit = options.length > 1;
  
  return {
    displayType,
    constraint: originalConstraint,
    options,
    canEdit,
  };
}

/**
 * 计算类型传播（完整流程）
 * 
 * 这是一个高级封装函数，整合了：
 * 1. 构建传播图
 * 2. 提取端口约束
 * 3. 提取类型源
 * 4. 传播类型（求交集）
 * 5. 计算端口 UI 状态
 * 6. 识别无效 pin（没有产生收窄效果的 pin）
 * 7. 按方言过滤 options（可选）
 * 
 * @param nodes - 当前函数图的节点（EditorNode 类型）
 * @param edges - 当前函数图的边（EditorEdge 类型）
 * @param currentFunction - 当前函数定义
 * @param getConstraintElements - 获取约束映射到的类型约束集合元素（来自 store）
 * @param pickConstraintName - 选择约束名称（来自 store）
 * @param findSubsetConstraints - 找出所有元素集合是有效集合子集的约束名（来自 store）
 * @param dialectFilter - 方言过滤配置（可选，用于按方言过滤 options）
 */
export function computePropagation(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null,
  findSubsetConstraints: (E: string[]) => string[],
  dialectFilter?: DialectFilterConfig
): PropagationResult & { portStates: Map<VariableId, PortState>; invalidPins: Map<string, string[]> } {
  // 1. 构建传播图（包含 trait 和连线）
  const graph = buildPropagationGraph(nodes, edges, currentFunction);

  // 2. 提取端口约束
  const portConstraints = extractPortConstraints(nodes, currentFunction);

  // 3. 提取类型源
  const sources = extractTypeSources(nodes);

  // 4. 传播类型
  const result = propagateTypes(graph, sources, portConstraints, getConstraintElements);

  // 5. 计算端口 UI 状态
  const portStates = new Map<VariableId, PortState>();
  
  // 收集所有节点的 pinnedTypes
  const allPinnedTypes = new Map<VariableId, string>();
  for (const node of nodes) {
    const data = node.data as { pinnedTypes?: Record<string, string> };
    if (data.pinnedTypes) {
      for (const [handleId, type] of Object.entries(data.pinnedTypes)) {
        const portRef = PortRef.fromHandle(node.id, handleId);
        if (portRef) {
          allPinnedTypes.set(portRef.key, type);
        }
      }
    }
  }

  // 获取节点方言
  const getNodeDialect = (varId: VariableId): string | null => {
    const portRef = PortRef.parse(varId);
    if (!portRef) return null;
    const node = nodes.find(n => n.id === portRef.nodeId);
    if (!node || node.type !== 'operation') return null;
    const data = node.data as BlueprintNodeData;
    return data.operation?.dialect || null;
  };

  // 6. 识别无效 pin（nodeId -> handleId[]）
  const invalidPins = new Map<string, string[]>();

  for (const [varId, constraint] of portConstraints) {
    const effectiveSet = result.effectiveSets.get(varId) || getConstraintElements(constraint);
    const pinnedType = allPinnedTypes.get(varId);
    const nodeDialect = getNodeDialect(varId);
    
    // 计算 options：用户可以选择的类型（排除自己后的可选集）
    const optionsSet = computeOptionsExcludingSelf(
      varId, nodes, edges, currentFunction, getConstraintElements
    );
    
    // 计算原始 options（未过滤）
    let options = optionsSet.length > 0 ? findSubsetConstraints(optionsSet) : [];
    
    // 7. 按方言过滤 options（如果提供了 dialectFilter）
    if (dialectFilter && options.length > 0) {
      const allowedDialects = getAllowedDialectsForPort(
        varId, nodes, currentFunction, dialectFilter.getReachableDialects
      );
      options = dialectFilter.filterConstraintsByDialects(options, allowedDialects);
    }
    
    const state = computePortState(
      effectiveSet,
      constraint,
      pinnedType,
      nodeDialect,
      pickConstraintName,
      // 传入已过滤的 options，而不是重新计算
      () => options,
      optionsSet
    );
    
    // 覆盖 options（因为 computePortState 内部会重新计算，我们需要用过滤后的）
    state.options = options;
    state.canEdit = options.length > 1;
    
    portStates.set(varId, state);
    
    // 检查 pin 是否有效
    if (pinnedType) {
      const pinnedElements = getConstraintElements(pinnedType);
      // 有效 pin = pinnedElements.length < optionsSet.length（产生了收窄效果）
      const isEffective = pinnedElements.length < optionsSet.length;
      
      if (!isEffective) {
        // 无效 pin，记录下来
        const portRef = PortRef.parse(varId);
        if (portRef) {
          if (!invalidPins.has(portRef.nodeId)) {
            invalidPins.set(portRef.nodeId, []);
          }
          invalidPins.get(portRef.nodeId)!.push(portRef.handleId);
        }
      }
    }
  }

  return { ...result, portStates, invalidPins };
}
