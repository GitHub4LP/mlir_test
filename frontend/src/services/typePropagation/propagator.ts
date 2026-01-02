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
 */

import type { EditorNode, EditorEdge } from '../../editor/types';
import type { PropagationGraph, PropagationResult, VariableId, TypeSource } from './types';
import { makeVariableId } from './types';
import type { BlueprintNodeData, FunctionEntryData, FunctionReturnData, FunctionCallData, FunctionDef, PortState } from '../../types';
import { hasSameOperandsAndResultTypeTrait } from '../typeSystem';
import { PortRef, dataIn, dataOut } from '../port';
import { typeConstraintStore } from '../../stores';

/**
 * 构建传播图
 * 
 * 传播图描述类型如何从一个端口流向另一个端口：
 * 1. 操作节点内传播：由操作的 Trait 决定（如 SameOperandsAndResultType）
 * 2. 函数级别传播：由函数的 Traits 决定（如 SameType）
 * 3. 节点间传播：由连线决定（双向）
 * 
 * @param nodes - 当前函数图的节点（EditorNode 类型）
 * @param edges - 当前函数图的边（EditorEdge 类型）
 * @param currentFunction - 当前函数定义（用于获取函数级别 Traits）
 */
export function buildPropagationGraph(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction?: FunctionDef
): PropagationGraph {
  const graph: PropagationGraph = new Map();

  // 辅助函数：添加边（确保双向）
  const addEdge = (from: VariableId, to: VariableId) => {
    if (!graph.has(from)) {
      graph.set(from, new Set());
    }
    graph.get(from)!.add(to);
  };

  // 辅助函数：添加双向边
  const addBidirectionalEdge = (a: VariableId, b: VariableId) => {
    addEdge(a, b);
    addEdge(b, a);
  };

  // 1. 从操作 Traits 构建节点内传播边
  for (const node of nodes) {
    if (node.type !== 'operation') continue;

    const data = node.data as BlueprintNodeData;
    const operation = data.operation;
    const variadicCounts = data.variadicCounts || {};

    // SameOperandsAndResultType：所有数据端口类型相同
    if (hasSameOperandsAndResultTypeTrait(operation)) {
      const ports: VariableId[] = [];

      // 收集所有数据端口（包括 variadic 展开的实例）
      for (const arg of operation.arguments) {
        if (arg.kind === 'operand') {
          if (arg.isVariadic) {
            // Variadic 端口：展开为多个实例
            const count = variadicCounts[arg.name] ?? 1;
            for (let i = 0; i < count; i++) {
              ports.push(makeVariableId(dataIn(node.id, `${arg.name}_${i}`)));
            }
          } else {
            ports.push(makeVariableId(dataIn(node.id, arg.name)));
          }
        }
      }
      for (const result of operation.results) {
        if (result.isVariadic) {
          const count = variadicCounts[result.name] ?? 1;
          for (let i = 0; i < count; i++) {
            ports.push(makeVariableId(dataOut(node.id, `${result.name}_${i}`)));
          }
        } else {
          ports.push(makeVariableId(dataOut(node.id, result.name)));
        }
      }

      // 任意两个端口之间双向传播
      for (let i = 0; i < ports.length; i++) {
        for (let j = i + 1; j < ports.length; j++) {
          addBidirectionalEdge(ports[i], ports[j]);
        }
      }
    }
  }

  // 2. 从函数级别 Traits 构建传播边
  if (currentFunction?.traits) {
    // 找到 Entry 和 Return 节点
    const entryNode = nodes.find(n => n.type === 'function-entry');
    const returnNode = nodes.find(n => n.type === 'function-return');

    for (const trait of currentFunction.traits) {
      if (trait.kind === 'SameType') {
        const ports: VariableId[] = [];

        for (const portName of trait.ports) {
          if (portName.startsWith('return:')) {
            // 返回值端口（FunctionReturn 的输入）
            const returnName = portName.slice(7);
            if (returnNode) {
              ports.push(makeVariableId(dataIn(returnNode.id, returnName)));
            }
          } else {
            // 参数端口（FunctionEntry 的输出）
            if (entryNode) {
              ports.push(makeVariableId(dataOut(entryNode.id, portName)));
            }
          }
        }

        // 任意两个端口之间双向传播
        for (let i = 0; i < ports.length; i++) {
          for (let j = i + 1; j < ports.length; j++) {
            addBidirectionalEdge(ports[i], ports[j]);
          }
        }
      }
    }
  }

  // 3. 从连线构建节点间传播边（双向）
  for (const edge of edges) {
    // 跳过执行边
    if (edge.sourceHandle?.startsWith('exec-') || edge.targetHandle?.startsWith('exec-')) {
      continue;
    }

    if (!edge.sourceHandle || !edge.targetHandle) continue;

    // 使用 PortRef 从 handle 创建变量 ID
    const sourceRef = PortRef.fromHandle(edge.source, edge.sourceHandle);
    const targetRef = PortRef.fromHandle(edge.target, edge.targetHandle);
    
    if (sourceRef && targetRef) {
      addBidirectionalEdge(sourceRef.key, targetRef.key);
    }
  }

  return graph;
}

/**
 * 从类型源传播类型（BFS + 交集计算）
 * 
 * 算法：
 * 1. 初始化：每个端口的有效集合 = 原始约束的元素集合
 * 2. 将所有源端口加入队列，设置其有效集合为源类型的元素集合
 * 3. BFS 传播：对于每个端口，与所有邻居的有效集合求交集
 * 4. 如果交集发生变化，将该端口重新加入队列
 * 
 * @param graph - 传播图
 * @param sources - 类型源（用户选择的类型）
 * @param portConstraints - 每个端口的原始约束
 * @param getConstraintElements - 获取约束映射到的类型集合元素
 * @returns 传播结果
 */
export function propagateTypes(
  graph: PropagationGraph,
  sources: TypeSource[],
  portConstraints: Map<VariableId, string>,
  getConstraintElements: (constraint: string) => string[]
): PropagationResult {
  const effectiveSets = new Map<VariableId, string[]>();
  const sourceMap = new Map<VariableId, VariableId | null>();
  const queue: VariableId[] = [];
  const inQueue = new Set<VariableId>();

  // 1. 初始化：每个端口的有效集合 = 原始约束的元素集合
  for (const [varId, constraint] of portConstraints) {
    effectiveSets.set(varId, getConstraintElements(constraint));
  }

  // 2. 应用类型源（用户 pin + 单一元素约束）
  for (const source of sources) {
    const varId = source.portRef.key;
    const sourceElements = getConstraintElements(source.type);
    
    // 源的有效集合 = 源类型的元素集合 ∩ 原始约束的元素集合
    const originalElements = effectiveSets.get(varId) || [];
    const intersection = sourceElements.filter(t => originalElements.includes(t));
    
    if (intersection.length > 0) {
      effectiveSets.set(varId, intersection);
    } else {
      // 如果交集为空，使用源类型的元素集合（允许用户覆盖）
      effectiveSets.set(varId, sourceElements);
    }
    
    sourceMap.set(varId, null);  // 源的来源是 null（用户选择）
  }

  // 3. 将所有有邻居的端口加入队列（不只是源）
  for (const varId of graph.keys()) {
    if (!inQueue.has(varId)) {
      queue.push(varId);
      inQueue.add(varId);
    }
  }

  // 4. BFS 传播：与邻居求交集
  while (queue.length > 0) {
    const varId = queue.shift()!;
    inQueue.delete(varId);
    
    const currentSet = effectiveSets.get(varId) || [];
    if (currentSet.length === 0) continue;

    // 获取所有邻居
    const neighbors = graph.get(varId);
    if (!neighbors) continue;

    for (const neighborId of neighbors) {
      const neighborSet = effectiveSets.get(neighborId) || [];
      if (neighborSet.length === 0) continue;

      // 计算交集
      const intersection = neighborSet.filter(t => currentSet.includes(t));
      
      // 如果交集发生变化（变小了），更新并重新入队
      if (intersection.length > 0 && intersection.length < neighborSet.length) {
        effectiveSets.set(neighborId, intersection);
        sourceMap.set(neighborId, varId);  // 记录传播来源
        
        // 如果不在队列中，加入队列
        if (!inQueue.has(neighborId)) {
          queue.push(neighborId);
          inQueue.add(neighborId);
        }
      }
    }
  }

  return { effectiveSets, sources: sourceMap };
}

/**
 * 从节点数据中提取类型源
 * 
 * 类型源包括：
 * 1. 用户显式选择的类型（pinnedTypes）
 * 2. 约束集合只有一个元素的约束（自动解析，如 I32 → [I32]，BoolLike → [I1]）
 * 
 * 统一处理所有节点类型：
 * - operation: 从 pinnedTypes 提取 + 自动解析单一元素约束
 * - function-entry: 从 pinnedTypes 提取 + 自动解析单一元素约束
 * - function-return: 从 pinnedTypes 提取 + 自动解析单一元素约束
 * - function-call: 从 pinnedTypes 提取 + 自动解析单一元素约束
 */
export function extractTypeSources(
  nodes: EditorNode[]
): TypeSource[] {
  const sources: TypeSource[] = [];
  // 用于去重：同一端口只添加一次
  const addedPorts = new Set<string>();

  const addSource = (portRef: PortRef, type: string) => {
    const key = portRef.key;
    if (!addedPorts.has(key)) {
      addedPorts.add(key);
      sources.push({ portRef, type });
    }
  };

  // 辅助函数：检查约束是否映射到只有一个元素，如果是，返回那个元素
  const getSingleElement = (constraint: string): string | null => {
    if (!constraint) return null;

    // 处理 Variadic<...> 类型：解析内部类型
    const variadicMatch = constraint.match(/^Variadic<(.+)>$/);
    if (variadicMatch) {
      return getSingleElement(variadicMatch[1]);
    }

    // AnyOf<...> 类型：合成约束，多个选项
    if (constraint.startsWith('AnyOf<')) {
      return null;
    }

    const { isLoaded, getConstraintElements } = typeConstraintStore.getState();
    
    // 如果数据还没加载，返回 null
    if (!isLoaded) {
      return null;
    }

    const elements = getConstraintElements(constraint);
    
    // 如果集合只有一个元素，返回它
    if (elements.length === 1) {
      return elements[0];
    }
    
    return null;
  };

  for (const node of nodes) {
    switch (node.type) {
      case 'operation': {
        const data = node.data as BlueprintNodeData;
        const operation = data.operation;
        const pinnedTypes = data.pinnedTypes || {};
        const variadicCounts = data.variadicCounts || {};

        // 1. 用户显式选择的类型
        for (const [handleId, type] of Object.entries(pinnedTypes)) {
          if (type) {
            const portRef = PortRef.fromHandle(node.id, handleId);
            if (portRef) {
              addSource(portRef, type);
            }
          }
        }

        // 2. 自动解析单一元素约束（输入端口）
        for (const arg of operation.arguments) {
          if (arg.kind !== 'operand') continue;

          const singleElement = getSingleElement(arg.typeConstraint);

          if (arg.isVariadic) {
            // Variadic 端口：为每个实例添加源
            const count = variadicCounts[arg.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataIn(node.id, `${arg.name}_${i}`);
              if (pinnedTypes[portRef.handleId]) continue;
              if (singleElement) {
                addSource(portRef, singleElement);
              }
            }
          } else {
            const portRef = dataIn(node.id, arg.name);
            if (pinnedTypes[portRef.handleId]) continue;
            if (singleElement) {
              addSource(portRef, singleElement);
            }
          }
        }

        // 3. 自动解析单一元素约束（输出端口）
        for (const result of operation.results) {
          const singleElement = getSingleElement(result.typeConstraint);

          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              if (pinnedTypes[portRef.handleId]) continue;
              if (singleElement) {
                addSource(portRef, singleElement);
              }
            }
          } else {
            const portRef = dataOut(node.id, result.name);
            if (pinnedTypes[portRef.handleId]) continue;
            if (singleElement) {
              addSource(portRef, singleElement);
            }
          }
        }
        break;
      }

      case 'function-entry': {
        const data = node.data as FunctionEntryData;
        const pinnedTypes = data.pinnedTypes || {};
        
        // 1. 用户显式选择的类型
        for (const [handleId, type] of Object.entries(pinnedTypes)) {
          if (type) {
            const portRef = PortRef.fromHandle(node.id, handleId);
            if (portRef) {
              addSource(portRef, type);
            }
          }
        }

        // 2. 自动解析单一元素约束
        for (const port of data.outputs) {
          const portRef = dataOut(node.id, port.name);
          if (pinnedTypes[portRef.handleId]) continue;
          const singleElement = getSingleElement(port.typeConstraint);
          if (singleElement) {
            addSource(portRef, singleElement);
          }
        }
        break;
      }

      case 'function-return': {
        const data = node.data as FunctionReturnData;
        const pinnedTypes = data.pinnedTypes || {};
        
        // 1. 用户显式选择的类型
        for (const [handleId, type] of Object.entries(pinnedTypes)) {
          if (type) {
            const portRef = PortRef.fromHandle(node.id, handleId);
            if (portRef) {
              addSource(portRef, type);
            }
          }
        }

        // 2. 自动解析单一元素约束
        for (const port of data.inputs) {
          const portRef = dataIn(node.id, port.name);
          if (pinnedTypes[portRef.handleId]) continue;
          const singleElement = getSingleElement(port.typeConstraint);
          if (singleElement) {
            addSource(portRef, singleElement);
          }
        }
        break;
      }

      case 'function-call': {
        const data = node.data as FunctionCallData;
        const pinnedTypes = data.pinnedTypes || {};

        // 1. 用户选择的类型优先
        for (const [handleId, type] of Object.entries(pinnedTypes)) {
          if (type) {
            const portRef = PortRef.fromHandle(node.id, handleId);
            if (portRef) {
              addSource(portRef, type);
            }
          }
        }

        // 2. 自动解析单一元素约束
        for (const port of data.inputs) {
          const portRef = dataIn(node.id, port.name);
          if (pinnedTypes[portRef.handleId]) continue;
          const singleElement = getSingleElement(port.typeConstraint);
          if (singleElement) {
            addSource(portRef, singleElement);
          }
        }

        for (const port of data.outputs) {
          const portRef = dataOut(node.id, port.name);
          if (pinnedTypes[portRef.handleId]) continue;
          const singleElement = getSingleElement(port.typeConstraint);
          if (singleElement) {
            addSource(portRef, singleElement);
          }
        }
        break;
      }
    }
  }

  return sources;
}

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
 */
export function computeDisplayTypes(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction?: FunctionDef
): Map<VariableId, string> {
  const { getConstraintElements, pickConstraintName } = typeConstraintStore.getState();
  
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
 * 
 * @param nodes - 当前函数图的节点（EditorNode 类型）
 * @param edges - 当前函数图的边（EditorEdge 类型）
 * @param currentFunction - 当前函数定义
 * @param getConstraintElements - 获取约束映射到的类型约束集合元素（来自 store）
 * @param pickConstraintName - 选择约束名称（来自 store）
 * @param findSubsetConstraints - 找出所有元素集合是有效集合子集的约束名（来自 store）
 */
export function computePropagation(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null,
  findSubsetConstraints: (E: string[]) => string[]
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
    
    const state = computePortState(
      effectiveSet,
      constraint,
      pinnedType,
      nodeDialect,
      pickConstraintName,
      findSubsetConstraints,
      optionsSet
    );
    
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

/**
 * 清理无效 pin
 * 
 * @param currentPinnedTypes - 当前的 pinnedTypes
 * @param invalidHandleIds - 无效的 handleId 列表
 * @returns 清理后的 pinnedTypes，如果没有变化返回原对象
 */
function cleanInvalidPins(
  currentPinnedTypes: Record<string, string> | undefined,
  invalidHandleIds: string[] | undefined
): Record<string, string> | undefined {
  if (!currentPinnedTypes || !invalidHandleIds || invalidHandleIds.length === 0) {
    return currentPinnedTypes;
  }
  
  const newPinnedTypes = { ...currentPinnedTypes };
  let changed = false;
  
  for (const handleId of invalidHandleIds) {
    if (handleId in newPinnedTypes) {
      delete newPinnedTypes[handleId];
      changed = true;
    }
  }
  
  if (!changed) {
    return currentPinnedTypes;
  }
  
  // 如果清理后为空，返回 undefined
  return Object.keys(newPinnedTypes).length > 0 ? newPinnedTypes : undefined;
}

/**
 * 根据传播结果更新所有节点的类型数据
 * 
 * 统一处理所有节点类型：
 * - operation: 更新 inputTypes/outputTypes（string[]）
 * - function-entry: 更新 outputTypes（string[]）
 * - function-return: 更新 inputTypes（string[]）
 * - function-call: 更新 inputTypes/outputTypes（string[]）
 * 
 * 同时清理无效 pin（没有产生收窄效果的 pin）
 */
export function applyPropagationResult(
  nodes: EditorNode[],
  propagationResult: PropagationResult & { portStates?: Map<VariableId, PortState>; invalidPins?: Map<string, string[]> }
): EditorNode[] {
  const { effectiveSets, portStates, invalidPins } = propagationResult;

  return nodes.map(node => {
    switch (node.type) {
      case 'operation': {
        const nodeData = node.data as BlueprintNodeData;
        const operation = nodeData.operation;
        const variadicCounts = nodeData.variadicCounts || {};
        const newInputTypes: Record<string, string[]> = {};
        const newOutputTypes: Record<string, string[]> = {};
        const newPortStates: Record<string, PortState> = {};

        // 输入端口
        for (const arg of operation.arguments) {
          if (arg.kind !== 'operand') continue;

          if (arg.isVariadic) {
            // Variadic 端口：使用第一个实例的有效集合
            const count = variadicCounts[arg.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataIn(node.id, `${arg.name}_${i}`);
              const effectiveSet = effectiveSets.get(portRef.key);
              if (effectiveSet) {
                newInputTypes[arg.name] = effectiveSet;
                break;
              }
            }
            if (!newInputTypes[arg.name]) {
              newInputTypes[arg.name] = [arg.typeConstraint];
            }
            // 收集 portStates
            for (let i = 0; i < count; i++) {
              const portRef = dataIn(node.id, `${arg.name}_${i}`);
              const state = portStates?.get(portRef.key);
              if (state) {
                newPortStates[portRef.handleId] = state;
              }
            }
          } else {
            const portRef = dataIn(node.id, arg.name);
            const effectiveSet = effectiveSets.get(portRef.key);
            newInputTypes[arg.name] = effectiveSet || [arg.typeConstraint];
            const state = portStates?.get(portRef.key);
            if (state) {
              newPortStates[portRef.handleId] = state;
            }
          }
        }

        // 输出端口
        for (const result of operation.results) {
          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              const effectiveSet = effectiveSets.get(portRef.key);
              if (effectiveSet) {
                newOutputTypes[result.name] = effectiveSet;
                break;
              }
            }
            if (!newOutputTypes[result.name]) {
              newOutputTypes[result.name] = [result.typeConstraint];
            }
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              const state = portStates?.get(portRef.key);
              if (state) {
                newPortStates[portRef.handleId] = state;
              }
            }
          } else {
            const portRef = dataOut(node.id, result.name);
            const effectiveSet = effectiveSets.get(portRef.key);
            newOutputTypes[result.name] = effectiveSet || [result.typeConstraint];
            const state = portStates?.get(portRef.key);
            if (state) {
              newPortStates[portRef.handleId] = state;
            }
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            pinnedTypes: cleanInvalidPins(nodeData.pinnedTypes, invalidPins?.get(node.id)),
            inputTypes: newInputTypes,
            outputTypes: newOutputTypes,
            portStates: newPortStates,
          },
        };
      }

      case 'function-entry': {
        const nodeData = node.data as FunctionEntryData;
        const newOutputTypes: Record<string, string[]> = {};
        const newPortStates: Record<string, PortState> = {};
        
        for (const port of nodeData.outputs) {
          const portRef = dataOut(node.id, port.name);
          const effectiveSet = effectiveSets.get(portRef.key);
          if (effectiveSet) {
            newOutputTypes[port.name] = effectiveSet;
          }
          const state = portStates?.get(portRef.key);
          if (state) {
            newPortStates[portRef.handleId] = state;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            pinnedTypes: cleanInvalidPins(nodeData.pinnedTypes, invalidPins?.get(node.id)),
            outputTypes: newOutputTypes,
            portStates: newPortStates,
          },
        };
      }

      case 'function-return': {
        const nodeData = node.data as FunctionReturnData;
        const newInputTypes: Record<string, string[]> = {};
        const newPortStates: Record<string, PortState> = {};
        
        for (const port of nodeData.inputs) {
          const portRef = dataIn(node.id, port.name);
          const effectiveSet = effectiveSets.get(portRef.key);
          if (effectiveSet) {
            newInputTypes[port.name] = effectiveSet;
          }
          const state = portStates?.get(portRef.key);
          if (state) {
            newPortStates[portRef.handleId] = state;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            pinnedTypes: cleanInvalidPins(nodeData.pinnedTypes, invalidPins?.get(node.id)),
            inputTypes: newInputTypes,
            portStates: newPortStates,
          },
        };
      }

      case 'function-call': {
        const nodeData = node.data as FunctionCallData;
        const newInputTypes: Record<string, string[]> = {};
        const newOutputTypes: Record<string, string[]> = {};
        const newPortStates: Record<string, PortState> = {};
        
        for (const port of nodeData.inputs) {
          const portRef = dataIn(node.id, port.name);
          const effectiveSet = effectiveSets.get(portRef.key);
          newInputTypes[port.name] = effectiveSet || [port.typeConstraint];
          const state = portStates?.get(portRef.key);
          if (state) {
            newPortStates[portRef.handleId] = state;
          }
        }
        
        for (const port of nodeData.outputs) {
          const portRef = dataOut(node.id, port.name);
          const effectiveSet = effectiveSets.get(portRef.key);
          newOutputTypes[port.name] = effectiveSet || [port.typeConstraint];
          const state = portStates?.get(portRef.key);
          if (state) {
            newPortStates[portRef.handleId] = state;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            pinnedTypes: cleanInvalidPins(nodeData.pinnedTypes, invalidPins?.get(node.id)),
            inputTypes: newInputTypes,
            outputTypes: newOutputTypes,
            portStates: newPortStates,
          },
        };
      }

      default:
        return node;
    }
  });
}


/**
 * 提取每个端口的原始约束
 * 
 * @param nodes - 当前函数图的节点（EditorNode 类型）
 * @param currentFunction - 当前函数定义（用于 Entry/Return 节点）
 * @returns varId → 原始约束名
 */
export function extractPortConstraints(
  nodes: EditorNode[],
  currentFunction?: FunctionDef
): Map<VariableId, string> {
  const constraints = new Map<VariableId, string>();

  for (const node of nodes) {
    switch (node.type) {
      case 'operation': {
        const data = node.data as BlueprintNodeData;
        const operation = data.operation;
        const variadicCounts = data.variadicCounts || {};

        // 输入端口
        for (const arg of operation.arguments) {
          if (arg.kind !== 'operand') continue;
          if (arg.isVariadic) {
            const count = variadicCounts[arg.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataIn(node.id, `${arg.name}_${i}`);
              constraints.set(portRef.key, arg.typeConstraint);
            }
          } else {
            const portRef = dataIn(node.id, arg.name);
            constraints.set(portRef.key, arg.typeConstraint);
          }
        }

        // 输出端口
        for (const result of operation.results) {
          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              constraints.set(portRef.key, result.typeConstraint);
            }
          } else {
            const portRef = dataOut(node.id, result.name);
            constraints.set(portRef.key, result.typeConstraint);
          }
        }
        break;
      }

      case 'function-entry': {
        const data = node.data as FunctionEntryData;
        const parameters = currentFunction?.parameters || data.outputs.map(p => ({ name: p.name, constraint: p.typeConstraint }));
        for (const param of parameters) {
          const portRef = dataOut(node.id, param.name);
          constraints.set(portRef.key, data.isMain ? param.constraint : 'AnyType');
        }
        break;
      }

      case 'function-return': {
        const data = node.data as FunctionReturnData;
        const returnTypes = currentFunction?.returnTypes || data.inputs.map(p => ({ name: p.name, constraint: p.typeConstraint }));
        for (const ret of returnTypes) {
          const portRef = dataIn(node.id, ret.name);
          constraints.set(portRef.key, data.isMain ? ret.constraint : 'AnyType');
        }
        break;
      }

      case 'function-call': {
        const data = node.data as FunctionCallData;
        for (const port of data.inputs) {
          const portRef = dataIn(node.id, port.name);
          constraints.set(portRef.key, port.typeConstraint);
        }
        for (const port of data.outputs) {
          const portRef = dataOut(node.id, port.name);
          constraints.set(portRef.key, port.typeConstraint);
        }
        break;
      }
    }
  }

  return constraints;
}

/**
 * 计算端口的可选类型（排除自己的影响）
 * 
 * 核心思想：计算端口 A 的可选集时，需要一个"假设 A 不存在"的世界。
 * 
 * 算法：
 * 1. 提取所有类型源，排除自己
 * 2. 执行传播（无自己）
 * 3. 可选集 = 自己原始约束 ∩ 邻居有效类型
 * 4. 返回所有元素集合是可选集子集的约束名
 * 
 * @param portKey - 要计算可选集的端口 key
 * @param nodes - 当前函数图的节点（EditorNode 类型）
 * @param edges - 当前函数图的边（EditorEdge 类型）
 * @param currentFunction - 当前函数定义
 * @param getConstraintElements - 获取约束映射到的类型约束集合元素
 * @returns 可选的具体类型列表
 */
export function computeOptionsExcludingSelf(
  portKey: VariableId,
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[]
): string[] {
  // 1. 提取端口原始约束（传入 currentFunction 以正确处理 Entry/Return 节点）
  const portConstraints = extractPortConstraints(nodes, currentFunction);
  const myConstraint = portConstraints.get(portKey);
  
  if (!myConstraint) {
    return [];
  }
  
  // 2. 提取所有类型源，排除自己
  const allSources = extractTypeSources(nodes);
  const sourcesWithoutSelf = allSources.filter(s => s.portRef.key !== portKey);
  
  // 3. 构建传播图
  const graph = buildPropagationGraph(nodes, edges, currentFunction);
  
  // 4. 执行传播（无自己）
  const result = propagateTypes(graph, sourcesWithoutSelf, portConstraints, getConstraintElements);
  
  // 5. 计算可选集 = 自己原始约束的集合元素
  let options = getConstraintElements(myConstraint);
  
  // 如果自己的约束无法展开，返回空（无法选择）
  if (options.length === 0) {
    return [];
  }
  
  // 6. 与邻居有效类型求交集
  // 注意：空集不参与交集运算（约束无法展开时跳过）
  const neighbors = graph.get(portKey);
  
  if (neighbors) {
    for (const neighborKey of neighbors) {
      // 邻居有效类型：传播结果（已经是交集后的有效集合）
      const neighborSet = result.effectiveSets.get(neighborKey) || [];
      // 空集不参与交集运算
      if (neighborSet.length > 0) {
        options = options.filter(t => neighborSet.includes(t));
      }
    }
  }
  
  return options;
}
