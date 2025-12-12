/**
 * 类型传播器
 * 
 * 从用户选择的类型源，沿着 Trait 和连线传播到其他端口。
 * 支持操作节点 Traits、函数级 Traits、跨函数边界传播。
 */

import type { Node, Edge } from '@xyflow/react';
import type { PropagationGraph, PropagationResult, VariableId, TypeSource } from './types';
import { makeVariableId } from './types';
import type { BlueprintNodeData, FunctionEntryData, FunctionReturnData, FunctionCallData, FunctionDef } from '../../types';
import { hasSameOperandsAndResultTypeTrait, analyzeConstraint } from '../typeSystem';
import { PortRef, dataIn, dataOut } from '../port';

/**
 * 获取约束的固定类型（如果是固定类型）
 * 只有 'fixed' 类型才自动成为传播源
 */
function getFixedType(constraint: string): string | null {
  const analysis = analyzeConstraint(constraint);
  // 'fixed' 和 'single' 都应该自动解析
  if (analysis.kind === 'fixed' || analysis.kind === 'single') {
    return analysis.resolvedType;
  }
  return null;
}

/**
 * 构建传播图
 * 
 * 传播图描述类型如何从一个端口流向另一个端口：
 * 1. 操作节点内传播：由操作的 Trait 决定（如 SameOperandsAndResultType）
 * 2. 函数级别传播：由函数的 Traits 决定（如 SameType）
 * 3. 节点间传播：由连线决定（双向）
 * 
 * @param nodes - 当前函数图的节点
 * @param edges - 当前函数图的边
 * @param currentFunction - 当前函数定义（用于获取函数级别 Traits）
 */
export function buildPropagationGraph(
  nodes: Node[],
  edges: Edge[],
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
 * 从类型源传播类型（BFS）
 * 
 * @param graph - 传播图
 * @param sources - 类型源（用户选择的类型）
 * @returns 传播结果
 */
export function propagateTypes(
  graph: PropagationGraph,
  sources: TypeSource[]
): PropagationResult {
  const types = new Map<VariableId, string>();
  const sourceMap = new Map<VariableId, VariableId | null>();
  const narrowedConstraints = new Map<VariableId, string>();
  const queue: VariableId[] = [];

  // 初始化：将所有源加入队列
  for (const source of sources) {
    const varId = source.portRef.key;
    types.set(varId, source.type);
    sourceMap.set(varId, null);  // 源的来源是 null（用户选择）
    queue.push(varId);
  }

  // BFS 传播
  while (queue.length > 0) {
    const varId = queue.shift()!;
    const type = types.get(varId)!;

    // 获取所有可以传播到的邻居
    const neighbors = graph.get(varId);
    if (!neighbors) continue;

    for (const neighbor of neighbors) {
      // 如果邻居还没有类型，传播过去
      if (!types.has(neighbor)) {
        types.set(neighbor, type);
        sourceMap.set(neighbor, varId);  // 记录传播来源
        queue.push(neighbor);
      }
      // 如果邻居已有类型，检查是否冲突（可选：记录冲突）
    }
  }

  return { types, sources: sourceMap, narrowedConstraints };
}

/**
 * 从节点数据中提取类型源
 * 
 * 类型源包括：
 * 1. 用户显式选择的类型（pinnedTypes, concreteType）
 * 2. 单一具体类型的约束（自动解析，如 BoolLike → I1）
 * 
 * 统一处理所有节点类型：
 * - operation: 从 pinnedTypes 提取 + 自动解析单一类型约束
 * - function-entry: 从 outputs[].concreteType 提取
 * - function-return: 从 inputs[].concreteType 提取
 * - function-call: 从 inputs/outputs[].concreteType 提取
 */
export function extractTypeSources(nodes: Node[]): TypeSource[] {
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

        // 2. 自动解析约束固定类型（输入端口）
        for (const arg of operation.arguments) {
          if (arg.kind !== 'operand') continue;

          const fixedType = getFixedType(arg.typeConstraint);

          if (arg.isVariadic) {
            // Variadic 端口：为每个实例添加源
            const count = variadicCounts[arg.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataIn(node.id, `${arg.name}_${i}`);
              if (pinnedTypes[portRef.handleId]) continue;
              if (fixedType) {
                addSource(portRef, fixedType);
              }
            }
          } else {
            const portRef = dataIn(node.id, arg.name);
            if (pinnedTypes[portRef.handleId]) continue;
            if (fixedType) {
              addSource(portRef, fixedType);
            }
          }
        }

        // 3. 自动解析约束固定类型（输出端口）
        for (const result of operation.results) {
          const fixedType = getFixedType(result.typeConstraint);

          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              if (pinnedTypes[portRef.handleId]) continue;
              if (fixedType) {
                addSource(portRef, fixedType);
              }
            }
          } else {
            const portRef = dataOut(node.id, result.name);
            if (pinnedTypes[portRef.handleId]) continue;
            if (fixedType) {
              addSource(portRef, fixedType);
            }
          }
        }
        break;
      }

      case 'function-entry': {
        const data = node.data as FunctionEntryData;
        const pinnedTypes = data.pinnedTypes || {};

        // 1. 用户选择的类型（pinnedTypes）优先
        for (const [handleId, type] of Object.entries(pinnedTypes)) {
          if (type) {
            const portRef = PortRef.fromHandle(node.id, handleId);
            if (portRef) {
              addSource(portRef, type);
            }
          }
        }

        // 2. 端口约束的固定类型
        for (const port of data.outputs) {
          const portRef = dataOut(node.id, port.name);
          if (!pinnedTypes[portRef.handleId]) {
            const fixedType = getFixedType(port.typeConstraint);
            if (fixedType) {
              addSource(portRef, fixedType);
            }
          }
        }
        break;
      }

      case 'function-return': {
        const data = node.data as FunctionReturnData;
        const pinnedTypes = data.pinnedTypes || {};

        // 1. 用户选择的类型（pinnedTypes）优先
        for (const [handleId, type] of Object.entries(pinnedTypes)) {
          if (type) {
            const portRef = PortRef.fromHandle(node.id, handleId);
            if (portRef) {
              addSource(portRef, type);
            }
          }
        }

        // 2. 端口约束的固定类型
        for (const port of data.inputs) {
          const portRef = dataIn(node.id, port.name);
          if (!pinnedTypes[portRef.handleId]) {
            const fixedType = getFixedType(port.typeConstraint);
            if (fixedType) {
              addSource(portRef, fixedType);
            }
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

        // 2. 只从固定类型约束提取源（不从 concreteType，因为它是传播派生的）
        for (const port of data.inputs) {
          const portRef = dataIn(node.id, port.name);
          if (pinnedTypes[portRef.handleId]) continue;
          const fixedType = getFixedType(port.typeConstraint);
          if (fixedType) {
            addSource(portRef, fixedType);
          }
        }

        for (const port of data.outputs) {
          const portRef = dataOut(node.id, port.name);
          if (pinnedTypes[portRef.handleId]) continue;
          const fixedType = getFixedType(port.typeConstraint);
          if (fixedType) {
            addSource(portRef, fixedType);
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
 * @param nodes - 当前函数图的节点
 * @param edges - 当前函数图的边
 * @param currentFunction - 当前函数定义（用于获取函数级别 Traits）
 */
export function computeDisplayTypes(
  nodes: Node[],
  edges: Edge[],
  currentFunction?: FunctionDef
): Map<VariableId, string> {
  // 1. 构建传播图（包含函数级别 Traits）
  const graph = buildPropagationGraph(nodes, edges, currentFunction);

  // 2. 提取类型源
  const sources = extractTypeSources(nodes);

  // 3. 传播类型
  const result = propagateTypes(graph, sources);

  return result.types;
}

/**
 * 计算类型传播和约束收窄
 * 
 * 这是一个高级封装函数，整合了：
 * 1. 构建传播图
 * 2. 提取类型源
 * 3. 传播类型
 * 4. 计算约束收窄
 * 
 * @param nodes - 当前函数图的节点
 * @param edges - 当前函数图的边
 * @param currentFunction - 当前函数定义
 * @param getConcreteTypes - 获取约束的具体类型列表（来自 store）
 * @param pickConstraintName - 选择约束名称（来自 store）
 */
export function computePropagationWithNarrowing(
  nodes: Node[],
  edges: Edge[],
  currentFunction: FunctionDef | undefined,
  getConcreteTypes: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null
): PropagationResult {
  // 1. 构建传播图（包含 trait 和连线）
  const graph = buildPropagationGraph(nodes, edges, currentFunction);

  // 2. 提取类型源
  const sources = extractTypeSources(nodes);

  // 3. 传播类型
  const result = propagateTypes(graph, sources);

  // 4. 计算约束收窄
  const portConstraints = extractPortConstraints(nodes);
  // 构建源集合
  const sourceSet = new Set(sources.map(s => s.portRef.key));
  const narrowed = computeNarrowedConstraints(
    graph,
    portConstraints, 
    result.types,
    sourceSet,
    getConcreteTypes, 
    pickConstraintName
  );
  result.narrowedConstraints = narrowed;

  return result;
}

/**
 * 根据传播结果更新所有节点的显示类型
 * 
 * 统一处理所有节点类型：
 * - operation: 更新 inputTypes/outputTypes
 * - function-entry: 更新 outputs[].concreteType
 * - function-return: 更新 inputs[].concreteType
 * - function-call: 更新 inputs[].concreteType 和 outputs[].concreteType
 */
export function applyPropagationResult(
  nodes: Node[],
  propagationResult: PropagationResult
): Node[] {
  const { types, narrowedConstraints } = propagationResult;
  
  // 辅助函数：获取端口的显示类型
  // 优先级：传播的具体类型 > 收窄后的约束 > 原始约束
  const getDisplayType = (portRef: PortRef, originalConstraint: string): string => {
    const propagatedType = types.get(portRef.key);
    if (propagatedType) return propagatedType;
    const narrowed = narrowedConstraints.get(portRef.key);
    if (narrowed) return narrowed;
    return originalConstraint;
  };

  return nodes.map(node => {
    switch (node.type) {
      case 'operation': {
        const nodeData = node.data as BlueprintNodeData;
        const operation = nodeData.operation;
        const variadicCounts = nodeData.variadicCounts || {};
        const newInputTypes: Record<string, string> = {};
        const newOutputTypes: Record<string, string> = {};

        // 输入端口
        for (const arg of operation.arguments) {
          if (arg.kind !== 'operand') continue;

          if (arg.isVariadic) {
            // Variadic 端口：检查所有实例，使用第一个有结果的类型
            const count = variadicCounts[arg.name] ?? 1;
            let displayType: string | undefined;
            for (let i = 0; i < count; i++) {
              const portRef = dataIn(node.id, `${arg.name}_${i}`);
              const type = types.get(portRef.key) || narrowedConstraints.get(portRef.key);
              if (type) {
                displayType = type;
                break;
              }
            }
            newInputTypes[arg.name] = displayType || arg.typeConstraint;
          } else {
            const portRef = dataIn(node.id, arg.name);
            newInputTypes[arg.name] = getDisplayType(portRef, arg.typeConstraint);
          }
        }

        // 输出端口
        for (const result of operation.results) {
          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            let displayType: string | undefined;
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              const type = types.get(portRef.key) || narrowedConstraints.get(portRef.key);
              if (type) {
                displayType = type;
                break;
              }
            }
            newOutputTypes[result.name] = displayType || result.typeConstraint;
          } else {
            const portRef = dataOut(node.id, result.name);
            newOutputTypes[result.name] = getDisplayType(portRef, result.typeConstraint);
          }
        }

        // 收集该节点的收窄约束
        const nodeNarrowedConstraints: Record<string, string> = {};
        for (const arg of operation.arguments) {
          if (arg.kind !== 'operand') continue;
          if (arg.isVariadic) {
            const count = variadicCounts[arg.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataIn(node.id, `${arg.name}_${i}`);
              const narrowed = narrowedConstraints.get(portRef.key);
              if (narrowed) nodeNarrowedConstraints[arg.name] = narrowed;
            }
          } else {
            const portRef = dataIn(node.id, arg.name);
            const narrowed = narrowedConstraints.get(portRef.key);
            if (narrowed) nodeNarrowedConstraints[arg.name] = narrowed;
          }
        }
        for (const result of operation.results) {
          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              const narrowed = narrowedConstraints.get(portRef.key);
              if (narrowed) nodeNarrowedConstraints[result.name] = narrowed;
            }
          } else {
            const portRef = dataOut(node.id, result.name);
            const narrowed = narrowedConstraints.get(portRef.key);
            if (narrowed) nodeNarrowedConstraints[result.name] = narrowed;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            inputTypes: newInputTypes,
            outputTypes: newOutputTypes,
            narrowedConstraints: nodeNarrowedConstraints,
          },
        };
      }

      case 'function-entry': {
        const nodeData = node.data as FunctionEntryData;
        const newOutputTypes: Record<string, string> = {};
        const nodeNarrowedConstraints: Record<string, string> = {};
        
        for (const port of nodeData.outputs) {
          const portRef = dataOut(node.id, port.name);
          const propagatedType = propagationResult.types.get(portRef.key);
          if (propagatedType) {
            newOutputTypes[port.name] = propagatedType;
          }
          const narrowed = narrowedConstraints.get(portRef.key);
          if (narrowed) {
            nodeNarrowedConstraints[port.name] = narrowed;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            outputTypes: newOutputTypes,
            narrowedConstraints: nodeNarrowedConstraints,
          },
        };
      }

      case 'function-return': {
        const nodeData = node.data as FunctionReturnData;
        const newInputTypes: Record<string, string> = {};
        const nodeNarrowedConstraints: Record<string, string> = {};
        
        for (const port of nodeData.inputs) {
          const portRef = dataIn(node.id, port.name);
          const propagatedType = propagationResult.types.get(portRef.key);
          if (propagatedType) {
            newInputTypes[port.name] = propagatedType;
          }
          const narrowed = narrowedConstraints.get(portRef.key);
          if (narrowed) {
            nodeNarrowedConstraints[port.name] = narrowed;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            inputTypes: newInputTypes,
            narrowedConstraints: nodeNarrowedConstraints,
          },
        };
      }

      case 'function-call': {
        const nodeData = node.data as FunctionCallData;
        const newInputTypes: Record<string, string> = {};
        const newOutputTypes: Record<string, string> = {};
        const nodeNarrowedConstraints: Record<string, string> = {};
        
        for (const port of nodeData.inputs) {
          const portRef = dataIn(node.id, port.name);
          const propagatedType = propagationResult.types.get(portRef.key);
          if (propagatedType) {
            newInputTypes[port.name] = propagatedType;
          }
          const narrowed = narrowedConstraints.get(portRef.key);
          if (narrowed) {
            nodeNarrowedConstraints[port.name] = narrowed;
          }
        }
        
        for (const port of nodeData.outputs) {
          const portRef = dataOut(node.id, port.name);
          const propagatedType = propagationResult.types.get(portRef.key);
          if (propagatedType) {
            newOutputTypes[port.name] = propagatedType;
          }
          const narrowed = narrowedConstraints.get(portRef.key);
          if (narrowed) {
            nodeNarrowedConstraints[port.name] = narrowed;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            inputTypes: newInputTypes,
            outputTypes: newOutputTypes,
            narrowedConstraints: nodeNarrowedConstraints,
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
 * @param nodes - 当前函数图的节点
 * @returns varId → 原始约束名
 */
export function extractPortConstraints(nodes: Node[]): Map<VariableId, string> {
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
        for (const port of data.outputs) {
          const portRef = dataOut(node.id, port.name);
          // Entry 节点：main 函数使用具体类型，自定义函数使用 AnyType
          // 这样自定义函数的参数可以选择任何类型
          constraints.set(portRef.key, data.isMain ? port.typeConstraint : 'AnyType');
        }
        break;
      }

      case 'function-return': {
        const data = node.data as FunctionReturnData;
        for (const port of data.inputs) {
          const portRef = dataIn(node.id, port.name);
          // Return 节点：main 函数使用具体类型（I32），自定义函数使用 AnyType
          // 这样自定义函数的返回值可以选择任何类型
          constraints.set(portRef.key, data.isMain ? port.typeConstraint : 'AnyType');
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
 * 计算约束收窄
 * 
 * 本端 options = 本端原始约束 ∩ 邻居有效类型
 * 邻居有效类型：
 * - 如果邻居能到达的源只有自己（或无源）→ 用邻居原始约束
 * - 如果邻居能到达其他源 → 用邻居传播结果
 * 
 * @param graph - 传播图（包含连线边和 trait 边）
 * @param portConstraints - 每个端口的原始约束
 * @param propagatedTypes - 传播结果（含用户选择）
 * @param sourceSet - 所有源端口的集合（用户 pinned 的端口）
 * @param getConcreteTypes - 获取约束的具体类型列表
 * @param pickConstraintName - 选择约束名称
 * @returns varId → 收窄后的约束名
 */
export function computeNarrowedConstraints(
  graph: PropagationGraph,
  portConstraints: Map<VariableId, string>,
  propagatedTypes: Map<VariableId, string>,
  sourceSet: Set<VariableId>,
  getConcreteTypes: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null
): Map<VariableId, string> {
  const narrowed = new Map<VariableId, string>();

  // 辅助函数：从 start 出发，在传播图中找到所有能到达的源（排除 exclude）
  const findReachableSources = (start: VariableId, exclude: VariableId): Set<VariableId> => {
    const reachable = new Set<VariableId>();
    const visited = new Set<VariableId>();
    const queue = [start];
    
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (visited.has(current)) continue;
      visited.add(current);
      
      // 如果是源且不是排除的，加入结果
      if (sourceSet.has(current) && current !== exclude) {
        reachable.add(current);
      }
      
      // 继续遍历邻居
      const neighbors = graph.get(current);
      if (neighbors) {
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor)) {
            queue.push(neighbor);
          }
        }
      }
    }
    return reachable;
  };

  // 遍历每个端口
  for (const [portKey, originalConstraint] of portConstraints) {
    const neighbors = graph.get(portKey);
    if (!neighbors || neighbors.size === 0) continue;

    const originalTypes = getConcreteTypes(originalConstraint);
    let intersection = originalTypes;

    // 与所有邻居的有效类型求交集
    for (const neighborKey of neighbors) {
      const neighborConstraint = portConstraints.get(neighborKey);
      if (!neighborConstraint) continue;

      // 邻居能到达的其他源（排除自己）
      const otherSources = findReachableSources(neighborKey, portKey);
      
      // 邻居有效类型：如果能到达其他源，用传播结果；否则用原始约束
      const neighborEffective = otherSources.size > 0
        ? (propagatedTypes.get(neighborKey) || neighborConstraint)
        : neighborConstraint;
      const neighborTypes = getConcreteTypes(neighborEffective);

      intersection = intersection.filter(t => neighborTypes.includes(t));
    }

    // 检查是否发生收窄
    if (intersection.length > 0 && intersection.length < originalTypes.length) {
      const constraintName = pickConstraintName(intersection, null, null);
      if (constraintName) {
        narrowed.set(portKey, constraintName);
      }
    }
  }

  return narrowed;
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
 * 
 * @param portKey - 要计算可选集的端口 key
 * @param nodes - 当前函数图的节点
 * @param edges - 当前函数图的边
 * @param currentFunction - 当前函数定义
 * @param getConcreteTypes - 获取约束的具体类型列表
 * @returns 可选的具体类型列表
 */
export function computeOptionsExcludingSelf(
  portKey: VariableId,
  nodes: Node[],
  edges: Edge[],
  currentFunction: FunctionDef | undefined,
  getConcreteTypes: (constraint: string) => string[]
): string[] {
  // 1. 提取端口原始约束
  const portConstraints = extractPortConstraints(nodes);
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
  const result = propagateTypes(graph, sourcesWithoutSelf);
  
  // 5. 计算可选集 = 自己原始约束的具体类型
  let options = getConcreteTypes(myConstraint);
  
  // 如果自己的约束无法展开为具体类型，返回空（无法选择）
  if (options.length === 0) {
    return [];
  }
  
  // 6. 与邻居有效类型求交集
  // 注意：空集不参与交集运算（约束无法展开时跳过）
  const neighbors = graph.get(portKey);
  if (neighbors) {
    for (const neighborKey of neighbors) {
      // 邻居有效类型：传播结果 > 原始约束
      const neighborType = result.types.get(neighborKey) || portConstraints.get(neighborKey);
      if (neighborType) {
        const neighborTypes = getConcreteTypes(neighborType);
        // 空集不参与交集运算
        if (neighborTypes.length > 0) {
          options = options.filter(t => neighborTypes.includes(t));
        }
      }
    }
  }
  
  return options;
}
