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

/**
 * 获取约束的固定类型（如果是固定类型）
 * 只有 'fixed' 类型才自动成为传播源
 */
function getFixedType(constraint: string): string | null {
  const analysis = analyzeConstraint(constraint);
  return analysis.kind === 'fixed' ? analysis.resolvedType : null;
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
              ports.push(makeVariableId(node.id, `input-${arg.name}_${i}`));
            }
          } else {
            ports.push(makeVariableId(node.id, `input-${arg.name}`));
          }
        }
      }
      for (const result of operation.results) {
        if (result.isVariadic) {
          const count = variadicCounts[result.name] ?? 1;
          for (let i = 0; i < count; i++) {
            ports.push(makeVariableId(node.id, `output-${result.name}_${i}`));
          }
        } else {
          ports.push(makeVariableId(node.id, `output-${result.name}`));
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
            // 返回值端口
            const returnName = portName.slice(7);
            if (returnNode) {
              ports.push(makeVariableId(returnNode.id, `return-${returnName}`));
            }
          } else {
            // 参数端口
            if (entryNode) {
              ports.push(makeVariableId(entryNode.id, `param-${portName}`));
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

    const sourceVar = makeVariableId(edge.source, edge.sourceHandle);
    const targetVar = makeVariableId(edge.target, edge.targetHandle);

    addBidirectionalEdge(sourceVar, targetVar);
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
  const queue: VariableId[] = [];

  // 初始化：将所有源加入队列
  for (const source of sources) {
    const varId = makeVariableId(source.nodeId, source.portId);
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

  return { types, sources: sourceMap };
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

  const addSource = (nodeId: string, portId: string, type: string) => {
    const key = `${nodeId}:${portId}`;
    if (!addedPorts.has(key)) {
      addedPorts.add(key);
      sources.push({ nodeId, portId, type });
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
        for (const [portId, type] of Object.entries(pinnedTypes)) {
          if (type) {
            addSource(node.id, portId, type);
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
              const portId = `input-${arg.name}_${i}`;
              if (pinnedTypes[portId]) continue;
              if (fixedType) {
                addSource(node.id, portId, fixedType);
              }
            }
          } else {
            const portId = `input-${arg.name}`;
            if (pinnedTypes[portId]) continue;
            if (fixedType) {
              addSource(node.id, portId, fixedType);
            }
          }
        }

        // 3. 自动解析约束固定类型（输出端口）
        for (const result of operation.results) {
          const fixedType = getFixedType(result.typeConstraint);

          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portId = `output-${result.name}_${i}`;
              if (pinnedTypes[portId]) continue;
              if (fixedType) {
                addSource(node.id, portId, fixedType);
              }
            }
          } else {
            const portId = `output-${result.name}`;
            if (pinnedTypes[portId]) continue;
            if (fixedType) {
              addSource(node.id, portId, fixedType);
            }
          }
        }
        break;
      }

      case 'function-entry': {
        const data = node.data as FunctionEntryData;
        for (const port of data.outputs) {
          // 优先使用 concreteType，否则尝试解析约束固定类型
          const type = port.concreteType || getFixedType(port.typeConstraint);
          if (type) {
            addSource(node.id, port.id, type);
          }
        }
        break;
      }

      case 'function-return': {
        const data = node.data as FunctionReturnData;
        for (const port of data.inputs) {
          const type = port.concreteType || getFixedType(port.typeConstraint);
          if (type) {
            addSource(node.id, port.id, type);
          }
        }
        break;
      }

      case 'function-call': {
        const data = node.data as FunctionCallData;
        for (const port of data.inputs) {
          const type = port.concreteType || getFixedType(port.typeConstraint);
          if (type) {
            addSource(node.id, port.id, type);
          }
        }
        for (const port of data.outputs) {
          const type = port.concreteType || getFixedType(port.typeConstraint);
          if (type) {
            addSource(node.id, port.id, type);
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
            // Variadic 端口：检查所有实例，使用第一个有传播结果的类型
            const count = variadicCounts[arg.name] ?? 1;
            let propagatedType: string | undefined;
            for (let i = 0; i < count; i++) {
              const varId = makeVariableId(node.id, `input-${arg.name}_${i}`);
              const type = propagationResult.types.get(varId);
              if (type) {
                propagatedType = type;
                break;
              }
            }
            newInputTypes[arg.name] = propagatedType || arg.typeConstraint;
          } else {
            const varId = makeVariableId(node.id, `input-${arg.name}`);
            const propagatedType = propagationResult.types.get(varId);
            newInputTypes[arg.name] = propagatedType || arg.typeConstraint;
          }
        }

        // 输出端口
        for (const result of operation.results) {
          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            let propagatedType: string | undefined;
            for (let i = 0; i < count; i++) {
              const varId = makeVariableId(node.id, `output-${result.name}_${i}`);
              const type = propagationResult.types.get(varId);
              if (type) {
                propagatedType = type;
                break;
              }
            }
            newOutputTypes[result.name] = propagatedType || result.typeConstraint;
          } else {
            const varId = makeVariableId(node.id, `output-${result.name}`);
            const propagatedType = propagationResult.types.get(varId);
            newOutputTypes[result.name] = propagatedType || result.typeConstraint;
          }
        }

        return {
          ...node,
          data: {
            ...nodeData,
            inputTypes: newInputTypes,
            outputTypes: newOutputTypes,
          },
        };
      }

      case 'function-entry': {
        const nodeData = node.data as FunctionEntryData;
        const newOutputs = nodeData.outputs.map(port => {
          const varId = makeVariableId(node.id, port.id);
          const propagatedType = propagationResult.types.get(varId);
          return {
            ...port,
            concreteType: propagatedType || port.concreteType,
          };
        });

        return {
          ...node,
          data: {
            ...nodeData,
            outputs: newOutputs,
          },
        };
      }

      case 'function-return': {
        const nodeData = node.data as FunctionReturnData;
        const newInputs = nodeData.inputs.map(port => {
          const varId = makeVariableId(node.id, port.id);
          const propagatedType = propagationResult.types.get(varId);
          return {
            ...port,
            concreteType: propagatedType || port.concreteType,
          };
        });

        return {
          ...node,
          data: {
            ...nodeData,
            inputs: newInputs,
          },
        };
      }

      case 'function-call': {
        const nodeData = node.data as FunctionCallData;
        const newInputs = nodeData.inputs.map(port => {
          const varId = makeVariableId(node.id, port.id);
          const propagatedType = propagationResult.types.get(varId);
          return {
            ...port,
            concreteType: propagatedType || port.concreteType,
          };
        });
        const newOutputs = nodeData.outputs.map(port => {
          const varId = makeVariableId(node.id, port.id);
          const propagatedType = propagationResult.types.get(varId);
          return {
            ...port,
            concreteType: propagatedType || port.concreteType,
          };
        });

        return {
          ...node,
          data: {
            ...nodeData,
            inputs: newInputs,
            outputs: newOutputs,
          },
        };
      }

      default:
        return node;
    }
  });
}
