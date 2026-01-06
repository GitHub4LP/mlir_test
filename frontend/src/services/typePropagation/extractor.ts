/**
 * 数据提取器
 * 
 * 从节点数据中提取类型源、端口约束等信息。
 */

import type { EditorNode, EditorEdge } from '../../editor/types';
import type { VariableId, TypeSource } from './types';
import type { BlueprintNodeData, FunctionEntryData, FunctionReturnData, FunctionCallData, FunctionDef } from '../../types';
import { PortRef, dataIn, dataOut } from '../port';
import { typeConstraintStore } from '../../stores';
import { buildPropagationGraph } from './graph';
import { propagateTypes } from './algorithm';

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
export function extractTypeSources(nodes: EditorNode[]): TypeSource[] {
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
 * @param getFunctionById - 函数查找器（可选，用于获取 Call 节点被调用函数的 Traits）
 * @returns 可选的具体类型列表
 */
export function computeOptionsExcludingSelf(
  portKey: VariableId,
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  getFunctionById?: (functionId: string) => FunctionDef | null
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
  
  // 3. 构建传播图（包含 Call 节点的被调用函数 traits）
  const graph = buildPropagationGraph(nodes, edges, currentFunction, getFunctionById);
  
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
        // 使用兼容容器类型的交集计算
        options = computeOptionsIntersection(options, neighborSet);
      }
    }
  }
  
  return options;
}

/**
 * 计算可选集与邻居集合的交集（支持容器类型）
 */
function computeOptionsIntersection(options: string[], neighborSet: string[]): string[] {
  const result: string[] = [];
  
  for (const opt of options) {
    // 检查 opt 是否与 neighborSet 中的任何类型兼容
    for (const neighbor of neighborSet) {
      if (opt === neighbor) {
        if (!result.includes(opt)) {
          result.push(opt);
        }
        break;
      }
      // 对于容器类型，使用 computeTypeIntersection
      // 但这里我们只需要检查兼容性，不需要计算具体交集
      // 因为 options 是标量类型列表，neighborSet 可能包含容器类型
      // 如果 neighborSet 包含容器类型，标量类型不应该与之兼容
      // 所以这里简单使用相等检查即可
    }
  }
  
  return result;
}
