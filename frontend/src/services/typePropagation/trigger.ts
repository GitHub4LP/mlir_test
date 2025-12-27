/**
 * 类型传播触发器
 * 
 * 统一的类型传播触发逻辑，供所有需要触发传播的地方使用。
 * 避免在多个组件中重复相同的传播代码。
 * 
 * 设计原则：
 * - 使用框架无关的 EditorNode/EditorEdge 类型
 * - 不依赖任何渲染框架（React Flow、Vue Flow 等）
 * - 传播完成后更新 portStateStore，供所有渲染器读取
 */

import type { EditorNode, EditorEdge } from '../../editor/types';
import type { FunctionDef, FunctionEntryData, FunctionReturnData, BlueprintNodeData, FunctionCallData } from '../../types';
import { computePropagationWithNarrowing, applyPropagationResult, extractPortConstraints, buildPropagationGraph, extractTypeSources } from './propagator';
import { getDisplayType } from '../typeSelectorRenderer';
import { dataOutHandle, dataInHandle, dataIn, dataOut, PortRef } from '../port';
import { usePortStateStore, makePortKey, type PortState } from '../../stores/portStateStore';

/**
 * 从传播后的节点提取 Entry/Return 端口的 displayType
 */
function extractSignatureDisplayTypes(
  nodes: EditorNode[]
): {
  parameters: Record<string, string>;
  returnTypes: Record<string, string>;
} {
  const parameters: Record<string, string> = {};
  const returnTypes: Record<string, string> = {};

  for (const node of nodes) {
    if (node.type === 'function-entry') {
      const data = node.data as FunctionEntryData;
      for (const port of data.outputs) {
        const pin = {
          id: dataOutHandle(port.name),
          label: port.name,
          typeConstraint: 'AnyType',
          displayName: 'AnyType',
        };
        parameters[port.name] = getDisplayType(pin, data);
      }
    } else if (node.type === 'function-return') {
      const data = node.data as FunctionReturnData;
      for (const port of data.inputs) {
        const pin = {
          id: dataInHandle(port.name),
          label: port.name,
          typeConstraint: 'AnyType',
          displayName: 'AnyType',
        };
        returnTypes[port.name] = getDisplayType(pin, data);
      }
    }
  }

  return { parameters, returnTypes };
}

/**
 * 类型传播结果（包含签名变化信息）
 */
export interface PropagationTriggerResult {
  /** 更新后的节点列表 */
  nodes: EditorNode[];
  /** Entry/Return 端口的 displayType（用于同步到 FunctionDef） */
  signature: {
    parameters: Record<string, string>;
    returnTypes: Record<string, string>;
  };
}

/**
 * 触发类型传播并返回更新后的节点
 * 
 * 这是一个纯函数，不依赖 React hooks，可以在任何地方调用。
 * 
 * @param nodes - 当前节点列表（EditorNode 类型）
 * @param edges - 当前边列表（EditorEdge 类型）
 * @param currentFunction - 当前函数定义（用于函数级 Traits）
 * @param getConstraintElements - 获取约束映射到的类型约束集合元素
 * @param pickConstraintName - 选择约束名称
 * @returns 更新后的节点列表（包含传播结果）
 */
export function triggerTypePropagation(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null
): EditorNode[] {
  const result = triggerTypePropagationWithSignature(
    nodes, edges, currentFunction, getConstraintElements, pickConstraintName
  );
  return result.nodes;
}

/**
 * 触发类型传播并返回更新后的节点和签名信息
 * 
 * 用于需要同步签名到 FunctionDef 的场景。
 * 同时更新 portStateStore，供所有渲染器读取。
 */
export function triggerTypePropagationWithSignature(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null
): PropagationTriggerResult {
  const propagationResult = computePropagationWithNarrowing(
    nodes,
    edges,
    currentFunction,
    getConstraintElements,
    pickConstraintName
  );
  const updatedNodes = applyPropagationResult(nodes, propagationResult);
  const signature = extractSignatureDisplayTypes(updatedNodes);
  
  // 计算并更新 portStateStore
  const portStates = computePortStates(
    updatedNodes,
    edges,
    currentFunction,
    getConstraintElements
  );
  usePortStateStore.getState().updatePortStates(portStates);
  
  return { nodes: updatedNodes, signature };
}

/**
 * 计算所有端口的状态（displayType, constraint, canEdit）
 * 
 * canEdit 的计算逻辑（来自 architecture.md）：
 * canEdit = options.length > 1 && !isExternallyDetermined
 * isExternallyDetermined = propagatedType !== null && !isPinned
 * 
 * - 被传播的端口不可编辑：类型由上游决定
 * - 自己 pin 的类型可修改：不算"外部决定"
 */
function computePortStates(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[]
): Map<string, PortState> {
  const portStates = new Map<string, PortState>();
  
  // 提取原始约束
  const portConstraints = extractPortConstraints(nodes, currentFunction);
  
  // 构建传播图（用于计算 options）
  const graph = buildPropagationGraph(nodes, edges, currentFunction);
  
  // 提取所有类型源（包含 pinnedTypes 和单一元素约束）
  const allSources = extractTypeSources(nodes);
  const sourceSet = new Set(allSources.map(s => s.portRef.key));
  
  // 提取所有用户显式 pin 的端口（用于判断 isExternallyDetermined）
  const pinnedPorts = extractPinnedPorts(nodes);
  
  // 遍历所有节点，收集端口状态
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
              const displayType = data.inputTypes?.[arg.name] || arg.typeConstraint;
              const canEdit = computeCanEdit(portRef.key, graph, portConstraints, sourceSet, pinnedPorts, getConstraintElements);
              portStates.set(makePortKey(node.id, portRef.handleId), {
                displayType,
                constraint: arg.typeConstraint,
                canEdit,
              });
            }
          } else {
            const portRef = dataIn(node.id, arg.name);
            const displayType = data.inputTypes?.[arg.name] || arg.typeConstraint;
            const canEdit = computeCanEdit(portRef.key, graph, portConstraints, sourceSet, pinnedPorts, getConstraintElements);
            portStates.set(makePortKey(node.id, portRef.handleId), {
              displayType,
              constraint: arg.typeConstraint,
              canEdit,
            });
          }
        }
        
        // 输出端口
        for (const result of operation.results) {
          if (result.isVariadic) {
            const count = variadicCounts[result.name] ?? 1;
            for (let i = 0; i < count; i++) {
              const portRef = dataOut(node.id, `${result.name}_${i}`);
              const displayType = data.outputTypes?.[result.name] || result.typeConstraint;
              const canEdit = computeCanEdit(portRef.key, graph, portConstraints, sourceSet, pinnedPorts, getConstraintElements);
              portStates.set(makePortKey(node.id, portRef.handleId), {
                displayType,
                constraint: result.typeConstraint,
                canEdit,
              });
            }
          } else {
            const portRef = dataOut(node.id, result.name);
            const displayType = data.outputTypes?.[result.name] || result.typeConstraint;
            const canEdit = computeCanEdit(portRef.key, graph, portConstraints, sourceSet, pinnedPorts, getConstraintElements);
            portStates.set(makePortKey(node.id, portRef.handleId), {
              displayType,
              constraint: result.typeConstraint,
              canEdit,
            });
          }
        }
        break;
      }
      
      case 'function-entry': {
        const data = node.data as FunctionEntryData;
        for (const port of data.outputs) {
          const portRef = dataOut(node.id, port.name);
          const displayType = data.outputTypes?.[port.name] || port.typeConstraint;
          const constraint = data.isMain ? port.typeConstraint : 'AnyType';
          const canEdit = computeCanEdit(portRef.key, graph, portConstraints, sourceSet, pinnedPorts, getConstraintElements);
          portStates.set(makePortKey(node.id, portRef.handleId), {
            displayType,
            constraint,
            canEdit,
          });
        }
        break;
      }
      
      case 'function-return': {
        const data = node.data as FunctionReturnData;
        for (const port of data.inputs) {
          const portRef = dataIn(node.id, port.name);
          const displayType = data.inputTypes?.[port.name] || port.typeConstraint;
          const constraint = data.isMain ? port.typeConstraint : 'AnyType';
          const canEdit = computeCanEdit(portRef.key, graph, portConstraints, sourceSet, pinnedPorts, getConstraintElements);
          portStates.set(makePortKey(node.id, portRef.handleId), {
            displayType,
            constraint,
            canEdit,
          });
        }
        break;
      }
      
      case 'function-call': {
        const data = node.data as FunctionCallData;
        for (const port of data.inputs) {
          const portRef = dataIn(node.id, port.name);
          const displayType = data.inputTypes?.[port.name] || port.typeConstraint;
          const canEdit = computeCanEdit(portRef.key, graph, portConstraints, sourceSet, pinnedPorts, getConstraintElements);
          portStates.set(makePortKey(node.id, portRef.handleId), {
            displayType,
            constraint: port.typeConstraint,
            canEdit,
          });
        }
        for (const port of data.outputs) {
          const portRef = dataOut(node.id, port.name);
          const displayType = data.outputTypes?.[port.name] || port.typeConstraint;
          const canEdit = computeCanEdit(portRef.key, graph, portConstraints, sourceSet, pinnedPorts, getConstraintElements);
          portStates.set(makePortKey(node.id, portRef.handleId), {
            displayType,
            constraint: port.typeConstraint,
            canEdit,
          });
        }
        break;
      }
    }
  }
  
  return portStates;
}

/**
 * 提取所有用户显式 pin 的端口
 * 
 * 与 extractTypeSources 不同，这里只提取用户显式选择的类型（pinnedTypes），
 * 不包括单一元素约束自动解析的类型。
 */
function extractPinnedPorts(nodes: EditorNode[]): Set<string> {
  const pinned = new Set<string>();
  
  for (const node of nodes) {
    let pinnedTypes: Record<string, string> | undefined;
    
    switch (node.type) {
      case 'operation':
        pinnedTypes = (node.data as BlueprintNodeData).pinnedTypes;
        break;
      case 'function-entry':
        pinnedTypes = (node.data as FunctionEntryData).pinnedTypes;
        break;
      case 'function-return':
        pinnedTypes = (node.data as FunctionReturnData).pinnedTypes;
        break;
      case 'function-call':
        pinnedTypes = (node.data as FunctionCallData).pinnedTypes;
        break;
    }
    
    if (pinnedTypes) {
      for (const [handleId, type] of Object.entries(pinnedTypes)) {
        if (type) {
          const portRef = PortRef.fromHandle(node.id, handleId);
          if (portRef) {
            pinned.add(portRef.key);
          }
        }
      }
    }
  }
  
  return pinned;
}

/**
 * 计算单个端口的 canEdit
 * 
 * 根据 architecture.md 的规则：
 * canEdit = options.length > 1 && !isExternallyDetermined
 * isExternallyDetermined = propagatedType !== null && !isPinned
 * 
 * - 被传播的端口不可编辑：类型由上游决定
 * - 自己 pin 的类型可修改：不算"外部决定"
 */
function computeCanEdit(
  portKey: string,
  graph: Map<string, Set<string>>,
  portConstraints: Map<string, string>,
  sourceSet: Set<string>,
  pinnedPorts: Set<string>,
  getConstraintElements: (constraint: string) => string[]
): boolean {
  const myConstraint = portConstraints.get(portKey);
  if (!myConstraint) return false;
  
  // 检查是否被外部传播（isExternallyDetermined）
  // isExternallyDetermined = propagatedType !== null && !isPinned
  // 如果端口是类型源（sourceSet 包含它）但不是用户显式 pin 的，说明是单一元素约束自动解析的
  // 这种情况下，端口有传播类型但不是用户 pin 的，应该算"外部决定"吗？
  // 根据规则，"被传播的端口不可编辑"，但单一元素约束是自己的约束决定的，不是被传播的
  // 所以这里需要区分：
  // - 用户 pin 的：可编辑（自己选的）
  // - 单一元素约束：不可编辑（约束只有一个选项）
  // - 被其他端口传播的：不可编辑（外部决定）
  
  const isPinned = pinnedPorts.has(portKey);
  
  // 如果是源但不是用户 pin 的，说明是单一元素约束，options 只有 1 个，canEdit = false
  // 如果不是源，需要检查是否被传播
  // 如果是用户 pin 的，可以修改自己的选择
  
  // 获取原始约束的元素
  let options = getConstraintElements(myConstraint);
  if (options.length === 0) return false;
  if (options.length === 1) return false; // 只有一个选项，不可编辑
  
  // 与邻居有效类型求交集（排除自己的影响）
  const neighbors = graph.get(portKey);
  if (neighbors) {
    // 执行一次排除自己的传播
    const sourcesWithoutSelf = Array.from(sourceSet)
      .filter(s => s !== portKey)
      .map(s => {
        const constraint = portConstraints.get(s);
        const elements = constraint ? getConstraintElements(constraint) : [];
        return { key: s, type: elements[0] || constraint || '' };
      })
      .filter(s => s.type);
    
    // 简化传播：BFS
    const propagatedTypes = new Map<string, string>();
    const queue = [...sourcesWithoutSelf];
    for (const s of queue) {
      propagatedTypes.set(s.key, s.type);
    }
    
    let idx = 0;
    while (idx < queue.length) {
      const current = queue[idx++];
      const currentNeighbors = graph.get(current.key);
      if (currentNeighbors) {
        for (const neighbor of currentNeighbors) {
          if (!propagatedTypes.has(neighbor)) {
            propagatedTypes.set(neighbor, current.type);
            queue.push({ key: neighbor, type: current.type });
          }
        }
      }
    }
    
    // 检查是否被外部传播（排除自己后，自己是否有传播类型）
    const propagatedType = propagatedTypes.get(portKey);
    const isExternallyDetermined = propagatedType !== undefined && !isPinned;
    
    // 如果被外部传播，不可编辑
    if (isExternallyDetermined) {
      return false;
    }
    
    // 与邻居有效类型求交集
    for (const neighborKey of neighbors) {
      const neighborType = propagatedTypes.get(neighborKey) || portConstraints.get(neighborKey);
      if (neighborType) {
        const neighborElements = getConstraintElements(neighborType);
        if (neighborElements.length > 0) {
          options = options.filter(t => neighborElements.includes(t));
        }
      }
    }
  }
  
  return options.length > 1;
}
