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
import type { FunctionDef, FunctionEntryData, FunctionReturnData } from '../../types';
import { computePropagation, applyPropagationResult } from './propagator';
import { getDisplayType } from '../typeSelectorRenderer';
import { dataOutHandle, dataInHandle } from '../port';
import { usePortStateStore, makePortKey, type PortState as StorePortState } from '../../stores/portStateStore';

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
 * @param findSubsetConstraints - 找出所有元素集合是给定集合子集的约束名
 * @returns 更新后的节点列表（包含传播结果）
 */
export function triggerTypePropagation(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null,
  findSubsetConstraints?: (E: string[]) => string[]
): EditorNode[] {
  const result = triggerTypePropagationWithSignature(
    nodes, edges, currentFunction, getConstraintElements, pickConstraintName, findSubsetConstraints
  );
  return result.nodes;
}

/**
 * 触发类型传播并返回更新后的节点和签名信息
 * 
 * 用于需要同步签名到 FunctionDef 的场景。
 * portStates 直接写入 node.data.portStates，供所有渲染器通过 props 读取。
 */
export function triggerTypePropagationWithSignature(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null,
  findSubsetConstraints?: (E: string[]) => string[]
): PropagationTriggerResult {
  // 使用新的 computePropagation 函数
  const propagationResult = computePropagation(
    nodes,
    edges,
    currentFunction,
    getConstraintElements,
    pickConstraintName,
    findSubsetConstraints || (() => [])
  );
  
  // 应用传播结果到节点（包含 portStates）
  const updatedNodes = applyPropagationResult(nodes, propagationResult);
  const signature = extractSignatureDisplayTypes(updatedNodes);
  
  // 同时更新 portStateStore（兼容现有代码，后续可移除）
  const portStatesMap = new Map<string, StorePortState>();
  for (const [varId, state] of propagationResult.portStates) {
    // varId 格式: nodeId:kind:name
    const parts = varId.split(':');
    if (parts.length >= 3) {
      const nodeId = parts[0];
      const handleId = `${parts[1]}-${parts[2]}`;
      portStatesMap.set(makePortKey(nodeId, handleId), state as StorePortState);
    }
  }
  usePortStateStore.getState().updatePortStates(portStatesMap);
  
  return { nodes: updatedNodes, signature };
}

