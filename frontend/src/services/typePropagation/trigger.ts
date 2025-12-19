/**
 * 类型传播触发器
 * 
 * 统一的类型传播触发逻辑，供所有需要触发传播的地方使用。
 * 避免在多个组件中重复相同的传播代码。
 */

import type { Node, Edge } from '@xyflow/react';
import type { FunctionDef, FunctionEntryData, FunctionReturnData } from '../../types';
import { computePropagationWithNarrowing, applyPropagationResult } from './propagator';
import { getDisplayType } from '../typeSelectorRenderer';
import { dataOutHandle, dataInHandle } from '../port';

/**
 * 从传播后的节点提取 Entry/Return 端口的 displayType
 */
function extractSignatureDisplayTypes(
  nodes: Node[]
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
  nodes: Node[];
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
 * @param nodes - 当前节点列表
 * @param edges - 当前边列表
 * @param currentFunction - 当前函数定义（用于函数级 Traits）
 * @param getConstraintElements - 获取约束映射到的类型约束集合元素
 * @param pickConstraintName - 选择约束名称
 * @returns 更新后的节点列表（包含传播结果）
 */
export function triggerTypePropagation(
  nodes: Node[],
  edges: Edge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null
): Node[] {
  const result = triggerTypePropagationWithSignature(
    nodes, edges, currentFunction, getConstraintElements, pickConstraintName
  );
  return result.nodes;
}

/**
 * 触发类型传播并返回更新后的节点和签名信息
 * 
 * 用于需要同步签名到 FunctionDef 的场景。
 */
export function triggerTypePropagationWithSignature(
  nodes: Node[],
  edges: Edge[],
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
  
  return { nodes: updatedNodes, signature };
}
