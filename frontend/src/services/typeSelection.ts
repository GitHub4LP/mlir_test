/**
 * 统一的类型选择逻辑
 *
 * 所有节点类型（Operation、Call、Entry、Return）使用完全相同的逻辑：
 * 1. 计算可选集 = computeOptionsExcludingSelf(...)
 * 2. canEdit = options.length > 1
 *
 * 唯一区别（通过 extractPortConstraints 内部处理）：
 * - Operation/Call：原始约束来自操作定义
 * - Entry/Return：原始约束是 'AnyType'
 *
 * main 函数不需要特殊处理：
 * - main Entry 没有参数端口
 * - main Return 的返回值是 I32，options = [I32]，canEdit = false
 */

import type { Node, Edge } from '@xyflow/react';
import type { FunctionDef } from '../types';
import { computeOptionsExcludingSelf } from './typePropagation/propagator';
import { PortRef } from './port';

export interface TypeSelectionResult {
  /** 可选的具体类型列表 */
  options: string[];
  /** 是否可编辑 */
  canEdit: boolean;
}

/**
 * 计算端口的类型选择状态（统一入口）
 *
 * @param nodeId - 节点 ID
 * @param portId - 端口 handle ID（如 'data-out-a'）
 * @param nodes - 当前图的所有节点
 * @param edges - 当前图的所有边
 * @param currentFunction - 当前函数定义
 * @param getConstraintElements - 获取约束映射到的类型约束集合元素
 */
export function computeTypeSelectionState(
  nodeId: string,
  portId: string,
  nodes: Node[],
  edges: Edge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[]
): TypeSelectionResult {
  // 计算可选集（排除自己的影响）
  const portRef = PortRef.fromHandle(nodeId, portId);
  const options = portRef
    ? computeOptionsExcludingSelf(
        portRef.key,
        nodes,
        edges,
        currentFunction,
        getConstraintElements
      )
    : [];

  // canEdit = options.length > 1
  return {
    options,
    canEdit: options.length > 1,
  };
}
