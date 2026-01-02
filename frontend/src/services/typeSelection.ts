/**
 * 统一的类型选择逻辑
 *
 * 所有节点类型（Operation、Call、Entry、Return）使用完全相同的逻辑：
 * 1. 计算可选集 = computeOptionsExcludingSelf(...)
 * 2. 从可选集中找出所有元素集合是其子集的约束名
 * 3. canEdit = options.length > 1
 *
 * 唯一区别（通过 extractPortConstraints 内部处理）：
 * - Operation/Call：原始约束来自操作定义
 * - Entry/Return：原始约束是 'AnyType'
 *
 * main 函数不需要特殊处理：
 * - main Entry 没有参数端口
 * - main Return 的返回值是 I32，options = [I32]，canEdit = false
 * 
 * 设计原则：
 * - 使用框架无关的 EditorNode/EditorEdge 类型
 * - 不依赖任何渲染框架（React Flow、Vue Flow 等）
 */

import type { EditorNode, EditorEdge } from '../editor/types';
import type { FunctionDef } from '../types';
import { computeOptionsExcludingSelf } from './typePropagation/propagator';
import { PortRef } from './port';

export interface TypeSelectionResult {
  /** 可选的约束名列表（元素集合是有效集合子集的约束） */
  options: string[];
  /** 是否可编辑 */
  canEdit: boolean;
}

/**
 * 计算端口的类型选择状态（统一入口）
 *
 * @param nodeId - 节点 ID
 * @param portId - 端口 handle ID（如 'data-out-a'）
 * @param nodes - 当前图的所有节点（EditorNode 类型）
 * @param edges - 当前图的所有边（EditorEdge 类型）
 * @param currentFunction - 当前函数定义
 * @param getConstraintElements - 获取约束映射到的类型约束集合元素
 * @param findSubsetConstraints - 找出所有元素集合是有效集合子集的约束名
 */
export function computeTypeSelectionState(
  nodeId: string,
  portId: string,
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction: FunctionDef | undefined,
  getConstraintElements: (constraint: string) => string[],
  findSubsetConstraints: (E: string[]) => string[]
): TypeSelectionResult {
  // 计算有效集合（排除自己的影响）
  const portRef = PortRef.fromHandle(nodeId, portId);
  const effectiveSet = portRef
    ? computeOptionsExcludingSelf(
        portRef.key,
        nodes,
        edges,
        currentFunction,
        getConstraintElements
      )
    : [];

  // 从有效集合中找出所有元素集合是其子集的约束名
  const options = effectiveSet.length > 0 ? findSubsetConstraints(effectiveSet) : [];

  // canEdit = options.length > 1
  return {
    options,
    canEdit: options.length > 1,
  };
}
