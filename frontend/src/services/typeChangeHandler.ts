/**
 * 统一的类型变更处理逻辑
 * 
 * 所有四种节点类型（Operation、Call、Entry、Return）使用完全相同的逻辑：
 * 1. 判断是否应该 pin（固定）类型
 * 2. 更新节点的 pinnedTypes
 * 3. 触发类型传播
 * 
 * 数据存储统一：
 * - pinnedTypes：用户显式选择的类型
 * - inputTypes/outputTypes：传播结果
 * - displayType = pinnedTypes[portId] > inputTypes/outputTypes[portName] > 原始约束
 * 
 * Entry/Return 的显示类型就是函数签名。
 */

import type { Node, Edge } from '@xyflow/react';
import type { FunctionDef } from '../types';
import { isAbstractConstraint } from './typeSystem';
import { triggerTypePropagationWithSignature } from './typePropagation';

/**
 * 判断是否应该 pin 类型
 * 
 * - 选择的类型等于原始约束 → 不 pin（恢复默认）
 * - 选择的是抽象约束 → 不 pin
 * - 其他情况 → pin
 */
export function shouldPinType(type: string, originalConstraint?: string): boolean {
  return Boolean(type && type !== originalConstraint && !isAbstractConstraint(type));
}

/**
 * 类型变更处理的依赖项
 */
export interface TypeChangeHandlerDeps {
  edges: Edge[];
  getCurrentFunction: () => FunctionDef | null;
  getConcreteTypes: (constraint: string) => string[];
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null;
  /** 可选：签名同步回调，传播完成后调用 */
  onSignatureChange?: (
    functionId: string,
    parameterConstraints: Record<string, string>,
    returnTypeConstraints: Record<string, string>
  ) => void;
}

/**
 * 更新节点的 pinnedTypes 并触发类型传播
 * 
 * 用于 BlueprintNode 和 FunctionCallNode 的类型选择处理。
 */
export function handlePinnedTypeChange<T extends { pinnedTypes?: Record<string, string> }>(
  nodeId: string,
  portId: string,
  type: string,
  originalConstraint: string | undefined,
  currentNodes: Node[],
  deps: TypeChangeHandlerDeps
): Node[] {
  const shouldPin = shouldPinType(type, originalConstraint);

  // 1. 更新当前节点的 pinnedTypes
  const updatedNodes = currentNodes.map(node => {
    if (node.id === nodeId) {
      const nodeData = node.data as T;
      const newPinnedTypes = { ...(nodeData.pinnedTypes || {}) };

      if (shouldPin) {
        newPinnedTypes[portId] = type;
      } else {
        delete newPinnedTypes[portId];
      }

      return {
        ...node,
        data: {
          ...nodeData,
          pinnedTypes: newPinnedTypes,
        },
      };
    }
    return node;
  });

  // 2. 触发类型传播
  const currentFunction = deps.getCurrentFunction() ?? undefined;
  const result = triggerTypePropagationWithSignature(
    updatedNodes,
    deps.edges,
    currentFunction,
    deps.getConcreteTypes,
    deps.pickConstraintName
  );

  // 3. 同步签名到 FunctionDef（如果提供了回调）
  if (deps.onSignatureChange && currentFunction) {
    deps.onSignatureChange(
      currentFunction.id,
      result.signature.parameters,
      result.signature.returnTypes
    );
  }

  return result.nodes;
}

/**
 * 触发类型传播（不更新节点数据）
 * 
 * 用于需要重新传播但不改变 pinnedTypes 的场景，如：
 * - 添加/删除连线后
 * - 添加/删除参数后
 */
export function triggerPropagationOnly(
  currentNodes: Node[],
  deps: TypeChangeHandlerDeps
): Node[] {
  const currentFunction = deps.getCurrentFunction() ?? undefined;
  const result = triggerTypePropagationWithSignature(
    currentNodes,
    deps.edges,
    currentFunction,
    deps.getConcreteTypes,
    deps.pickConstraintName
  );

  // 同步签名到 FunctionDef（如果提供了回调）
  if (deps.onSignatureChange && currentFunction) {
    deps.onSignatureChange(
      currentFunction.id,
      result.signature.parameters,
      result.signature.returnTypes
    );
  }

  return result.nodes;
}
