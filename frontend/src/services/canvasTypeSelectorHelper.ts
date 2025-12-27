/**
 * Canvas 类型选择器辅助函数
 * 
 * 提供从 typeConstraintStore 构建 ConstraintData 的辅助函数，
 * 供 Canvas/GPU 编辑器使用。
 */

import type { ConstraintData } from '../editor/adapters/canvas/ui/TypeSelector';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';

/**
 * 从 typeConstraintStore 获取 ConstraintData
 * 
 * @param constraint - 约束名称（如 'SignlessIntegerLike'）
 * @param allowedTypes - 允许的具体类型列表（来自后端 AnyTypeOf 解析）
 * @returns ConstraintData 对象
 */
export function getConstraintDataFromStore(
  constraint?: string,
  allowedTypes?: string[]
): ConstraintData {
  const store = useTypeConstraintStore.getState();
  
  return {
    constraint,
    allowedTypes,
    buildableTypes: store.buildableTypes,
    constraintDefs: store.constraintDefs,
    getConstraintElements: store.getConstraintElements,
    isShapedConstraint: store.isShapedConstraint,
    getAllowedContainers: store.getAllowedContainers,
  };
}

/**
 * 创建空的 ConstraintData（无约束限制）
 * 
 * 用于 Entry/Return 节点，允许选择任意类型
 */
export function getUnconstrainedData(): ConstraintData {
  return getConstraintDataFromStore('AnyType');
}
