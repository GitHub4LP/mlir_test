/**
 * Type System Service
 * 
 * 管理类型约束、类型兼容性检查。
 * 
 * 注意：类型传播现在由 typePropagation 模块处理。
 * 这个模块主要提供：
 * 1. 类型兼容性检查（isCompatible, canConnect）
 * 2. 类型颜色（getTypeColor）
 * 3. Trait 检查（hasSameOperandsAndResultTypeTrait）
 * 
 * 类型约束数据从后端动态加载，见 typeConstraintStore.ts
 */

import type { OperationDef } from '../types';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { PortRef, PortKind } from './port';

// ============================================================================
// 类型约束查询（委托给 store）
// ============================================================================

/**
 * 规范化类型字符串（小写 → 大写形式）
 */
export function normalizeType(type: string): string {
  // i32 -> I32
  const intMatch = type.match(/^i(\d+)$/);
  if (intMatch) {
    return `I${intMatch[1]}`;
  }
  
  // si32 -> SI32
  const sintMatch = type.match(/^si(\d+)$/);
  if (sintMatch) {
    return `SI${sintMatch[1]}`;
  }
  
  // ui32 -> UI32
  const uintMatch = type.match(/^ui(\d+)$/);
  if (uintMatch) {
    return `UI${uintMatch[1]}`;
  }
  
  // f32 -> F32
  const floatMatch = type.match(/^f(\d+)$/);
  if (floatMatch) {
    return `F${floatMatch[1]}`;
  }
  
  // bf16 -> BF16
  const bfloatMatch = type.match(/^bf(\d+)$/);
  if (bfloatMatch) {
    return `BF${bfloatMatch[1]}`;
  }
  
  // tf32 -> TF32
  const tfloatMatch = type.match(/^tf(\d+)$/);
  if (tfloatMatch) {
    return `TF${tfloatMatch[1]}`;
  }
  
  // index -> Index
  if (type === 'index') {
    return 'Index';
  }
  
  return type;
}

/**
 * 获取约束映射到的类型约束集合的元素
 * 委托给 typeConstraintStore
 */
export function getConstraintElements(constraint: string): string[] {
  return useTypeConstraintStore.getState().getConstraintElements(constraint);
}

/**
 * 检查类型约束是否是抽象的（映射到多个元素）
 */
export function isAbstractConstraint(constraint: string): boolean {
  const types = useTypeConstraintStore.getState().getConstraintElements(constraint);
  return types.length > 1;
}

/**
 * 约束解析结果
 * 
 * 三种情况：
 * - 'fixed': 固定类型，约束本身就是具体类型（I1 → [I1]），自动成为传播源
 * - 'single': 单一映射，约束只映射到一个类型（BoolLike → [I1]），被动接收传播
 * - 'multi': 多选，约束映射到多个类型，需要用户选择
 */
export type ConstraintKind = 'fixed' | 'single' | 'multi';

export interface ConstraintAnalysis {
  kind: ConstraintKind;
  /** 如果是 fixed 或 single，这是唯一的具体类型 */
  resolvedType: string | null;
}

/**
 * 分析约束类型
 * 
 * @returns 约束分析结果
 */
export function analyzeConstraint(constraint: string): ConstraintAnalysis {
  if (!constraint) {
    return { kind: 'multi', resolvedType: null };
  }
  
  // Variadic<...> 类型：解析内部类型
  const variadicMatch = constraint.match(/^Variadic<(.+)>$/);
  if (variadicMatch) {
    return analyzeConstraint(variadicMatch[1]);
  }
  
  // AnyOf<...> 类型：合成约束，允许选择
  if (constraint.startsWith('AnyOf<')) {
    return { kind: 'multi', resolvedType: null };
  }
  
  const { buildableTypes, isLoaded, getConstraintElements: storeGetConstraintElements } = useTypeConstraintStore.getState();
  
  // 如果数据还没加载，默认允许选择
  if (!isLoaded) {
    return { kind: 'multi', resolvedType: null };
  }
  
  const isBuildableType = buildableTypes.includes(constraint);
  const types = storeGetConstraintElements(constraint);
  
  // 空类型列表：允许选择任意类型
  if (types.length === 0) {
    return { kind: 'multi', resolvedType: null };
  }
  
  if (types.length === 1) {
    const resolvedType = types[0];
    // 检查解析出的类型是否是 BuildableType
    const resolvedIsBuildable = buildableTypes.includes(resolvedType);
    
    if (resolvedType === constraint && isBuildableType) {
      // 约束本身是具体类型：固定类型
      return { kind: 'fixed', resolvedType: constraint };
    } else if (resolvedIsBuildable) {
      // 约束映射到一个具体类型：单一映射
      return { kind: 'single', resolvedType };
    } else {
      // 解析出的类型不是 BuildableType（如 TypedAttrInterface → TypedAttrInterface）
      // 这种情况应该视为多选
      return { kind: 'multi', resolvedType: null };
    }
  }
  
  // 多个选项
  return { kind: 'multi', resolvedType: null };
}

/**
 * 判断端口的类型状态
 * 
 * @param portId - 端口ID（如 "data-in-lhs", "data-out-result"）
 * @param nodeId - 节点ID
 * @param pinnedTypes - 用户选择的类型
 * @param propagatedTypes - 传播得到的类型（inputTypes 或 outputTypes）
 * @param edges - 图中的连线
 * @returns 类型状态
 */
export function getPortTypeState(
  portId: string,
  nodeId: string,
  pinnedTypes: Record<string, string>,
  inputTypes: Record<string, string>,
  outputTypes: Record<string, string>,
  edges: Array<{source: string; target: string; sourceHandle?: string | null; targetHandle?: string | null}>
): {
  displayType: string | null;
  canEdit: boolean;
} {
  const isPinned = !!pinnedTypes[portId];
  
  // 获取传播类型
  let propagated: string | undefined;
  const parsed = PortRef.parseHandleId(portId);
  if (parsed) {
    const name = parsed.name.replace(/_\d+$/, '');
    if (parsed.kind === PortKind.DataIn) {
      propagated = inputTypes[name];
    } else if (parsed.kind === PortKind.DataOut) {
      propagated = outputTypes[name];
    }
  }
  
  // 检查是否有连线
  const isConnected = edges.some(e => 
    (e.source === nodeId && e.sourceHandle === portId) ||
    (e.target === nodeId && e.targetHandle === portId)
  );
  
  // 情况1：用户已选择且有连线 → 禁用
  if (isPinned && isConnected) {
    return {
      displayType: pinnedTypes[portId],
      canEdit: false
    };
  }
  
  // 情况2：用户已选择但无连线 → 可编辑
  if (isPinned) {
    return {
      displayType: pinnedTypes[portId],
      canEdit: true
    };
  }
  
  // 情况3：有传播类型 → 禁用
  if (propagated) {
    return {
      displayType: propagated,
      canEdit: false
    };
  }
  
  // 情况4：都没有 → 可选择
  return {
    displayType: null,
    canEdit: true
  };
}

/**
 * Checks if a source type is compatible with a target type constraint.
 */
export function isCompatible(sourceType: string, targetConstraint: string): boolean {
  const normalizedSource = normalizeType(sourceType);
  const normalizedTarget = normalizeType(targetConstraint);
  
  if (normalizedSource === normalizedTarget) {
    return true;
  }
  
  if (sourceType === targetConstraint) {
    return true;
  }
  
  const sourceElements = getConstraintElements(sourceType);
  const targetElements = getConstraintElements(targetConstraint);
  const targetTypeSet = new Set(targetElements);
  
  if (sourceElements.length === 1) {
    return targetTypeSet.has(sourceElements[0]);
  }
  
  return sourceElements.some(type => targetTypeSet.has(type));
}

/**
 * Checks if two types can be connected (bidirectional compatibility check).
 */
export function canConnect(type1: string, type2: string): boolean {
  return isCompatible(type1, type2) || isCompatible(type2, type1);
}

/**
 * Finds the most specific common type between two type constraints.
 */
export function findCommonType(constraint1: string, constraint2: string): string | null {
  const types1 = getConstraintElements(constraint1);
  const types2 = getConstraintElements(constraint2);
  
  const commonTypes = types1.filter(t => types2.includes(t));
  
  if (commonTypes.length === 0) {
    return null;
  }
  
  if (types1.length === 1 && commonTypes.includes(types1[0])) {
    return constraint1;
  }
  if (types2.length === 1 && commonTypes.includes(types2[0])) {
    return constraint2;
  }
  
  return commonTypes[0];
}

// ============================================================================
// Trait 检查
// ============================================================================

/**
 * Trait names that affect type propagation
 */
export const TYPE_PROPAGATION_TRAITS = {
  SAME_OPERANDS_AND_RESULT_TYPE: 'SameOperandsAndResultType',
  ALL_TYPES_MATCH: 'AllTypesMatch',
  SAME_TYPE_OPERANDS: 'SameTypeOperands',
} as const;

/**
 * Checks if an operation has the SameOperandsAndResultType trait
 */
export function hasSameOperandsAndResultTypeTrait(operation: OperationDef): boolean {
  return operation.traits.some(
    trait => trait === TYPE_PROPAGATION_TRAITS.SAME_OPERANDS_AND_RESULT_TYPE ||
             trait.includes(TYPE_PROPAGATION_TRAITS.SAME_OPERANDS_AND_RESULT_TYPE)
  );
}

/**
 * Checks if an operation has the AllTypesMatch trait
 */
export function hasAllTypesMatchTrait(operation: OperationDef): boolean {
  return operation.traits.some(
    trait => trait === TYPE_PROPAGATION_TRAITS.ALL_TYPES_MATCH ||
             trait.includes(TYPE_PROPAGATION_TRAITS.ALL_TYPES_MATCH)
  );
}

// ============================================================================
// 类型颜色
// ============================================================================

import { getTypeColor as getTypeColorFromMapping } from './typeColorMapping';

/**
 * Gets the color for a type (used for port visualization).
 * 
 * 使用程序化的颜色映射系统：
 * - 基本类型（BuildableType）使用配置的颜色
 * - 复合类型（约束）通过展开到基本类型集合，然后对颜色进行平均
 * 
 * @param displayType - 显示类型（可能是 BuildableType 或约束）
 * @returns hex 颜色字符串
 */
export function getTypeColor(displayType: string): string {
  const getConstraintElements = useTypeConstraintStore.getState().getConstraintElements;
  return getTypeColorFromMapping(displayType, getConstraintElements);
}
