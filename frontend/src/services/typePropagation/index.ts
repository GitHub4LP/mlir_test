/**
 * Type Propagation System
 * 
 * 基于数据流的类型传播模型。
 * 
 * 核心概念：
 * - Source：用户显式选择的类型（存储在节点数据中）
 * - 传播路径：由 Trait（节点内）和连线（节点间）定义
 * - 派生类型：通过传播计算得到，不持久化
 * - 有效集合：传播结果，存储为具体类型数组
 */

export { 
  propagateTypes, 
  buildPropagationGraph,
  extractPortConstraints,
  extractTypeSources,
  computePropagation,
  computeOptionsExcludingSelf,
  applyPropagationResult,
  computeDisplayTypes,
  computeDisplayTypeFromSet,
  computePortState,
  getAllowedDialectsForPort,
} from './propagator';
export type { DialectFilterConfig } from './propagator';
export { triggerTypePropagation, triggerTypePropagationWithSignature } from './trigger';
export type { PropagationGraph, TypeSource, PropagationResult } from './types';
export type { PropagationTriggerResult } from './trigger';
