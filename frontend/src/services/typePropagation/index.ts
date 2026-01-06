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
export type { PropagationGraph, ExtendedPropagationGraph, PropagationEdge, EdgeKind, TypeSource, PropagationResult } from './types';
export type { PropagationTriggerResult } from './trigger';

// 元素类型工具
export { 
  isContainerType, 
  extractElementType, 
  getContainerStructure, 
  applyElementToStructure 
} from './elementType';
export type { ContainerStructure } from './elementType';

// 扩展传播图
export { buildExtendedPropagationGraph, toSimplePropagationGraph } from './graph';

// 元素类型传播算法
export { propagateTypesWithElementEdges } from './elementPropagation';

// Traits 推断
export { inferFunctionTraits } from './traitsInference';
