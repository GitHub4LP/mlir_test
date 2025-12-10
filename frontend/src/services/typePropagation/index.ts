/**
 * Type Propagation System
 * 
 * 基于数据流的类型传播模型。
 * 
 * 核心概念：
 * - Source：用户显式选择的类型（存储在节点数据中）
 * - 传播路径：由 Trait（节点内）和连线（节点间）定义
 * - 派生类型：通过传播计算得到，不持久化
 */

export { 
  propagateTypes, 
  buildPropagationGraph,
  extractPortConstraints,
  computeNarrowedConstraints,
  computePropagationWithNarrowing,
} from './propagator';
export type { PropagationGraph, TypeSource, PropagationResult } from './types';
