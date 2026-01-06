/**
 * Type Propagation Types
 * 
 * 使用 PortRef 作为端口标识符，确保类型安全。
 */

import { PortRef } from '../port';

/**
 * 变量标识符：使用 PortRef.key 格式 (nodeId:kind:name)
 */
export type VariableId = string;

/**
 * 类型源：用户显式选择的类型
 */
export interface TypeSource {
  portRef: PortRef;
  type: string;  // 具体类型或约束名，如 "I32" 或 "SignlessIntegerLike"
}

/**
 * 传播边类型
 * 
 * - 'full': 完整类型传播（SameOperandsAndResultType, SameTypeOperands）
 * - 'element': 元素类型传播（SameOperandsElementType, SameOperandsAndResultElementType）
 */
export type EdgeKind = 'full' | 'element';

/**
 * 传播边信息
 */
export interface PropagationEdge {
  target: VariableId;
  kind: EdgeKind;
}

/**
 * 传播图：描述类型如何从一个端口流向另一个端口
 * 
 * 键：源变量 ID (PortRef.key)
 * 值：可以传播到的目标边集合（包含边类型）
 * 
 * 注意：为了向后兼容，同时支持旧的 Set<VariableId> 格式
 */
export type PropagationGraph = Map<VariableId, Set<VariableId>>;

/**
 * 扩展传播图：支持边类型标记
 */
export type ExtendedPropagationGraph = Map<VariableId, PropagationEdge[]>;

/**
 * 传播结果
 * 
 * effectiveSets: 每个端口的有效集合（具体类型数组）
 * sources: 传播路径（用于调试）
 */
export interface PropagationResult {
  /** 所有端口的有效集合：varId → 具体类型数组 */
  effectiveSets: Map<VariableId, string[]>;
  
  /** 传播路径（用于调试）：varId → 从哪个 varId 传播来的 */
  sources: Map<VariableId, VariableId | null>;
}

/**
 * 从 PortRef 创建变量 ID
 */
export function makeVariableId(portRef: PortRef): VariableId {
  return portRef.key;
}

/**
 * 兼容旧 API：从 nodeId 和 handleId 创建变量 ID
 * 用于从 React Flow 的 edge 数据创建变量 ID
 */
export function makeVariableIdFromHandle(nodeId: string, handleId: string): VariableId | null {
  const portRef = PortRef.fromHandle(nodeId, handleId);
  return portRef ? portRef.key : null;
}

/**
 * 解析变量 ID 为 PortRef
 */
export function parseVariableId(varId: VariableId): PortRef | null {
  return PortRef.parse(varId);
}
