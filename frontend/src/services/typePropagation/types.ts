/**
 * Type Propagation Types
 */

/**
 * 变量标识符：nodeId:portId
 */
export type VariableId = string;

/**
 * 类型源：用户显式选择的类型
 */
export interface TypeSource {
  nodeId: string;
  portId: string;
  type: string;  // 具体类型，如 "I32"
}

/**
 * 传播图：描述类型如何从一个端口流向另一个端口
 * 
 * 键：源变量 ID
 * 值：可以传播到的目标变量 ID 集合
 */
export type PropagationGraph = Map<VariableId, Set<VariableId>>;

/**
 * 传播结果
 */
export interface PropagationResult {
  /** 所有端口的类型（包括源和派生） */
  types: Map<VariableId, string>;
  
  /** 传播路径（用于调试）：varId → 从哪个 varId 传播来的 */
  sources: Map<VariableId, VariableId | null>;
}

/**
 * 创建变量 ID
 */
export function makeVariableId(nodeId: string, portId: string): VariableId {
  return `${nodeId}:${portId}`;
}

/**
 * 解析变量 ID
 */
export function parseVariableId(varId: VariableId): { nodeId: string; portId: string } {
  const colonIndex = varId.indexOf(':');
  if (colonIndex === -1) {
    throw new Error(`Invalid variable ID: ${varId}`);
  }
  return {
    nodeId: varId.slice(0, colonIndex),
    portId: varId.slice(colonIndex + 1),
  };
}
