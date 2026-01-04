/**
 * React Flow 连接验证工具
 * 
 * React Flow 特定的连接验证逻辑。
 * 
 * 职责划分：
 * - 数据层 (services/typeSystem.ts)：类型兼容性计算
 * - 数据层 (services/port.ts)：端口类型检测
 * - 本模块：React Flow 特定的节点数据访问和验证
 */

import type { Node, Connection } from '@xyflow/react';
import type { BlueprintNodeData, FunctionEntryData, FunctionReturnData, FunctionCallData, PortConfig } from '../../../types';
import { getConstraintElements } from '../../../services/typeSystem';
import { PortRef, PortKind, PortKindUtils } from '../../../services/port';
import { computeTypeIntersection } from '../../../services/typeIntersection';

// ============================================================
// 类型定义
// ============================================================

/**
 * 连接验证结果
 */
export interface ConnectionValidationResult {
  /** 是否有效 */
  isValid: boolean;
  /** 错误信息（如果无效） */
  errorMessage?: string;
  /** 源端口类型 */
  sourceType?: string;
  /** 目标端口类型 */
  targetType?: string;
  /** 类型交集大小 */
  intersectionCount?: number;
}

// ============================================================
// 端口类型获取（React Flow 特定）
// ============================================================

/**
 * 从 React Flow 节点获取端口的有效集合
 * 
 * 有效集合是传播后求交集的结果，存储在 inputTypes/outputTypes 中。
 * 连线验证应该直接使用有效集合计算交集。
 * 
 * @param node - React Flow 节点
 * @param handleId - 端口 ID
 * @returns 有效集合（具体类型数组），或 null
 */
export function getPortEffectiveSet(
  node: Node,
  handleId: string
): string[] | null {
  // 解析端口 ID
  const parsed = PortRef.parseHandleId(handleId);
  if (!parsed) return null;

  const portName = parsed.name.replace(/_\d+$/, '');  // 移除 variadic 索引

  // 根据节点类型获取有效集合
  if (node.type === 'function-entry') {
    const data = node.data as FunctionEntryData & { outputTypes?: Record<string, string[]> };
    if (parsed.kind === PortKind.DataOut) {
      // 优先使用有效集合
      if (data.outputTypes?.[portName]) {
        return data.outputTypes[portName];
      }
      // 回退：从端口定义获取原始约束，返回 null 让调用方展开
      return null;
    }
  } else if (node.type === 'function-return') {
    const data = node.data as FunctionReturnData & { inputTypes?: Record<string, string[]> };
    if (parsed.kind === PortKind.DataIn) {
      if (data.inputTypes?.[portName]) {
        return data.inputTypes[portName];
      }
      return null;
    }
  } else if (node.type === 'function-call') {
    const data = node.data as FunctionCallData & { inputTypes?: Record<string, string[]>; outputTypes?: Record<string, string[]> };
    if (parsed.kind === PortKind.DataIn) {
      if (data.inputTypes?.[portName]) {
        return data.inputTypes[portName];
      }
      return null;
    } else if (parsed.kind === PortKind.DataOut) {
      if (data.outputTypes?.[portName]) {
        return data.outputTypes[portName];
      }
      return null;
    }
  } else {
    // Operation 节点
    const data = node.data as { 
      inputTypes?: Record<string, string[]>; 
      outputTypes?: Record<string, string[]>;
    };

    if (parsed.kind === PortKind.DataOut) {
      if (data.outputTypes?.[portName]) {
        return data.outputTypes[portName];
      }
      return null;
    } else if (parsed.kind === PortKind.DataIn) {
      if (data.inputTypes?.[portName]) {
        return data.inputTypes[portName];
      }
      return null;
    }
  }

  return null;
}

/**
 * 从 React Flow 节点获取端口的原始约束
 * 
 * 当有效集合不存在时（节点刚创建），使用原始约束。
 * 
 * @param node - React Flow 节点
 * @param handleId - 端口 ID
 * @returns 原始约束名，或 null
 */
export function getPortOriginalConstraint(
  node: Node,
  handleId: string
): string | null {
  // 解析端口 ID
  const parsed = PortRef.parseHandleId(handleId);
  if (!parsed) return null;

  const portName = parsed.name.replace(/_\d+$/, '');  // 移除 variadic 索引

  // 根据节点类型获取原始约束
  if (node.type === 'function-entry') {
    const data = node.data as FunctionEntryData;
    if (parsed.kind === PortKind.DataOut && Array.isArray(data.outputs)) {
      const port = data.outputs.find((p: PortConfig) => p.id === handleId);
      return port?.typeConstraint || null;
    }
  } else if (node.type === 'function-return') {
    const data = node.data as FunctionReturnData;
    if (parsed.kind === PortKind.DataIn && Array.isArray(data.inputs)) {
      const port = data.inputs.find((p: PortConfig) => p.id === handleId);
      return port?.typeConstraint || null;
    }
  } else if (node.type === 'function-call') {
    const data = node.data as FunctionCallData;
    if (parsed.kind === PortKind.DataIn && Array.isArray(data.inputs)) {
      const port = data.inputs.find((p: PortConfig) => p.id === handleId);
      return port?.typeConstraint || null;
    } else if (parsed.kind === PortKind.DataOut && Array.isArray(data.outputs)) {
      const port = data.outputs.find((p: PortConfig) => p.id === handleId);
      return port?.typeConstraint || null;
    }
  } else {
    // Operation 节点
    const data = node.data as { 
      operation?: { arguments: { name: string; typeConstraint: string; kind: string }[]; results: { name: string; typeConstraint: string }[] };
    };

    if (data.operation) {
      if (parsed.kind === PortKind.DataOut) {
        const result = data.operation.results.find(r => r.name === portName);
        return result?.typeConstraint || null;
      } else if (parsed.kind === PortKind.DataIn) {
        const arg = data.operation.arguments.find(a => a.name === portName && a.kind === 'operand');
        return arg?.typeConstraint || null;
      }
    }
  }

  return null;
}

// ============================================================
// 连接计数验证
// ============================================================

/**
 * 验证连接计数约束
 * 
 * 规则：
 * - 执行输出 (exec-out)：最多 1 个连接
 * - 执行输入 (exec-in)：无限制
 * - 数据输出：无限制
 * - 数据输入：最多 1 个连接
 */
export function validateConnectionCount(
  connection: Connection,
  existingEdges: { source: string; sourceHandle: string | null; target: string; targetHandle: string | null }[]
): { isValid: boolean; errorMessage?: string } {
  const { source, sourceHandle, target, targetHandle } = connection;

  if (!sourceHandle || !targetHandle) {
    return { isValid: true };
  }

  const sourceParsed = PortRef.parseHandleId(sourceHandle);
  const targetParsed = PortRef.parseHandleId(targetHandle);
  
  if (!sourceParsed || !targetParsed) {
    return { isValid: true };
  }

  const isSourceExec = PortKindUtils.isExec(sourceParsed.kind);
  const isTargetExec = PortKindUtils.isExec(targetParsed.kind);

  // 执行输出只能有 1 个连接
  if (isSourceExec) {
    const existingFromSource = existingEdges.filter(
      e => e.source === source && e.sourceHandle === sourceHandle
    );
    if (existingFromSource.length >= 1) {
      return {
        isValid: false,
        errorMessage: '执行输出只能有一个连接',
      };
    }
  }

  // 数据输入只能有 1 个连接
  if (!isTargetExec) {
    const existingToTarget = existingEdges.filter(
      e => e.target === target && e.targetHandle === targetHandle
    );
    if (existingToTarget.length >= 1) {
      return {
        isValid: false,
        errorMessage: '数据输入只能有一个连接',
      };
    }
  }

  return { isValid: true };
}

// ============================================================
// 完整连接验证
// ============================================================

/**
 * 验证 React Flow 连接
 * 
 * @param connection - React Flow 连接对象
 * @param nodes - 所有节点
 * @param existingEdges - 现有连线（用于计数验证）
 * @returns 验证结果
 */
export function validateConnection(
  connection: Connection,
  nodes: Node[],
  existingEdges?: { source: string; sourceHandle: string | null; target: string; targetHandle: string | null }[]
): ConnectionValidationResult {
  const { source, sourceHandle, target, targetHandle } = connection;

  // 1. 基本验证
  if (!source || !sourceHandle || !target || !targetHandle) {
    return {
      isValid: false,
      errorMessage: '连接信息不完整',
    };
  }

  // 2. 防止自连接
  if (source === target) {
    return {
      isValid: false,
      errorMessage: '不能连接到自身',
    };
  }

  // 3. 连接计数验证
  if (existingEdges) {
    const countResult = validateConnectionCount(connection, existingEdges);
    if (!countResult.isValid) {
      return {
        isValid: false,
        errorMessage: countResult.errorMessage,
      };
    }
  }

  // 4. 解析端口类型
  const sourceParsed = PortRef.parseHandleId(sourceHandle);
  const targetParsed = PortRef.parseHandleId(targetHandle);
  
  if (!sourceParsed || !targetParsed) {
    return {
      isValid: false,
      errorMessage: '无效的端口 ID',
    };
  }

  const isSourceExec = PortKindUtils.isExec(sourceParsed.kind);
  const isTargetExec = PortKindUtils.isExec(targetParsed.kind);

  // 5. 执行引脚和数据引脚不能混连
  if (isSourceExec !== isTargetExec) {
    return {
      isValid: false,
      errorMessage: '不能将执行引脚连接到数据引脚',
    };
  }

  // 6. 执行引脚不需要类型检查
  if (isSourceExec && isTargetExec) {
    return {
      isValid: true,
      sourceType: 'exec',
      targetType: 'exec',
      intersectionCount: 1,
    };
  }

  // 7. 查找节点
  const sourceNode = nodes.find(n => n.id === source);
  const targetNode = nodes.find(n => n.id === target);

  if (!sourceNode) {
    return { isValid: false, errorMessage: '源节点不存在' };
  }
  if (!targetNode) {
    return { isValid: false, errorMessage: '目标节点不存在' };
  }

  // 8. 获取端口的有效集合（优先）或原始约束
  // 有效集合是传播后求交集的结果，直接用于连线验证
  let sourceSet = getPortEffectiveSet(sourceNode, sourceHandle);
  let targetSet = getPortEffectiveSet(targetNode, targetHandle);
  
  // 如果有效集合不存在（节点刚创建），从原始约束展开
  if (!sourceSet) {
    const sourceConstraint = getPortOriginalConstraint(sourceNode, sourceHandle);
    if (sourceConstraint) {
      sourceSet = getConstraintElements(sourceConstraint);
    }
  }
  if (!targetSet) {
    const targetConstraint = getPortOriginalConstraint(targetNode, targetHandle);
    if (targetConstraint) {
      targetSet = getConstraintElements(targetConstraint);
    }
  }

  // 无法获取类型时，允许连接（宽松模式）
  if (!sourceSet || sourceSet.length === 0 || !targetSet || targetSet.length === 0) {
    return {
      isValid: true,
      sourceType: 'unknown',
      targetType: 'unknown',
      intersectionCount: 1,
    };
  }

  // 9. 计算两个有效集合的交集（支持容器类型与约束的交集）
  // 使用 computeTypeIntersection 处理容器类型（如 memref<4xI32>）与约束（如 AnyType）的交集
  let hasIntersection = false;
  
  for (const srcType of sourceSet) {
    for (const tgtType of targetSet) {
      // 快速路径：完全相同
      if (srcType === tgtType) {
        hasIntersection = true;
        break;
      }
      // 使用类型交集计算（处理容器类型与约束）
      const intersection = computeTypeIntersection(srcType, tgtType);
      if (intersection !== null) {
        hasIntersection = true;
        break;
      }
    }
    if (hasIntersection) break;
  }

  if (hasIntersection) {
    return {
      isValid: true,
      sourceType: sourceSet.length === 1 ? sourceSet[0] : `${sourceSet.length} types`,
      targetType: targetSet.length === 1 ? targetSet[0] : `${targetSet.length} types`,
      intersectionCount: 1,  // 简化：只要有交集就返回 1
    };
  }

  // 10. 生成错误信息
  const sourceTypesStr = sourceSet.length > 3
    ? `${sourceSet.slice(0, 3).join(', ')}...`
    : sourceSet.join(', ');
  const targetTypesStr = targetSet.length > 3
    ? `${targetSet.slice(0, 3).join(', ')}...`
    : targetSet.join(', ');

  const errorMessage = `类型不兼容: [${sourceTypesStr}] 与 [${targetTypesStr}] 没有交集`;

  return {
    isValid: false,
    errorMessage,
    sourceType: sourceSet.length === 1 ? sourceSet[0] : `${sourceSet.length} types`,
    targetType: targetSet.length === 1 ? targetSet[0] : `${targetSet.length} types`,
    intersectionCount: 0,
  };
}

// ============================================================
// 验证器工厂
// ============================================================

/**
 * 创建 React Flow isValidConnection 回调
 */
export function createConnectionValidator(
  nodes: Node[]
): (connection: Connection) => boolean {
  return (connection: Connection) => {
    const result = validateConnection(connection, nodes);
    return result.isValid;
  };
}

// ============================================================
// 节点错误检查
// ============================================================

/**
 * 获取节点的错误列表
 * 
 * 检查未连接的必需端口
 */
export function getNodeErrors(
  node: Node,
  edges: { source: string; sourceHandle: string; target: string; targetHandle: string }[]
): string[] {
  const errors: string[] = [];

  if (node.type !== 'operation') {
    return errors;
  }

  const data = node.data as BlueprintNodeData;
  const operation = data.operation;

  // 检查每个必需的输入端口
  for (const arg of operation.arguments) {
    if (arg.kind === 'operand' && !arg.isOptional) {
      const portId = `${PortKind.DataIn}-${arg.name}`;

      const isConnected = edges.some(
        e => e.target === node.id && e.targetHandle === portId
      );

      if (!isConnected) {
        errors.push(`必需输入 '${arg.name}' 未连接`);
      }
    }
  }

  return errors;
}
