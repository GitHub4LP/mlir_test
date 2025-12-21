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
import { getConstraintElements, getTypeIntersectionCount } from '../../../services/typeSystem';
import { PortRef, PortKind, PortKindUtils } from '../../../services/port';

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
 * 从 React Flow 节点获取端口类型约束
 * 
 * @param node - React Flow 节点
 * @param handleId - 端口 ID
 * @param resolvedTypes - 已解析的具体类型映射
 * @returns 类型约束字符串，或 null
 */
export function getPortTypeConstraint(
  node: Node,
  handleId: string,
  resolvedTypes?: Map<string, Map<string, string>>
): string | null {
  // 优先使用已解析的类型
  if (resolvedTypes) {
    const nodeTypes = resolvedTypes.get(node.id);
    if (nodeTypes) {
      const resolvedType = nodeTypes.get(handleId);
      if (resolvedType) {
        return resolvedType;
      }
    }
  }

  // 解析端口 ID
  const parsed = PortRef.parseHandleId(handleId);
  if (!parsed) return null;

  // 根据节点类型获取端口类型
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
    const data = node.data as { inputTypes?: Record<string, string>; outputTypes?: Record<string, string> };
    const portName = parsed.name.replace(/_\d+$/, '');  // 移除 variadic 索引

    if (parsed.kind === PortKind.DataOut) {
      return data.outputTypes?.[portName] || null;
    } else if (parsed.kind === PortKind.DataIn) {
      return data.inputTypes?.[portName] || null;
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
 * @param resolvedTypes - 已解析的具体类型映射
 * @param existingEdges - 现有连线（用于计数验证）
 * @returns 验证结果
 */
export function validateConnection(
  connection: Connection,
  nodes: Node[],
  resolvedTypes?: Map<string, Map<string, string>>,
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

  // 8. 获取端口类型
  const sourceType = getPortTypeConstraint(sourceNode, sourceHandle, resolvedTypes);
  const targetType = getPortTypeConstraint(targetNode, targetHandle, resolvedTypes);

  // 无法获取类型时，允许连接（宽松模式）
  if (!sourceType || !targetType) {
    return {
      isValid: true,
      sourceType: sourceType || 'unknown',
      targetType: targetType || 'unknown',
      intersectionCount: 1,
    };
  }

  // 9. 使用 typeSystem 计算类型交集
  const intersectionCount = getTypeIntersectionCount(sourceType, targetType);

  if (intersectionCount > 0) {
    return {
      isValid: true,
      sourceType,
      targetType,
      intersectionCount,
    };
  }

  // 10. 生成错误信息
  const sourceElements = getConstraintElements(sourceType);
  const targetElements = getConstraintElements(targetType);

  let errorMessage = `类型不兼容: '${sourceType}' 与 '${targetType}' 没有交集`;

  if (sourceElements.length > 1 || targetElements.length > 1) {
    const sourceTypesStr = sourceElements.length > 3
      ? `${sourceElements.slice(0, 3).join(', ')}...`
      : sourceElements.join(', ');
    const targetTypesStr = targetElements.length > 3
      ? `${targetElements.slice(0, 3).join(', ')}...`
      : targetElements.join(', ');

    errorMessage += `。源类型 [${sourceTypesStr}]，目标类型 [${targetTypesStr}]`;
  }

  return {
    isValid: false,
    errorMessage,
    sourceType,
    targetType,
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
  nodes: Node[],
  resolvedTypes?: Map<string, Map<string, string>>
): (connection: Connection) => boolean {
  return (connection: Connection) => {
    const result = validateConnection(connection, nodes, resolvedTypes);
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
