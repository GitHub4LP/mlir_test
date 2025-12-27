/**
 * 连线验证适配器
 * 
 * 为不同渲染框架提供连线验证的薄包装。
 * 
 * 职责划分：
 * - 数据层 (services/typeSystem.ts)：类型兼容性计算
 * - 数据层 (services/port.ts)：端口类型检测
 * - 适配层 (本模块)：框架特定的回调格式转换
 * 
 * 设计原则：
 * - 本模块不包含业务逻辑，只做格式转换
 * - 核心逻辑委托给 typeSystem 和 port 模块
 */

import { PortRef, PortKindUtils } from '../../../services/port';
import { getTypeIntersectionCount } from '../../../services/typeSystem';

// ============================================================
// 类型定义
// ============================================================

/** 端口类型获取函数 */
export type GetPortTypeFn = (nodeId: string, portId: string) => string | null;

/** 连接验证结果 */
export interface ConnectionValidationResult {
  isValid: boolean;
  intersectionCount: number;
  errorMessage?: string;
}

// ============================================================
// 核心验证函数（委托给数据层）
// ============================================================

/**
 * 验证两个端口是否可以连接
 * 
 * @param sourceNodeId 源节点 ID
 * @param sourcePortId 源端口 ID
 * @param targetNodeId 目标节点 ID
 * @param targetPortId 目标端口 ID
 * @param getPortType 获取端口类型的函数
 * @returns 验证结果
 */
export function validatePorts(
  sourceNodeId: string,
  sourcePortId: string,
  targetNodeId: string,
  targetPortId: string,
  getPortType: GetPortTypeFn
): ConnectionValidationResult {
  // 1. 防止自连接
  if (sourceNodeId === targetNodeId) {
    return { isValid: false, intersectionCount: 0, errorMessage: '不能连接到自身' };
  }
  
  // 2. 解析端口类型
  const sourceParsed = PortRef.parseHandleId(sourcePortId);
  const targetParsed = PortRef.parseHandleId(targetPortId);
  
  if (!sourceParsed || !targetParsed) {
    return { isValid: false, intersectionCount: 0, errorMessage: '无效的端口 ID' };
  }
  
  const sourceIsExec = PortKindUtils.isExec(sourceParsed.kind);
  const targetIsExec = PortKindUtils.isExec(targetParsed.kind);
  
  // 3. 执行引脚和数据引脚不能混连
  if (sourceIsExec !== targetIsExec) {
    return { isValid: false, intersectionCount: 0, errorMessage: '不能将执行引脚连接到数据引脚' };
  }
  
  // 4. 执行引脚不需要类型检查
  if (sourceIsExec && targetIsExec) {
    return { isValid: true, intersectionCount: 1 };
  }
  
  // 5. 数据引脚：获取类型并检查兼容性
  const sourceType = getPortType(sourceNodeId, sourcePortId);
  const targetType = getPortType(targetNodeId, targetPortId);
  
  // 无法获取类型时，允许连接（宽松模式）
  if (!sourceType || !targetType) {
    return { isValid: true, intersectionCount: 1 };
  }
  
  // 6. 使用 typeSystem 计算类型交集
  const intersectionCount = getTypeIntersectionCount(sourceType, targetType);
  
  if (intersectionCount === 0) {
    return {
      isValid: false,
      intersectionCount: 0,
      errorMessage: `类型不兼容: '${sourceType}' 与 '${targetType}' 没有交集`,
    };
  }
  
  return { isValid: true, intersectionCount };
}

// ============================================================
// 框架适配器
// ============================================================

/**
 * 创建 React Flow 连接验证器
 * 
 * @param getPortType 获取端口类型的函数
 * @returns React Flow isValidConnection 回调
 */
export function createReactFlowValidator(
  getPortType: GetPortTypeFn
): (connection: { source: string | null; target: string | null; sourceHandle: string | null; targetHandle: string | null }) => boolean {
  return (connection) => {
    const { source, target, sourceHandle, targetHandle } = connection;
    
    if (!source || !target || !sourceHandle || !targetHandle) {
      return false;
    }
    
    const result = validatePorts(source, sourceHandle, target, targetHandle, getPortType);
    return result.isValid;
  };
}

/**
 * 创建 Vue Flow 连接验证器
 * 
 * @param getPortType 获取端口类型的函数
 * @returns Vue Flow isValidConnection 回调
 */
export function createVueFlowValidator(
  getPortType: GetPortTypeFn
): (connection: { source: string; target: string; sourceHandle?: string | null; targetHandle?: string | null }) => boolean {
  return (connection) => {
    const { source, target, sourceHandle, targetHandle } = connection;
    
    if (!sourceHandle || !targetHandle) {
      return false;
    }
    
    const result = validatePorts(source, sourceHandle, target, targetHandle, getPortType);
    return result.isValid;
  };
}

/**
 * 创建使用 store 的连接验证器
 * 
 * 便捷函数，自动从 typeConstraintStore 获取 getConstraintElements
 * 
 * @param getPortType 获取端口类型的函数
 * @returns 验证函数
 */
export function createValidatorWithStore(
  getPortType: GetPortTypeFn
): (sourceNodeId: string, sourcePortId: string, targetNodeId: string, targetPortId: string) => ConnectionValidationResult {
  return (sourceNodeId, sourcePortId, targetNodeId, targetPortId) => {
    return validatePorts(sourceNodeId, sourcePortId, targetNodeId, targetPortId, getPortType);
  };
}

// ============================================================
// 端口类型检测（重导出，方便使用）
// ============================================================

/**
 * 检查端口 ID 是否是执行引脚
 */
export function isExecPort(portId: string): boolean {
  const parsed = PortRef.parseHandleId(portId);
  return parsed !== null && PortKindUtils.isExec(parsed.kind);
}

/**
 * 检查端口 ID 是否是数据引脚
 */
export function isDataPort(portId: string): boolean {
  const parsed = PortRef.parseHandleId(portId);
  return parsed !== null && PortKindUtils.isData(parsed.kind);
}
