/**
 * LayoutConfig 解析模块
 * 从 tokens 模块导入布局配置，保持向后兼容
 */

import type {
  Padding,
  NormalizedPadding,
} from './types';

// 从 tokens 模块导入配置
export { layoutConfig, getContainerConfig } from './tokens';
import { layoutConfig } from './tokens';

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 规范化内边距
 * @param padding - 单值或 [top, right, bottom, left]
 * @returns 规范化的内边距对象
 */
export function normalizePadding(padding: Padding | undefined): NormalizedPadding {
  if (padding === undefined) {
    return { top: 0, right: 0, bottom: 0, left: 0 };
  }
  if (typeof padding === 'number') {
    return { top: padding, right: padding, bottom: padding, left: padding };
  }
  return {
    top: padding[0],
    right: padding[1],
    bottom: padding[2],
    left: padding[3],
  };
}

/**
 * 格式化内边距为 CSS 字符串
 * @param padding - 内边距
 * @returns CSS padding 字符串
 */
export function formatPadding(padding: Padding | undefined): string {
  if (padding === undefined) return '0';
  if (typeof padding === 'number') return `${padding}px`;
  return padding.map((p) => `${p}px`).join(' ');
}

// ============================================================================
// 颜色工具函数
// ============================================================================

/**
 * 获取方言颜色
 * @param dialectName - 方言名称（如 'arith', 'func', 'scf'）
 * @returns hex 颜色字符串
 */
export function getDialectColor(dialectName: string): string {
  return layoutConfig.dialect[dialectName] ?? layoutConfig.dialect.default;
}

/**
 * 获取节点类型颜色
 * @param type - 节点类型
 * @returns hex 颜色字符串
 */
export function getNodeTypeColor(type: 'entry' | 'return' | 'call' | 'operation'): string {
  return layoutConfig.nodeType[type];
}

/**
 * 获取类型颜色
 * 
 * 优先级：
 * 1. 精确匹配 layoutConfig.type[typeConstraint]
 * 2. 类型类别匹配（整数、浮点等）
 * 3. 默认颜色
 * 
 * @param typeConstraint - 类型约束名称（如 'I32', 'F32', 'AnyType'）
 * @returns hex 颜色字符串
 */
export function getTypeColor(typeConstraint: string): string {
  if (!typeConstraint) return layoutConfig.type.default;
  
  // 1. 精确匹配
  if (layoutConfig.type[typeConstraint]) {
    return layoutConfig.type[typeConstraint];
  }
  
  // 2. 类型类别匹配
  const upperType = typeConstraint.toUpperCase();
  
  // 无符号整数
  if (upperType.startsWith('UI') || upperType.includes('UNSIGNED')) {
    return layoutConfig.type.unsignedInteger;
  }
  
  // 有符号整数
  if (upperType.startsWith('SI') || upperType.includes('SIGNED')) {
    return layoutConfig.type.signedInteger;
  }
  
  // 无符号整数（I1, I8, I16, I32, I64, I128, Index）
  if (/^I\d+$/.test(typeConstraint) || upperType === 'INDEX' || upperType.includes('SIGNLESSINTEGER')) {
    return layoutConfig.type.signlessInteger;
  }
  
  // 浮点数
  if (/^(F|BF)\d+/.test(typeConstraint) || upperType.includes('FLOAT')) {
    return layoutConfig.type.float;
  }
  
  // Tensor 浮点
  if (upperType.includes('TENSOR') && upperType.includes('FLOAT')) {
    return layoutConfig.type.tensorFloat;
  }
  
  // 3. 默认颜色
  return layoutConfig.type.default;
}
