/**
 * 类型颜色缓存
 * 
 * 设计原则：
 * 1. 按需缓存：首次访问时计算并缓存
 * 2. 延迟绑定：通过 setter 注入 getConstraintElements，避免循环依赖
 * 3. 缓存失效：typeConstraintStore 数据变化时清空
 * 
 * 为什么放在 stores 层：
 * - 颜色是 typeConstraint 的派生数据
 * - 需要访问 typeConstraintStore.getConstraintElements
 * - 避免 services 层依赖 stores 层导致循环依赖
 */

import { getTypeColor as computeTypeColor } from '../services/typeColorMapping';

// ============================================================================
// 缓存实现
// ============================================================================

const colorCache = new Map<string, string>();

// 延迟绑定的约束解析函数
let constraintResolver: ((constraint: string) => string[]) | null = null;

/**
 * 设置约束解析函数
 * 
 * 由 typeConstraintStore 在初始化时调用，避免循环依赖
 */
export function setConstraintResolver(resolver: (constraint: string) => string[]): void {
  constraintResolver = resolver;
  // 解析器变化时清空缓存
  colorCache.clear();
}

/**
 * 获取类型颜色（带缓存）
 * 
 * @param typeConstraint - 类型约束名称
 * @returns hex 颜色字符串
 */
export function getTypeColor(typeConstraint: string): string {
  if (!typeConstraint) {
    return '#95A5A6'; // 默认灰色
  }

  // 检查缓存
  const cached = colorCache.get(typeConstraint);
  if (cached) {
    return cached;
  }

  // 如果解析器未设置，返回默认颜色
  if (!constraintResolver) {
    return '#95A5A6';
  }

  // 计算颜色
  const color = computeTypeColor(typeConstraint, constraintResolver);

  // 缓存结果
  colorCache.set(typeConstraint, color);

  return color;
}

/**
 * 清空颜色缓存
 * 
 * 当 typeConstraintStore 数据更新时调用
 */
export function clearColorCache(): void {
  colorCache.clear();
}

/**
 * 获取缓存大小（用于调试）
 */
export function getColorCacheSize(): number {
  return colorCache.size;
}
