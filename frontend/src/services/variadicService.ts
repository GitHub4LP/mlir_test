/**
 * Variadic 端口管理服务
 * 
 * 提供 variadic 端口数量计算的纯函数，供所有渲染器复用。
 */

/**
 * 计算新的 variadic 端口数量
 * 
 * @param currentCounts 当前的 variadic 端口数量
 * @param groupName variadic 组名
 * @param delta 变化量（+1 或 -1）
 * @param minCount 最小数量（默认 0）
 * @returns 新的 variadic 端口数量
 */
export function computeVariadicCounts(
  currentCounts: Record<string, number>,
  groupName: string,
  delta: number,
  minCount: number = 0
): Record<string, number> {
  const currentCount = currentCounts[groupName] ?? 1;
  const newCount = Math.max(minCount, currentCount + delta);

  return {
    ...currentCounts,
    [groupName]: newCount,
  };
}

/**
 * 增加 variadic 端口数量
 */
export function incrementVariadicCount(
  currentCounts: Record<string, number>,
  groupName: string
): Record<string, number> {
  return computeVariadicCounts(currentCounts, groupName, 1);
}

/**
 * 减少 variadic 端口数量
 */
export function decrementVariadicCount(
  currentCounts: Record<string, number>,
  groupName: string,
  minCount: number = 0
): Record<string, number> {
  return computeVariadicCounts(currentCounts, groupName, -1, minCount);
}

/**
 * 获取 variadic 组的当前数量
 */
export function getVariadicCount(
  counts: Record<string, number>,
  groupName: string,
  defaultCount: number = 1
): number {
  return counts[groupName] ?? defaultCount;
}
