/**
 * 几何工具函数
 * 
 * 用于边路径计算和命中测试。
 * 纯函数，无业务逻辑依赖。
 */

import { tokens } from '../adapters/shared/styles';

const BEZIER_OFFSET = typeof tokens.edge.bezierOffset === 'string' 
  ? parseInt(tokens.edge.bezierOffset) 
  : tokens.edge.bezierOffset;

/**
 * 计算边路径点（贝塞尔曲线）
 * @param sourceX - 源点 X
 * @param sourceY - 源点 Y
 * @param targetX - 目标点 X
 * @param targetY - 目标点 Y
 * @returns 路径点数组（4 个点：起点、控制点1、控制点2、终点）
 */
export function computeEdgePath(
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number
): Array<{ x: number; y: number }> {
  const dx = Math.abs(targetX - sourceX);
  const offset = Math.min(BEZIER_OFFSET, dx * 0.5);
  
  return [
    { x: sourceX, y: sourceY },
    { x: sourceX + offset, y: sourceY },
    { x: targetX - offset, y: targetY },
    { x: targetX, y: targetY },
  ];
}

/**
 * 计算点到贝塞尔曲线的最近距离（用于命中测试）
 * 使用采样近似法
 */
export function distanceToEdge(
  px: number,
  py: number,
  points: Array<{ x: number; y: number }>
): number {
  if (points.length !== 4) return Infinity;
  
  const [p0, p1, p2, p3] = points;
  let minDist = Infinity;
  
  const samples = 20;
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const t2 = t * t;
    const t3 = t2 * t;
    const mt = 1 - t;
    const mt2 = mt * mt;
    const mt3 = mt2 * mt;
    
    // 三次贝塞尔曲线公式
    const x = mt3 * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t3 * p3.x;
    const y = mt3 * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t3 * p3.y;
    
    const dist = Math.sqrt((px - x) ** 2 + (py - y) ** 2);
    minDist = Math.min(minDist, dist);
  }
  
  return minDist;
}

/**
 * 检查点是否在矩形内
 */
export function isPointInRect(
  px: number,
  py: number,
  rx: number,
  ry: number,
  rw: number,
  rh: number
): boolean {
  return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh;
}

/**
 * 检查点是否在圆形内
 */
export function isPointInCircle(
  px: number,
  py: number,
  cx: number,
  cy: number,
  radius: number
): boolean {
  const dx = px - cx;
  const dy = py - cy;
  return dx * dx + dy * dy <= radius * radius;
}
