/**
 * GPU 渲染共用工具函数
 * WebGL 和 WebGPU 共用
 */

import type { RGBA, Point } from './types';

/**
 * 解析颜色字符串为 RGBA（0-1 范围）
 * 支持格式：#RRGGBB, #RRGGBBAA, rgb(r,g,b), rgba(r,g,b,a), transparent
 */
export function parseColor(color: string): RGBA {
  if (!color || color === 'transparent') {
    return { r: 0, g: 0, b: 0, a: 0 };
  }

  // #RRGGBB 或 #RRGGBBAA
  if (color.startsWith('#')) {
    const hex = color.slice(1);
    return {
      r: parseInt(hex.slice(0, 2), 16) / 255,
      g: parseInt(hex.slice(2, 4), 16) / 255,
      b: parseInt(hex.slice(4, 6), 16) / 255,
      a: hex.length > 6 ? parseInt(hex.slice(6, 8), 16) / 255 : 1,
    };
  }

  // rgba(r, g, b, a) 或 rgb(r, g, b)
  const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
  if (match) {
    return {
      r: parseInt(match[1]) / 255,
      g: parseInt(match[2]) / 255,
      b: parseInt(match[3]) / 255,
      a: match[4] ? parseFloat(match[4]) : 1,
    };
  }

  return { r: 1, g: 1, b: 1, a: 1 };
}

/**
 * 计算三次贝塞尔曲线上的点
 * @param p0 起点
 * @param p1 控制点1
 * @param p2 控制点2
 * @param p3 终点
 * @param t 参数 [0, 1]
 */
export function cubicBezier(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
  const mt = 1 - t;
  const mt2 = mt * mt;
  const mt3 = mt2 * mt;
  const t2 = t * t;
  const t3 = t2 * t;

  return {
    x: mt3 * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t3 * p3.x,
    y: mt3 * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t3 * p3.y,
  };
}

/** 单位正方形顶点 [-1, 1] */
export const QUAD_VERTICES_CENTERED = new Float32Array([
  -1, -1,
   1, -1,
   1,  1,
  -1,  1,
]);

/** 单位正方形顶点 [0, 1] */
export const QUAD_VERTICES_UNIT = new Float32Array([
  0, 0,
  1, 0,
  0, 1,
  1, 1,
]);

/** 三角形索引（两个三角形组成四边形） */
export const QUAD_INDICES = new Uint16Array([0, 1, 2, 0, 2, 3]);

/** 三角形带索引 */
export const QUAD_STRIP_INDICES = new Uint16Array([0, 1, 3, 2]);

/** 贝塞尔曲线细分段数 */
export const BEZIER_SEGMENTS = 32;

/**
 * 生成贝塞尔曲线 t 参数数组
 */
export function generateBezierTValues(segments: number = BEZIER_SEGMENTS): Float32Array {
  const values = new Float32Array(segments + 1);
  for (let i = 0; i <= segments; i++) {
    values[i] = i / segments;
  }
  return values;
}


// ============================================================
// 实例数据构建函数
// ============================================================

import type { RenderRect, RenderCircle, RenderTriangle, RenderPath } from '../../../core/RenderData';

/** 节点实例数据布局（每实例 floats 数量） */
export const NODE_INSTANCE_FLOATS = 16;

/** 边实例数据布局（每实例 floats 数量） */
export const EDGE_INSTANCE_FLOATS = 14;

/** 圆形实例数据布局（每实例 floats 数量） */
export const CIRCLE_INSTANCE_FLOATS = 12;

/** 三角形实例数据布局（每实例 floats 数量） */
export const TRIANGLE_INSTANCE_FLOATS = 13;

/**
 * 构建矩形实例数据
 * 布局: position(2), size(2), headerHeight(1), borderRadius(1), bodyColor(4), headerColor(4), selected(1), padding(1)
 */
export function buildRectInstanceData(rects: RenderRect[]): Float32Array {
  const data = new Float32Array(rects.length * NODE_INSTANCE_FLOATS);
  
  for (let i = 0; i < rects.length; i++) {
    const rect = rects[i];
    const offset = i * NODE_INSTANCE_FLOATS;
    const color = parseColor(rect.fillColor);
    const borderColor = parseColor(rect.borderColor);
    const radius = typeof rect.borderRadius === 'number' ? rect.borderRadius : rect.borderRadius.topLeft;
    
    data[offset + 0] = rect.x;
    data[offset + 1] = rect.y;
    data[offset + 2] = rect.width;
    data[offset + 3] = rect.height;
    data[offset + 4] = 0; // headerHeight (not used for generic rects)
    data[offset + 5] = radius;
    data[offset + 6] = color.r;
    data[offset + 7] = color.g;
    data[offset + 8] = color.b;
    data[offset + 9] = color.a;
    data[offset + 10] = borderColor.r;
    data[offset + 11] = borderColor.g;
    data[offset + 12] = borderColor.b;
    data[offset + 13] = borderColor.a;
    data[offset + 14] = rect.selected ? 1 : 0;
    data[offset + 15] = 0; // padding
  }
  
  return data;
}

/**
 * 构建圆形实例数据
 * 布局: position(2), radius(1), fillColor(4), borderColor(4), borderWidth(1)
 */
export function buildCircleInstanceData(circles: RenderCircle[]): Float32Array {
  const data = new Float32Array(circles.length * CIRCLE_INSTANCE_FLOATS);
  
  for (let i = 0; i < circles.length; i++) {
    const circle = circles[i];
    const offset = i * CIRCLE_INSTANCE_FLOATS;
    const fillColor = parseColor(circle.fillColor);
    const borderColor = parseColor(circle.borderColor);
    
    data[offset + 0] = circle.x;
    data[offset + 1] = circle.y;
    data[offset + 2] = circle.radius;
    data[offset + 3] = fillColor.r;
    data[offset + 4] = fillColor.g;
    data[offset + 5] = fillColor.b;
    data[offset + 6] = fillColor.a;
    data[offset + 7] = borderColor.r;
    data[offset + 8] = borderColor.g;
    data[offset + 9] = borderColor.b;
    data[offset + 10] = borderColor.a;
    data[offset + 11] = circle.borderWidth;
  }
  
  return data;
}

/**
 * 构建三角形实例数据
 * 布局: position(2), size(1), direction(1), fillColor(4), borderColor(4), borderWidth(1)
 */
export function buildTriangleInstanceData(triangles: RenderTriangle[]): Float32Array {
  const data = new Float32Array(triangles.length * TRIANGLE_INSTANCE_FLOATS);
  
  for (let i = 0; i < triangles.length; i++) {
    const tri = triangles[i];
    const offset = i * TRIANGLE_INSTANCE_FLOATS;
    const fillColor = parseColor(tri.fillColor);
    const borderColor = parseColor(tri.borderColor);
    
    data[offset + 0] = tri.x;
    data[offset + 1] = tri.y;
    data[offset + 2] = tri.size;
    data[offset + 3] = tri.direction === 'right' ? 1 : -1;
    data[offset + 4] = fillColor.r;
    data[offset + 5] = fillColor.g;
    data[offset + 6] = fillColor.b;
    data[offset + 7] = fillColor.a;
    data[offset + 8] = borderColor.r;
    data[offset + 9] = borderColor.g;
    data[offset + 10] = borderColor.b;
    data[offset + 11] = borderColor.a;
    data[offset + 12] = tri.borderWidth;
  }
  
  return data;
}

/**
 * 构建边实例数据（贝塞尔曲线）
 * 布局: start(2), end(2), control1(2), control2(2), color(4), width(1), selected(1)
 */
export function buildEdgeInstanceData(paths: RenderPath[]): Float32Array {
  const data = new Float32Array(paths.length * EDGE_INSTANCE_FLOATS);
  
  for (let i = 0; i < paths.length; i++) {
    const path = paths[i];
    const offset = i * EDGE_INSTANCE_FLOATS;
    const color = parseColor(path.color);
    
    // 假设 points 包含 4 个点：起点、控制点1、控制点2、终点
    const p0 = path.points[0] || { x: 0, y: 0 };
    const p1 = path.points[1] || p0;
    const p2 = path.points[2] || p1;
    const p3 = path.points[3] || p2;
    
    data[offset + 0] = p0.x;
    data[offset + 1] = p0.y;
    data[offset + 2] = p3.x;
    data[offset + 3] = p3.y;
    data[offset + 4] = p1.x;
    data[offset + 5] = p1.y;
    data[offset + 6] = p2.x;
    data[offset + 7] = p2.y;
    data[offset + 8] = color.r;
    data[offset + 9] = color.g;
    data[offset + 10] = color.b;
    data[offset + 11] = color.a;
    data[offset + 12] = path.width;
    data[offset + 13] = 0; // selected (not in RenderPath, default to 0)
  }
  
  return data;
}
