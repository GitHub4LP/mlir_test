/**
 * 边批次管理器
 * 
 * 将 RenderPath 数据转换为 GPU 实例数据。
 * 支持贝塞尔曲线控制点计算。
 */

import type { EdgeBatch } from '../backends/IGPUBackend';
import type { RenderPath } from '../../canvas/types';

/** 边实例数据布局（每实例 floats 数量） */
const EDGE_INSTANCE_FLOATS = 14;

/**
 * 解析 CSS 颜色为 RGBA 数组
 */
function parseColor(color: string): [number, number, number, number] {
  if (color.startsWith('#')) {
    const hex = color.slice(1);
    if (hex.length === 3) {
      const r = parseInt(hex[0] + hex[0], 16) / 255;
      const g = parseInt(hex[1] + hex[1], 16) / 255;
      const b = parseInt(hex[2] + hex[2], 16) / 255;
      return [r, g, b, 1];
    } else if (hex.length === 6) {
      const r = parseInt(hex.slice(0, 2), 16) / 255;
      const g = parseInt(hex.slice(2, 4), 16) / 255;
      const b = parseInt(hex.slice(4, 6), 16) / 255;
      return [r, g, b, 1];
    }
  }
  
  const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
  if (rgbaMatch) {
    const r = parseInt(rgbaMatch[1]) / 255;
    const g = parseInt(rgbaMatch[2]) / 255;
    const b = parseInt(rgbaMatch[3]) / 255;
    const a = rgbaMatch[4] ? parseFloat(rgbaMatch[4]) : 1;
    return [r, g, b, a];
  }
  
  return [0.5, 0.5, 0.5, 1];
}

/**
 * 从路径点计算贝塞尔曲线控制点
 * 假设路径是 [start, control1, control2, end] 或简单的 [start, end]
 */
function computeBezierControls(
  points: Array<{ x: number; y: number }>
): { start: { x: number; y: number }; end: { x: number; y: number }; control1: { x: number; y: number }; control2: { x: number; y: number } } {
  if (points.length < 2) {
    return {
      start: { x: 0, y: 0 },
      end: { x: 0, y: 0 },
      control1: { x: 0, y: 0 },
      control2: { x: 0, y: 0 },
    };
  }
  
  const start = points[0];
  const end = points[points.length - 1];
  
  // 如果有 4 个点，假设是贝塞尔曲线控制点
  if (points.length >= 4) {
    return {
      start,
      end,
      control1: points[1],
      control2: points[2],
    };
  }
  
  // 否则自动计算控制点（水平曲线）
  const dx = end.x - start.x;
  const controlOffset = Math.abs(dx) * 0.5;
  
  return {
    start,
    end,
    control1: { x: start.x + controlOffset, y: start.y },
    control2: { x: end.x - controlOffset, y: end.y },
  };
}


/**
 * 边批次管理器
 */
export class EdgeBatchManager {
  private batch: EdgeBatch;
  private edgeMap: Map<string, number> = new Map();
  private capacity: number;
  
  constructor(initialCapacity: number = 100) {
    this.capacity = initialCapacity;
    this.batch = {
      count: 0,
      instanceData: new Float32Array(initialCapacity * EDGE_INSTANCE_FLOATS),
      dirty: true,
    };
  }
  
  /**
   * 获取当前批次数据
   */
  getBatch(): EdgeBatch {
    return this.batch;
  }
  
  /**
   * 从 RenderPath 数组更新批次
   */
  updateFromPaths(paths: RenderPath[]): void {
    if (paths.length > this.capacity) {
      this.resize(Math.max(paths.length, this.capacity * 2));
    }
    
    this.edgeMap.clear();
    
    const data = this.batch.instanceData;
    for (let i = 0; i < paths.length; i++) {
      const path = paths[i];
      const offset = i * EDGE_INSTANCE_FLOATS;
      
      this.edgeMap.set(path.id, i);
      
      const { start, end, control1, control2 } = computeBezierControls(path.points);
      
      // start (vec2)
      data[offset + 0] = start.x;
      data[offset + 1] = start.y;
      
      // end (vec2)
      data[offset + 2] = end.x;
      data[offset + 3] = end.y;
      
      // control1 (vec2)
      data[offset + 4] = control1.x;
      data[offset + 5] = control1.y;
      
      // control2 (vec2)
      data[offset + 6] = control2.x;
      data[offset + 7] = control2.y;
      
      // color (vec4)
      const color = parseColor(path.color);
      data[offset + 8] = color[0];
      data[offset + 9] = color[1];
      data[offset + 10] = color[2];
      data[offset + 11] = color[3];
      
      // width (float)
      data[offset + 12] = path.width;
      
      // selected (float) - 从 animated 推断选中状态
      data[offset + 13] = path.animated ? 1.0 : 0.0;
    }
    
    this.batch.count = paths.length;
    this.batch.dirty = true;
  }
  
  /**
   * 更新单条边
   */
  updateEdge(path: RenderPath): void {
    const index = this.edgeMap.get(path.id);
    if (index === undefined) return;
    
    const data = this.batch.instanceData;
    const offset = index * EDGE_INSTANCE_FLOATS;
    
    const { start, end, control1, control2 } = computeBezierControls(path.points);
    
    data[offset + 0] = start.x;
    data[offset + 1] = start.y;
    data[offset + 2] = end.x;
    data[offset + 3] = end.y;
    data[offset + 4] = control1.x;
    data[offset + 5] = control1.y;
    data[offset + 6] = control2.x;
    data[offset + 7] = control2.y;
    data[offset + 13] = path.animated ? 1.0 : 0.0;
    
    this.batch.dirty = true;
  }
  
  private resize(newCapacity: number): void {
    const newData = new Float32Array(newCapacity * EDGE_INSTANCE_FLOATS);
    newData.set(this.batch.instanceData);
    this.batch.instanceData = newData;
    this.capacity = newCapacity;
  }
  
  clear(): void {
    this.batch.count = 0;
    this.edgeMap.clear();
    this.batch.dirty = true;
  }
}
