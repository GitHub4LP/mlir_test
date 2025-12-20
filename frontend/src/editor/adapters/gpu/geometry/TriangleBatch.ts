/**
 * 三角形批次管理器
 * 将 RenderTriangle 数据转换为 GPU 实例数据
 */

import type { TriangleBatch } from '../backends/IGPUBackend';
import type { RenderTriangle } from '../../canvas/types';

/** 三角形实例数据布局（每实例 floats 数量） */
const TRIANGLE_INSTANCE_FLOATS = 13;
// position(2) + size(1) + direction(1) + fillColor(4) + borderColor(4) + borderWidth(1)

/**
 * 解析 CSS 颜色为 RGBA 数组
 */
function parseColor(color: string | undefined): [number, number, number, number] {
  if (!color || color === 'transparent') {
    return [0, 0, 0, 0];
  }
  
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
    } else if (hex.length === 8) {
      const r = parseInt(hex.slice(0, 2), 16) / 255;
      const g = parseInt(hex.slice(2, 4), 16) / 255;
      const b = parseInt(hex.slice(4, 6), 16) / 255;
      const a = parseInt(hex.slice(6, 8), 16) / 255;
      return [r, g, b, a];
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
 * 三角形批次管理器
 */
export class TriangleBatchManager {
  private batch: TriangleBatch;
  private capacity: number;
  
  constructor(initialCapacity: number = 100) {
    this.capacity = initialCapacity;
    this.batch = {
      count: 0,
      instanceData: new Float32Array(initialCapacity * TRIANGLE_INSTANCE_FLOATS),
      dirty: true,
    };
  }
  
  getBatch(): TriangleBatch {
    return this.batch;
  }
  
  updateFromTriangles(triangles: RenderTriangle[]): void {
    if (triangles.length > this.capacity) {
      this.resize(Math.max(triangles.length, this.capacity * 2));
    }
    
    const data = this.batch.instanceData;
    for (let i = 0; i < triangles.length; i++) {
      const triangle = triangles[i];
      const offset = i * TRIANGLE_INSTANCE_FLOATS;
      
      // position (vec2)
      data[offset + 0] = triangle.x;
      data[offset + 1] = triangle.y;
      
      // size (float)
      data[offset + 2] = triangle.size;
      
      // direction (float): 1.0 = right, -1.0 = left
      data[offset + 3] = triangle.direction === 'right' ? 1.0 : -1.0;
      
      // fillColor (vec4)
      const fillColor = parseColor(triangle.fillColor);
      data[offset + 4] = fillColor[0];
      data[offset + 5] = fillColor[1];
      data[offset + 6] = fillColor[2];
      data[offset + 7] = fillColor[3];
      
      // borderColor (vec4)
      const borderColor = parseColor(triangle.borderColor);
      data[offset + 8] = borderColor[0];
      data[offset + 9] = borderColor[1];
      data[offset + 10] = borderColor[2];
      data[offset + 11] = borderColor[3];
      
      // borderWidth (float)
      data[offset + 12] = triangle.borderWidth ?? 0;
    }
    
    this.batch.count = triangles.length;
    this.batch.dirty = true;
  }
  
  private resize(newCapacity: number): void {
    const newData = new Float32Array(newCapacity * TRIANGLE_INSTANCE_FLOATS);
    newData.set(this.batch.instanceData);
    this.batch.instanceData = newData;
    this.capacity = newCapacity;
  }
  
  clear(): void {
    this.batch.count = 0;
    this.batch.dirty = true;
  }
}
