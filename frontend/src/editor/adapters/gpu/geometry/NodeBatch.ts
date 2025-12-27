/**
 * 节点批次管理器
 * 
 * 将 RenderRect 数据转换为 GPU 实例数据。
 * 支持增量更新以减少 GPU 数据传输。
 */

import type { NodeBatch } from '../backends/IGPUBackend';
import type { RenderRect } from '../../canvas/types';

/** 节点实例数据布局（每实例 floats 数量）
 * position: vec2 (2)
 * size: vec2 (2)
 * headerHeight: float (1)
 * borderRadius: vec4 (4) - topLeft, topRight, bottomRight, bottomLeft
 * bodyColor: vec4 (4)
 * headerColor: vec4 (4)
 * selected: float (1)
 * Total: 18
 */
const NODE_INSTANCE_FLOATS = 18;

/**
 * 解析 CSS 颜色为 RGBA 数组
 */
function parseColor(color: string): [number, number, number, number] {
  // 处理 transparent
  if (color === 'transparent') {
    return [0, 0, 0, 0];
  }
  
  // 处理 hex 颜色
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
  
  // 处理 rgb/rgba 颜色
  const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
  if (rgbaMatch) {
    const r = parseInt(rgbaMatch[1]) / 255;
    const g = parseInt(rgbaMatch[2]) / 255;
    const b = parseInt(rgbaMatch[3]) / 255;
    const a = rgbaMatch[4] ? parseFloat(rgbaMatch[4]) : 1;
    return [r, g, b, a];
  }
  
  // 默认灰色
  return [0.5, 0.5, 0.5, 1];
}

/**
 * 节点批次管理器
 */
export class NodeBatchManager {
  private batch: NodeBatch;
  private nodeMap: Map<string, number> = new Map(); // id -> index
  private capacity: number;
  
  constructor(initialCapacity: number = 100) {
    this.capacity = initialCapacity;
    this.batch = {
      count: 0,
      instanceData: new Float32Array(initialCapacity * NODE_INSTANCE_FLOATS),
      dirty: true,
    };
  }

  
  /**
   * 获取当前批次数据
   */
  getBatch(): NodeBatch {
    return this.batch;
  }
  
  /**
   * 从 RenderRect 数组更新批次
   * 支持任意矩形渲染，不依赖特定 ID 格式
   */
  updateFromRects(rects: RenderRect[]): void {
    // 按 zIndex 排序
    const sortedRects = [...rects].sort((a, b) => (a.zIndex ?? 0) - (b.zIndex ?? 0));
    
    // 检查是否需要扩容
    if (sortedRects.length > this.capacity) {
      this.resize(Math.max(sortedRects.length, this.capacity * 2));
    }
    
    // 重建映射
    this.nodeMap.clear();
    
    // 填充实例数据
    const data = this.batch.instanceData;
    for (let i = 0; i < sortedRects.length; i++) {
      const rect = sortedRects[i];
      const offset = i * NODE_INSTANCE_FLOATS;
      
      this.nodeMap.set(rect.id, i);
      
      // position (vec2)
      data[offset + 0] = rect.x;
      data[offset + 1] = rect.y;
      
      // size (vec2)
      data[offset + 2] = rect.width;
      data[offset + 3] = rect.height;
      
      // headerHeight (float) - 通用矩形设为 0（无 header 概念）
      data[offset + 4] = 0;
      
      // borderRadius (vec4) - topLeft, topRight, bottomRight, bottomLeft
      const br = rect.borderRadius;
      if (typeof br === 'number') {
        data[offset + 5] = br;
        data[offset + 6] = br;
        data[offset + 7] = br;
        data[offset + 8] = br;
      } else if (br) {
        data[offset + 5] = br.topLeft;
        data[offset + 6] = br.topRight;
        data[offset + 7] = br.bottomRight;
        data[offset + 8] = br.bottomLeft;
      } else {
        data[offset + 5] = 0;
        data[offset + 6] = 0;
        data[offset + 7] = 0;
        data[offset + 8] = 0;
      }
      
      // bodyColor (vec4)
      const bodyColor = parseColor(rect.fillColor);
      data[offset + 9] = bodyColor[0];
      data[offset + 10] = bodyColor[1];
      data[offset + 11] = bodyColor[2];
      data[offset + 12] = bodyColor[3];
      
      // headerColor (vec4) - 与 bodyColor 相同
      data[offset + 13] = bodyColor[0];
      data[offset + 14] = bodyColor[1];
      data[offset + 15] = bodyColor[2];
      data[offset + 16] = bodyColor[3];
      
      // selected (float)
      data[offset + 17] = rect.selected ? 1.0 : 0.0;
    }
    
    this.batch.count = sortedRects.length;
    this.batch.dirty = true;
  }
  
  /**
   * 更新单个节点
   */
  updateNode(rect: RenderRect): void {
    const index = this.nodeMap.get(rect.id);
    if (index === undefined) return;
    
    const data = this.batch.instanceData;
    const offset = index * NODE_INSTANCE_FLOATS;
    
    // position
    data[offset + 0] = rect.x;
    data[offset + 1] = rect.y;
    
    // size
    data[offset + 2] = rect.width;
    data[offset + 3] = rect.height;
    
    // selected
    data[offset + 14] = rect.selected ? 1.0 : 0.0;
    
    this.batch.dirty = true;
  }
  
  /**
   * 扩容
   */
  private resize(newCapacity: number): void {
    const newData = new Float32Array(newCapacity * NODE_INSTANCE_FLOATS);
    newData.set(this.batch.instanceData);
    this.batch.instanceData = newData;
    this.capacity = newCapacity;
  }
  
  /**
   * 清空批次
   */
  clear(): void {
    this.batch.count = 0;
    this.nodeMap.clear();
    this.batch.dirty = true;
  }
}
