/**
 * 节点批次管理器
 * 
 * 将 RenderRect 数据转换为 GPU 实例数据。
 * 支持增量更新以减少 GPU 数据传输。
 */

import type { NodeBatch } from '../backends/IGPUBackend';
import type { RenderRect } from '../../canvas/types';

/** 节点实例数据布局（每实例 floats 数量） */
const NODE_INSTANCE_FLOATS = 16;

/** 默认标题高度 */
const DEFAULT_HEADER_HEIGHT = 28;

/**
 * 解析 CSS 颜色为 RGBA 数组
 */
function parseColor(color: string): [number, number, number, number] {
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
   * 注意：处理节点主体矩形（id 以 'rect-' 开头）和对应的 header 矩形
   */
  updateFromRects(rects: RenderRect[]): void {
    // 分离节点主体矩形和 header 矩形
    const nodeRects = rects.filter(r => r.id.startsWith('rect-'));
    const headerRects = new Map<string, RenderRect>();
    for (const r of rects) {
      if (r.id.startsWith('header-')) {
        // header-{nodeId} -> nodeId
        const nodeId = r.id.slice(7);
        headerRects.set(nodeId, r);
      }
    }
    
    // 检查是否需要扩容
    if (nodeRects.length > this.capacity) {
      this.resize(Math.max(nodeRects.length, this.capacity * 2));
    }
    
    // 重建节点映射
    this.nodeMap.clear();
    
    // 填充实例数据
    const data = this.batch.instanceData;
    for (let i = 0; i < nodeRects.length; i++) {
      const rect = nodeRects[i];
      const offset = i * NODE_INSTANCE_FLOATS;
      
      // rect-{nodeId} -> nodeId
      const nodeId = rect.id.slice(5);
      const header = headerRects.get(nodeId);
      
      this.nodeMap.set(rect.id, i);
      
      // position (vec2)
      data[offset + 0] = rect.x;
      data[offset + 1] = rect.y;
      
      // size (vec2)
      data[offset + 2] = rect.width;
      data[offset + 3] = rect.height;
      
      // headerHeight (float) - 从 header 矩形获取，或使用默认值
      data[offset + 4] = header?.height ?? DEFAULT_HEADER_HEIGHT;
      
      // borderRadius (float) - 如果是对象则取平均值
      const br = rect.borderRadius;
      data[offset + 5] = typeof br === 'number' ? br : (br.topLeft + br.topRight + br.bottomLeft + br.bottomRight) / 4;
      
      // bodyColor (vec4)
      const bodyColor = parseColor(rect.fillColor);
      data[offset + 6] = bodyColor[0];
      data[offset + 7] = bodyColor[1];
      data[offset + 8] = bodyColor[2];
      data[offset + 9] = bodyColor[3];
      
      // headerColor (vec4) - 从 header 矩形获取颜色
      const headerColor = header ? parseColor(header.fillColor) : bodyColor;
      data[offset + 10] = headerColor[0];
      data[offset + 11] = headerColor[1];
      data[offset + 12] = headerColor[2];
      data[offset + 13] = headerColor[3];
      
      // selected (float)
      data[offset + 14] = rect.selected ? 1.0 : 0.0;
      
      // padding (float)
      data[offset + 15] = 0;
    }
    
    this.batch.count = nodeRects.length;
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
