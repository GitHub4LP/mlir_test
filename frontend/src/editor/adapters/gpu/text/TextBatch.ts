/**
 * 文本批次管理器
 * 
 * 将 RenderText 数据转换为 GPU 实例数据。
 * 每个字符是一个实例，使用字体图集进行渲染。
 */

import type { TextBatch } from '../backends/IGPUBackend';
import type { RenderText } from '../../canvas/types';
import type { FontAtlas } from './FontAtlas';

/** 文本实例数据布局（每字符 floats 数量） */
const TEXT_INSTANCE_FLOATS = 12; // position(2) + size(2) + uv0(2) + uv1(2) + color(4)

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
  
  return [1, 1, 1, 1]; // 默认白色
}


/**
 * 文本批次管理器
 */
export class TextBatchManager {
  private batch: TextBatch;
  private capacity: number;
  private fontAtlas: FontAtlas | null = null;
  
  constructor(initialCapacity: number = 1000) {
    this.capacity = initialCapacity;
    this.batch = {
      count: 0,
      instanceData: new Float32Array(initialCapacity * TEXT_INSTANCE_FLOATS),
      dirty: true,
    };
  }
  
  /**
   * 设置字体图集
   */
  setFontAtlas(atlas: FontAtlas): void {
    this.fontAtlas = atlas;
  }
  
  /**
   * 获取当前批次数据
   */
  getBatch(): TextBatch {
    return this.batch;
  }
  
  /**
   * 从 RenderText 数组更新批次
   */
  updateFromTexts(texts: RenderText[]): void {
    if (!this.fontAtlas || this.fontAtlas.getState() !== 'ready') {
      this.batch.count = 0;
      return;
    }
    
    // 计算总字符数
    let totalChars = 0;
    for (const text of texts) {
      totalChars += text.text.length;
    }
    
    // 检查是否需要扩容
    if (totalChars > this.capacity) {
      this.resize(Math.max(totalChars, this.capacity * 2));
    }
    
    const data = this.batch.instanceData;
    let charIndex = 0;
    
    for (const text of texts) {
      const color = parseColor(text.color);
      
      // 计算基线偏移
      let baselineOffset = 0;
      const fontData = this.fontAtlas.getData();
      if (fontData) {
        const scale = text.fontSize / fontData.size;
        if (text.baseline === 'middle') {
          baselineOffset = -fontData.lineHeight * scale / 2;
        } else if (text.baseline === 'bottom') {
          baselineOffset = -fontData.lineHeight * scale;
        }
      }
      
      // 布局文本
      const layout = this.fontAtlas.layoutText(
        text.text,
        text.x,
        text.y + baselineOffset,
        text.fontSize,
        text.align
      );
      
      // 填充实例数据
      for (const char of layout) {
        const offset = charIndex * TEXT_INSTANCE_FLOATS;
        
        // position (vec2)
        data[offset + 0] = char.x;
        data[offset + 1] = char.y;
        
        // size (vec2)
        data[offset + 2] = char.width;
        data[offset + 3] = char.height;
        
        // uv0 (vec2)
        data[offset + 4] = char.u0;
        data[offset + 5] = char.v0;
        
        // uv1 (vec2)
        data[offset + 6] = char.u1;
        data[offset + 7] = char.v1;
        
        // color (vec4)
        data[offset + 8] = color[0];
        data[offset + 9] = color[1];
        data[offset + 10] = color[2];
        data[offset + 11] = color[3];
        
        charIndex++;
      }
    }
    
    this.batch.count = charIndex;
    this.batch.dirty = true;
  }
  
  private resize(newCapacity: number): void {
    const newData = new Float32Array(newCapacity * TEXT_INSTANCE_FLOATS);
    newData.set(this.batch.instanceData);
    this.batch.instanceData = newData;
    this.capacity = newCapacity;
  }
  
  clear(): void {
    this.batch.count = 0;
    this.batch.dirty = true;
  }
}
