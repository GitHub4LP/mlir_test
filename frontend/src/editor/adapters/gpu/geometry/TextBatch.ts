/**
 * 文字批次管理器
 * 使用离屏 Canvas 生成文字纹理，然后用 WebGL 渲染
 * 
 * DPR 处理策略：
 * - 图集使用物理像素尺寸（baseSize * dpr）以获得清晰文字
 * - 字体渲染使用物理像素大小（fontSize * dpr）
 * - entry 存储物理像素尺寸（用于 UV 计算）
 * - 渲染时使用逻辑尺寸（物理尺寸 / dpr）
 */

import type { TextBatch } from '../backends/IGPUBackend';
import type { RenderText } from '../../canvas/types';
import { layoutConfig } from '../../shared/styles';

/** 文字实例数据布局（每实例 floats 数量） */
const TEXT_INSTANCE_FLOATS = 16;
// position(2) + size(2) + uv0(2) + uv1(2) + color(4) + padding(4)

/** 纹理图集配置 */
const ATLAS_PADDING = 2;
const ATLAS_BASE_SIZE = 2048;

interface TextEntry {
  text: string;
  fontSize: number;      // 逻辑字体大小
  fontFamily: string;
  x: number;             // 图集中的物理像素位置
  y: number;
  physicalWidth: number; // 物理像素宽度（用于 UV 计算）
  physicalHeight: number;// 物理像素高度
}

/**
 * 解析 CSS 颜色为 RGBA 数组
 */
function parseColor(color: string | undefined): [number, number, number, number] {
  if (!color || color === 'transparent') return [1, 1, 1, 1];
  
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
  
  return [1, 1, 1, 1];
}

/**
 * 文字批次管理器
 */
export class TextBatchManager {
  private batch: TextBatch;
  private capacity: number;
  
  // 离屏 Canvas 用于生成文字纹理
  private atlasCanvas: OffscreenCanvas | HTMLCanvasElement;
  private atlasCtx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
  
  // 文字缓存
  private textCache: Map<string, TextEntry> = new Map();
  private currentX: number = ATLAS_PADDING;
  private currentY: number = ATLAS_PADDING;
  private rowHeight: number = 0;
  
  // 纹理是否需要更新
  private textureNeedsUpdate: boolean = false;
  
  // DPR 和图集尺寸（物理像素）
  private dpr: number = 1;
  private atlasSize: number = ATLAS_BASE_SIZE;
  
  constructor(initialCapacity: number = 500) {
    this.capacity = initialCapacity;
    this.batch = {
      count: 0,
      instanceData: new Float32Array(initialCapacity * TEXT_INSTANCE_FLOATS),
      dirty: true,
    };
    
    // 强制至少 2x 渲染，确保缩小时仍清晰
    const systemDpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1;
    this.dpr = Math.max(2, systemDpr);
    this.atlasSize = Math.ceil(ATLAS_BASE_SIZE * this.dpr);
    
    // 创建离屏 Canvas（物理像素尺寸）
    if (typeof OffscreenCanvas !== 'undefined') {
      this.atlasCanvas = new OffscreenCanvas(this.atlasSize, this.atlasSize);
      this.atlasCtx = this.atlasCanvas.getContext('2d')!;
    } else {
      this.atlasCanvas = document.createElement('canvas');
      this.atlasCanvas.width = this.atlasSize;
      this.atlasCanvas.height = this.atlasSize;
      this.atlasCtx = this.atlasCanvas.getContext('2d')!;
    }
    
    // 初始化 Canvas 状态
    this.atlasCtx.textBaseline = 'top';
    // 启用高质量文字渲染
    this.atlasCtx.imageSmoothingEnabled = true;
    this.atlasCtx.imageSmoothingQuality = 'high';
  }
  
  getBatch(): TextBatch {
    return this.batch;
  }
  
  getAtlasCanvas(): OffscreenCanvas | HTMLCanvasElement {
    return this.atlasCanvas;
  }
  
  needsTextureUpdate(): boolean {
    return this.textureNeedsUpdate;
  }
  
  clearTextureUpdateFlag(): void {
    this.textureNeedsUpdate = false;
  }
  
  /**
   * 生成文字的缓存 key
   */
  private getTextKey(text: string, fontSize: number, fontFamily: string): string {
    return `${text}|${fontSize}|${fontFamily}`;
  }
  
  /**
   * 将文字渲染到图集中
   */
  private renderTextToAtlas(text: string, fontSize: number, fontFamily: string): TextEntry {
    const key = this.getTextKey(text, fontSize, fontFamily);
    
    // 检查缓存
    const cached = this.textCache.get(key);
    if (cached) return cached;
    
    // 使用物理像素大小渲染以获得清晰文字
    const physicalFontSize = fontSize * this.dpr;
    
    // 设置字体（物理像素大小）
    this.atlasCtx.font = `${physicalFontSize}px ${fontFamily}`;
    // 使用 top baseline，这样 y 坐标就是文字顶部
    this.atlasCtx.textBaseline = 'top';
    
    // 测量文字尺寸（物理像素）
    const metrics = this.atlasCtx.measureText(text);
    const physicalWidth = Math.ceil(metrics.width) + ATLAS_PADDING * 2;
    // 使用 fontBoundingBox 获取更精确的高度（如果可用）
    const actualHeight = metrics.fontBoundingBoxAscent !== undefined
      ? metrics.fontBoundingBoxAscent + metrics.fontBoundingBoxDescent
      : physicalFontSize * 1.2;
    const physicalHeight = Math.ceil(actualHeight) + ATLAS_PADDING * 2;
    
    // 检查是否需要换行
    if (this.currentX + physicalWidth > this.atlasSize) {
      this.currentX = ATLAS_PADDING;
      this.currentY += this.rowHeight + ATLAS_PADDING;
      this.rowHeight = 0;
    }
    
    // 检查是否超出图集
    if (this.currentY + physicalHeight > this.atlasSize) {
      // 图集已满，清空重新开始
      this.clearAtlas();
    }
    
    // 渲染文字（白色，颜色在着色器中应用）
    // 使用 textBaseline: 'top'，所以 y 是文字顶部
    this.atlasCtx.fillStyle = 'white';
    this.atlasCtx.fillText(text, this.currentX + ATLAS_PADDING, this.currentY + ATLAS_PADDING);
    
    // 创建条目（存储物理像素尺寸，用于 UV 计算）
    const entry: TextEntry = {
      text,
      fontSize,
      fontFamily,
      x: this.currentX,
      y: this.currentY,
      physicalWidth,
      physicalHeight,
    };
    
    // 更新位置
    this.currentX += physicalWidth + ATLAS_PADDING;
    this.rowHeight = Math.max(this.rowHeight, physicalHeight);
    
    // 缓存
    this.textCache.set(key, entry);
    this.textureNeedsUpdate = true;
    
    return entry;
  }
  
  /**
   * 清空图集
   */
  private clearAtlas(): void {
    this.atlasCtx.clearRect(0, 0, this.atlasSize, this.atlasSize);
    this.textCache.clear();
    this.currentX = ATLAS_PADDING;
    this.currentY = ATLAS_PADDING;
    this.rowHeight = 0;
    this.textureNeedsUpdate = true;
  }
  
  /**
   * 从 RenderText 数组更新批次
   */
  updateFromTexts(texts: RenderText[]): void {
    if (texts.length > this.capacity) {
      this.resize(Math.max(texts.length, this.capacity * 2));
    }
    
    const data = this.batch.instanceData;
    
    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      const offset = i * TEXT_INSTANCE_FLOATS;
      
      const fontSize = text.fontSize ?? 12;
      const fontFamily = text.fontFamily ?? layoutConfig.text.fontFamily;
      
      // 处理 ellipsis 文本截断
      let displayText = text.text;
      if (text.ellipsis && text.maxWidth !== undefined && text.maxWidth > 0) {
        displayText = this.truncateTextWithEllipsis(text.text, fontSize, fontFamily, text.maxWidth);
      }
      
      // 渲染文字到图集并获取位置
      const entry = this.renderTextToAtlas(displayText, fontSize, fontFamily);
      
      // 逻辑尺寸（用于渲染定位）
      const logicalWidth = entry.physicalWidth / this.dpr;
      const logicalHeight = entry.physicalHeight / this.dpr;
      
      // 计算实际渲染位置（考虑对齐）
      let x = text.x;
      let y = text.y;
      
      // 水平对齐
      if (text.align === 'center') {
        x -= logicalWidth / 2;
      } else if (text.align === 'right') {
        x -= logicalWidth;
      }
      
      // 垂直对齐（与 Canvas textBaseline 一致）
      // Canvas textBaseline: 'top' | 'middle' | 'bottom' | 'alphabetic' | 'hanging' | 'ideographic'
      // 我们主要支持 'top', 'middle', 'bottom'
      const baseline = text.baseline ?? 'top';
      if (baseline === 'middle') {
        y -= logicalHeight / 2;
      } else if (baseline === 'bottom') {
        y -= logicalHeight;
      }
      // 'top' 不需要调整，y 就是顶部位置
      
      // position (vec2)
      data[offset + 0] = x;
      data[offset + 1] = y;
      
      // size (vec2) - 逻辑尺寸
      data[offset + 2] = logicalWidth;
      data[offset + 3] = logicalHeight;
      
      // uv0 (vec2) - 左上角（使用物理像素计算 UV）
      data[offset + 4] = entry.x / this.atlasSize;
      data[offset + 5] = entry.y / this.atlasSize;
      
      // uv1 (vec2) - 右下角
      data[offset + 6] = (entry.x + entry.physicalWidth) / this.atlasSize;
      data[offset + 7] = (entry.y + entry.physicalHeight) / this.atlasSize;
      
      // color (vec4)
      const color = parseColor(text.color);
      data[offset + 8] = color[0];
      data[offset + 9] = color[1];
      data[offset + 10] = color[2];
      data[offset + 11] = color[3];
      
      // padding (4 floats)
      data[offset + 12] = 0;
      data[offset + 13] = 0;
      data[offset + 14] = 0;
      data[offset + 15] = 0;
    }
    
    this.batch.count = texts.length;
    this.batch.dirty = true;
  }
  
  /**
   * 截断文本并添加省略号
   */
  private truncateTextWithEllipsis(text: string, fontSize: number, fontFamily: string, maxWidth: number): string {
    const physicalFontSize = fontSize * this.dpr;
    const physicalMaxWidth = maxWidth * this.dpr;
    
    this.atlasCtx.font = `${physicalFontSize}px ${fontFamily}`;
    
    // 测量完整文本宽度
    const fullWidth = this.atlasCtx.measureText(text).width;
    if (fullWidth <= physicalMaxWidth) {
      return text;
    }
    
    // 测量省略号宽度
    const ellipsis = '…';
    const ellipsisWidth = this.atlasCtx.measureText(ellipsis).width;
    const availableWidth = physicalMaxWidth - ellipsisWidth;
    
    if (availableWidth <= 0) {
      return ellipsis;
    }
    
    // 二分查找合适的截断位置
    let low = 0;
    let high = text.length;
    
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      const truncated = text.slice(0, mid);
      const width = this.atlasCtx.measureText(truncated).width;
      
      if (width <= availableWidth) {
        low = mid;
      } else {
        high = mid - 1;
      }
    }
    
    if (low === 0) {
      return ellipsis;
    }
    
    return text.slice(0, low) + ellipsis;
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
