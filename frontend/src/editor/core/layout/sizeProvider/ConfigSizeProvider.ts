/**
 * ConfigSizeProvider
 * 从配置计算尺寸，用于 Canvas/GPU 渲染器
 */

import type { SizeProvider } from './types';
import type { Size, ContainerConfig } from '../types';
import { getContainerConfig } from '../tokens';

/** 文本测量缓存 */
const textWidthCache = new Map<string, number>();

/** Canvas 上下文（用于测量文本） */
let measureContext: CanvasRenderingContext2D | null = null;

/**
 * 获取 Canvas 测量上下文
 */
function getMeasureContext(): CanvasRenderingContext2D {
  if (!measureContext) {
    const canvas = document.createElement('canvas');
    measureContext = canvas.getContext('2d')!;
  }
  return measureContext;
}

/**
 * 测量文本宽度
 * @param text - 文本内容
 * @param fontSize - 字体大小
 * @param fontFamily - 字体族
 */
function measureTextWidth(
  text: string,
  fontSize: number = 12,
  fontFamily: string = 'system-ui, -apple-system, sans-serif'
): number {
  const cacheKey = `${text}|${fontSize}|${fontFamily}`;
  
  if (textWidthCache.has(cacheKey)) {
    return textWidthCache.get(cacheKey)!;
  }
  
  const ctx = getMeasureContext();
  ctx.font = `${fontSize}px ${fontFamily}`;
  const width = ctx.measureText(text).width;
  
  textWidthCache.set(cacheKey, width);
  return width;
}

/**
 * 从配置获取 padding
 */
function getPadding(config: ContainerConfig): { left: number; right: number; top: number; bottom: number } {
  return {
    left: config.paddingLeft ?? 0,
    right: config.paddingRight ?? 0,
    top: config.paddingTop ?? 0,
    bottom: config.paddingBottom ?? 0,
  };
}

/**
 * ConfigSizeProvider 实现
 * 从配置读取 padding、minWidth、minHeight，计算组件尺寸
 */
export class ConfigSizeProvider implements SizeProvider {
  private fontFamily: string;
  private fontSize: number;

  constructor(options?: { fontFamily?: string; fontSize?: number }) {
    this.fontFamily = options?.fontFamily ?? 'system-ui, -apple-system, sans-serif';
    this.fontSize = options?.fontSize ?? 12;
  }

  /**
   * 获取组件尺寸
   * 计算逻辑：
   * 1. 读取配置中的 padding、minWidth、minHeight
   * 2. 计算 文本宽度 + padding.left + padding.right
   * 3. 应用 minWidth/minHeight 约束
   */
  getSize(type: string, content?: string): Size {
    const config = getContainerConfig(type);
    const padding = getPadding(config);
    
    // 计算内容宽度
    let contentWidth = 0;
    if (content) {
      contentWidth = measureTextWidth(content, this.fontSize, this.fontFamily);
    }
    
    // 计算总宽度和高度
    let width = contentWidth + padding.left + padding.right;
    let height = this.fontSize + padding.top + padding.bottom;
    
    // 应用 minWidth/minHeight 约束
    if (config.minWidth !== undefined && width < config.minWidth) {
      width = config.minWidth;
    }
    if (config.minHeight !== undefined && height < config.minHeight) {
      height = config.minHeight;
    }
    
    // 如果配置了固定宽高，使用固定值
    if (typeof config.width === 'number') {
      width = config.width;
    }
    if (typeof config.height === 'number') {
      height = config.height;
    }
    
    return { width, height };
  }

  /**
   * 清除文本测量缓存
   */
  clearCache(): void {
    textWidthCache.clear();
  }
}

/** 默认 ConfigSizeProvider 实例 */
export const configSizeProvider = new ConfigSizeProvider();
