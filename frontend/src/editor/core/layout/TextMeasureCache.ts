/**
 * 文本测量缓存
 * 使用 LRU 策略缓存 Canvas measureText 结果，减少重复测量开销
 */

import { layoutConfig } from './LayoutConfig';

/**
 * LRU 缓存条目
 */
interface CacheEntry {
  width: number;
  /** 用于 LRU 排序的访问时间戳 */
  lastAccess: number;
}

/**
 * 文本测量缓存类
 * 
 * 缓存 Canvas measureText 的结果，避免重复测量相同文本。
 * 使用 LRU（最近最少使用）策略限制缓存大小。
 */
export class TextMeasureCache {
  private cache = new Map<string, CacheEntry>();
  private maxSize: number;
  private measureContext: CanvasRenderingContext2D | null = null;
  
  constructor(maxSize: number = 1000) {
    this.maxSize = maxSize;
  }
  
  /**
   * 生成缓存键
   * @param text - 文本内容
   * @param fontSize - 字体大小
   * @param fontWeight - 字重
   * @returns 缓存键
   */
  private makeKey(text: string, fontSize: number, fontWeight: number): string {
    return `${text}|${fontSize}|${fontWeight}`;
  }
  
  /**
   * 获取用于测量的 Canvas 上下文
   */
  private getMeasureContext(): CanvasRenderingContext2D {
    if (!this.measureContext) {
      const canvas = document.createElement('canvas');
      this.measureContext = canvas.getContext('2d')!;
    }
    return this.measureContext;
  }
  
  /**
   * 测量文本宽度（带缓存）
   * @param text - 文本内容
   * @param fontSize - 字体大小
   * @param fontWeight - 字重（默认 400）
   * @returns 文本宽度
   */
  measureText(text: string, fontSize: number, fontWeight: number = 400): number {
    const key = this.makeKey(text, fontSize, fontWeight);
    
    // 检查缓存
    const cached = this.cache.get(key);
    if (cached !== undefined) {
      // 更新访问时间
      cached.lastAccess = Date.now();
      return cached.width;
    }
    
    // 缓存未命中，执行测量
    const ctx = this.getMeasureContext();
    const fontFamily = layoutConfig.text.fontFamily;
    ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;
    const width = ctx.measureText(text).width;
    
    // 检查是否需要淘汰
    if (this.cache.size >= this.maxSize) {
      this.evictOldest();
    }
    
    // 添加到缓存
    this.cache.set(key, {
      width,
      lastAccess: Date.now(),
    });
    
    return width;
  }
  
  /**
   * 淘汰最旧的缓存条目
   */
  private evictOldest(): void {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;
    
    for (const [key, entry] of this.cache) {
      if (entry.lastAccess < oldestTime) {
        oldestTime = entry.lastAccess;
        oldestKey = key;
      }
    }
    
    if (oldestKey !== null) {
      this.cache.delete(oldestKey);
    }
  }
  
  /**
   * 检查缓存是否包含指定键
   * @param text - 文本内容
   * @param fontSize - 字体大小
   * @param fontWeight - 字重
   * @returns 是否存在缓存
   */
  has(text: string, fontSize: number, fontWeight: number = 400): boolean {
    const key = this.makeKey(text, fontSize, fontWeight);
    return this.cache.has(key);
  }
  
  /**
   * 获取缓存大小
   */
  get size(): number {
    return this.cache.size;
  }
  
  /**
   * 获取最大缓存大小
   */
  get capacity(): number {
    return this.maxSize;
  }
  
  /**
   * 清空缓存
   */
  clear(): void {
    this.cache.clear();
  }
}

/**
 * 全局文本测量缓存实例
 */
export const textMeasureCache = new TextMeasureCache();
