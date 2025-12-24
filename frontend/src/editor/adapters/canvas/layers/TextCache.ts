/**
 * TextCache - 文字渲染缓存
 * 
 * 将节点文字预渲染到 ImageBitmap，用于拖拽/缩放过程中的快速渲染。
 * 停止后重新渲染高质量文字。
 */

import type { RenderText } from '../../../core/RenderData';

/** 缓存条目 */
interface CacheEntry {
  /** 缓存的位图 */
  bitmap: ImageBitmap;
  /** 缓存时的缩放级别 */
  zoom: number;
  /** 缓存时的文字内容哈希 */
  hash: string;
  /** 最后访问时间 */
  lastAccess: number;
  /** 位图在画布坐标系中的位置 */
  x: number;
  y: number;
  /** 位图尺寸 */
  width: number;
  height: number;
}

/** 缓存配置 */
export interface TextCacheConfig {
  /** 最大缓存条目数 */
  maxEntries: number;
  /** 缓存过期时间（毫秒） */
  expireTime: number;
  /** 缩放变化阈值（超过此值重新渲染） */
  zoomThreshold: number;
  /** 预渲染时的 padding */
  padding: number;
}

const DEFAULT_CONFIG: TextCacheConfig = {
  maxEntries: 100,
  expireTime: 30000,
  zoomThreshold: 0.2,
  padding: 4,
};

/**
 * 文字缓存管理器
 */
export class TextCache {
  private cache: Map<string, CacheEntry> = new Map();
  private config: TextCacheConfig;
  private offscreenCanvas: OffscreenCanvas | null = null;
  private offscreenCtx: OffscreenCanvasRenderingContext2D | null = null;

  constructor(config: Partial<TextCacheConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.initOffscreenCanvas();
  }

  private initOffscreenCanvas(): void {
    if (typeof OffscreenCanvas !== 'undefined') {
      this.offscreenCanvas = new OffscreenCanvas(512, 256);
      this.offscreenCtx = this.offscreenCanvas.getContext('2d');
    }
  }

  /**
   * 获取或创建节点文字缓存
   */
  async getOrCreate(
    nodeId: string,
    texts: RenderText[],
    zoom: number
  ): Promise<CacheEntry | null> {
    if (!this.offscreenCanvas || !this.offscreenCtx) return null;

    const hash = this.computeHash(texts);
    const cacheKey = nodeId;
    const existing = this.cache.get(cacheKey);

    // 检查缓存是否有效
    if (existing && existing.hash === hash && Math.abs(existing.zoom - zoom) < this.config.zoomThreshold) {
      existing.lastAccess = Date.now();
      return existing;
    }

    // 创建新缓存
    const entry = await this.createCacheEntry(texts, zoom);
    if (entry) {
      entry.hash = hash;
      this.cache.set(cacheKey, entry);
      this.evictIfNeeded();
    }

    return entry;
  }

  /**
   * 使缓存失效
   */
  invalidate(nodeId: string): void {
    this.cache.delete(nodeId);
  }

  /**
   * 清空所有缓存
   */
  clear(): void {
    for (const entry of this.cache.values()) {
      entry.bitmap.close();
    }
    this.cache.clear();
  }

  /**
   * 绘制缓存的文字
   */
  draw(
    ctx: CanvasRenderingContext2D,
    entry: CacheEntry,
    offsetX: number = 0,
    offsetY: number = 0
  ): void {
    ctx.drawImage(
      entry.bitmap,
      entry.x + offsetX,
      entry.y + offsetY,
      entry.width,
      entry.height
    );
  }

  private async createCacheEntry(texts: RenderText[], zoom: number): Promise<CacheEntry | null> {
    if (!this.offscreenCanvas || !this.offscreenCtx || texts.length === 0) return null;

    // 计算边界
    const bounds = this.computeBounds(texts);
    const { padding } = this.config;
    const width = Math.ceil(bounds.width + padding * 2);
    const height = Math.ceil(bounds.height + padding * 2);

    // 调整 offscreen canvas 尺寸
    if (this.offscreenCanvas.width < width || this.offscreenCanvas.height < height) {
      this.offscreenCanvas.width = Math.max(this.offscreenCanvas.width, width);
      this.offscreenCanvas.height = Math.max(this.offscreenCanvas.height, height);
    }

    const ctx = this.offscreenCtx;
    ctx.clearRect(0, 0, width, height);

    // 渲染文字
    for (const text of texts) {
      ctx.font = `${text.fontSize ?? 12}px ${text.fontFamily ?? 'system-ui, sans-serif'}`;
      ctx.fillStyle = text.color ?? '#ffffff';
      ctx.textAlign = (text.align ?? 'left') as CanvasTextAlign;
      ctx.textBaseline = (text.baseline ?? 'top') as CanvasTextBaseline;
      ctx.fillText(
        text.text,
        text.x - bounds.x + padding,
        text.y - bounds.y + padding
      );
    }

    // 创建 ImageBitmap
    try {
      const bitmap = await createImageBitmap(
        this.offscreenCanvas,
        0, 0, width, height
      );

      return {
        bitmap,
        zoom,
        hash: '',
        lastAccess: Date.now(),
        x: bounds.x - padding,
        y: bounds.y - padding,
        width,
        height,
      };
    } catch {
      return null;
    }
  }

  private computeBounds(texts: RenderText[]): { x: number; y: number; width: number; height: number } {
    if (texts.length === 0) {
      return { x: 0, y: 0, width: 0, height: 0 };
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const text of texts) {
      const fontSize = text.fontSize ?? 12;
      // 估算文字宽度（实际应该用 measureText，但这里简化处理）
      const estimatedWidth = text.text.length * fontSize * 0.6;
      
      minX = Math.min(minX, text.x);
      minY = Math.min(minY, text.y);
      maxX = Math.max(maxX, text.x + estimatedWidth);
      maxY = Math.max(maxY, text.y + fontSize);
    }

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  private computeHash(texts: RenderText[]): string {
    // 简单哈希：拼接所有文字内容
    return texts.map(t => `${t.id}:${t.text}:${t.x}:${t.y}`).join('|');
  }

  private evictIfNeeded(): void {
    if (this.cache.size <= this.config.maxEntries) return;

    const now = Date.now();
    const toDelete: string[] = [];

    // 找出过期或最旧的条目
    let oldest: { key: string; time: number } | null = null;

    for (const [key, entry] of this.cache) {
      if (now - entry.lastAccess > this.config.expireTime) {
        toDelete.push(key);
      } else if (!oldest || entry.lastAccess < oldest.time) {
        oldest = { key, time: entry.lastAccess };
      }
    }

    // 删除过期条目
    for (const key of toDelete) {
      const entry = this.cache.get(key);
      entry?.bitmap.close();
      this.cache.delete(key);
    }

    // 如果还是超出限制，删除最旧的
    while (this.cache.size > this.config.maxEntries && oldest) {
      const entry = this.cache.get(oldest.key);
      entry?.bitmap.close();
      this.cache.delete(oldest.key);

      // 找下一个最旧的
      oldest = null;
      for (const [key, entry] of this.cache) {
        if (!oldest || entry.lastAccess < oldest.time) {
          oldest = { key, time: entry.lastAccess };
        }
      }
    }
  }

  /**
   * 销毁
   */
  dispose(): void {
    this.clear();
    this.offscreenCanvas = null;
    this.offscreenCtx = null;
  }
}
