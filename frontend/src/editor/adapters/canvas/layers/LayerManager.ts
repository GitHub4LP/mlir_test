/**
 * LayerManager - 多层 Canvas 管理器
 * 
 * 管理多个 Canvas 层，实现分层渲染：
 * - Content Layer: 节点、连线、端口、文字
 * - Interaction Layer: 连线预览、选择框、hover
 * - UI Layer: 类型选择器、属性编辑器、菜单
 */

import type { Viewport } from '../../../core/RenderData';

export type LayerName = 'content' | 'interaction' | 'ui';

export interface LayerConfig {
  name: LayerName;
  zIndex: number;
  /** 是否接收指针事件 */
  pointerEvents: boolean;
}

// Content 层由 IContentRenderer 自己管理，LayerManager 只管理 interaction 和 ui 层
const LAYER_CONFIGS: LayerConfig[] = [
  { name: 'interaction', zIndex: 2, pointerEvents: false },
  { name: 'ui', zIndex: 3, pointerEvents: true },
];

export class LayerManager {
  private container: HTMLElement | null = null;
  private layers: Map<LayerName, HTMLCanvasElement> = new Map();
  private contexts: Map<LayerName, CanvasRenderingContext2D> = new Map();
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private width: number = 0;
  private height: number = 0;
  private dpr: number = 1;

  /**
   * 挂载到容器
   */
  mount(container: HTMLElement): void {
    this.container = container;
    this.dpr = window.devicePixelRatio || 1;
    
    // 创建所有层
    for (const config of LAYER_CONFIGS) {
      const canvas = document.createElement('canvas');
      canvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: ${config.pointerEvents ? 'auto' : 'none'};
        z-index: ${config.zIndex};
      `;
      canvas.dataset.layer = config.name;
      
      container.appendChild(canvas);
      this.layers.set(config.name, canvas);
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        this.contexts.set(config.name, ctx);
      }
    }
    
    this.updateSize();
  }

  /**
   * 卸载
   */
  unmount(): void {
    for (const canvas of this.layers.values()) {
      canvas.remove();
    }
    this.layers.clear();
    this.contexts.clear();
    this.container = null;
  }

  /**
   * 更新尺寸
   */
  resize(): void {
    this.updateSize();
  }

  private updateSize(): void {
    if (!this.container) return;
    
    const rect = this.container.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    this.dpr = window.devicePixelRatio || 1;
    
    for (const canvas of this.layers.values()) {
      canvas.width = this.width * this.dpr;
      canvas.height = this.height * this.dpr;
      canvas.style.width = `${this.width}px`;
      canvas.style.height = `${this.height}px`;
    }
  }

  /**
   * 获取指定层的 Canvas
   */
  getCanvas(layer: LayerName): HTMLCanvasElement | null {
    return this.layers.get(layer) ?? null;
  }

  /**
   * 获取指定层的 Context
   */
  getContext(layer: LayerName): CanvasRenderingContext2D | null {
    return this.contexts.get(layer) ?? null;
  }

  /**
   * 清空指定层
   */
  clearLayer(layer: LayerName): void {
    const canvas = this.layers.get(layer);
    const ctx = this.contexts.get(layer);
    if (canvas && ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  /**
   * 清空所有层
   */
  clearAll(): void {
    for (const layer of LAYER_CONFIGS) {
      this.clearLayer(layer.name);
    }
  }

  /**
   * 设置视口
   */
  setViewport(viewport: Viewport): void {
    this.viewport = { ...viewport };
  }

  /**
   * 获取视口
   */
  getViewport(): Viewport {
    return { ...this.viewport };
  }

  /**
   * 获取尺寸
   */
  getSize(): { width: number; height: number; dpr: number } {
    return { width: this.width, height: this.height, dpr: this.dpr };
  }

  /**
   * 应用视口变换到 Context
   */
  applyViewportTransform(ctx: CanvasRenderingContext2D): void {
    ctx.scale(this.dpr, this.dpr);
    ctx.translate(this.viewport.x, this.viewport.y);
    ctx.scale(this.viewport.zoom, this.viewport.zoom);
  }

  /**
   * 应用 DPR 缩放（不含视口变换，用于 UI 层）
   */
  applyDPRScale(ctx: CanvasRenderingContext2D): void {
    ctx.scale(this.dpr, this.dpr);
  }

  /**
   * 屏幕坐标转画布坐标
   */
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number } {
    return {
      x: (screenX - this.viewport.x) / this.viewport.zoom,
      y: (screenY - this.viewport.y) / this.viewport.zoom,
    };
  }

  /**
   * 画布坐标转屏幕坐标
   */
  canvasToScreen(canvasX: number, canvasY: number): { x: number; y: number } {
    return {
      x: canvasX * this.viewport.zoom + this.viewport.x,
      y: canvasY * this.viewport.zoom + this.viewport.y,
    };
  }
}
