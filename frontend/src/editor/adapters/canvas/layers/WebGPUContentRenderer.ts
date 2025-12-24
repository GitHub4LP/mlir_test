/**
 * WebGPUContentRenderer - WebGPU 内容层渲染器
 * 
 * 使用 WebGPU 渲染图形，Canvas 2D 渲染文字。
 * 与 WebGLContentRenderer 类似，但使用更现代的 WebGPU API。
 */

import type {
  RenderData,
  RenderText,
  Viewport,
} from '../../../core/RenderData';
import type { IContentRenderer } from './IContentRenderer';
import { WebGPUGraphics } from './WebGPUGraphics';
import { TextLODManager, type TextStrategy } from './TextLOD';

/**
 * WebGPU 内容层渲染器
 */
export class WebGPUContentRenderer implements IContentRenderer {
  private container: HTMLElement | null = null;
  private webgpuCanvas: HTMLCanvasElement | null = null;
  private textCanvas: HTMLCanvasElement | null = null;
  private textCtx: CanvasRenderingContext2D | null = null;
  private webgpuGraphics: WebGPUGraphics;
  private lodManager: TextLODManager;
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private width: number = 0;
  private height: number = 0;
  private dpr: number = 1;
  private initialized: boolean = false;

  constructor() {
    this.webgpuGraphics = new WebGPUGraphics();
    this.lodManager = new TextLODManager();
  }

  /**
   * 检查 WebGPU 是否可用
   */
  static isAvailable(): boolean {
    return WebGPUGraphics.isAvailable();
  }

  /**
   * 异步初始化
   */
  async init(container: HTMLElement): Promise<boolean> {
    if (!WebGPUContentRenderer.isAvailable()) {
      console.warn('WebGPU not available');
      return false;
    }

    this.container = container;
    this.dpr = window.devicePixelRatio || 1;

    // 创建 WebGPU Canvas（底层）
    this.webgpuCanvas = document.createElement('canvas');
    this.webgpuCanvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
    `;
    container.appendChild(this.webgpuCanvas);

    // 初始化 WebGPU
    const success = await this.webgpuGraphics.init(this.webgpuCanvas);
    if (!success) {
      console.warn('WebGPU initialization failed');
      this.webgpuCanvas.remove();
      this.webgpuCanvas = null;
      return false;
    }

    // 创建文字 Canvas（顶层）
    this.textCanvas = document.createElement('canvas');
    this.textCanvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
    `;
    container.appendChild(this.textCanvas);
    this.textCtx = this.textCanvas.getContext('2d');

    this.initialized = true;
    this.updateSize();

    return true;
  }

  /**
   * 销毁
   */
  dispose(): void {
    this.webgpuGraphics.dispose();

    if (this.webgpuCanvas) {
      this.webgpuCanvas.remove();
      this.webgpuCanvas = null;
    }

    if (this.textCanvas) {
      this.textCanvas.remove();
      this.textCanvas = null;
    }

    this.textCtx = null;
    this.container = null;
    this.initialized = false;
  }

  /**
   * 是否已初始化
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * 调整尺寸
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

    // 更新 WebGPU Canvas
    if (this.webgpuCanvas) {
      this.webgpuCanvas.width = this.width * this.dpr;
      this.webgpuCanvas.height = this.height * this.dpr;
      this.webgpuCanvas.style.width = `${this.width}px`;
      this.webgpuCanvas.style.height = `${this.height}px`;
      this.webgpuGraphics.resize(this.width, this.height);
    }

    // 更新文字 Canvas
    if (this.textCanvas) {
      this.textCanvas.width = this.width * this.dpr;
      this.textCanvas.height = this.height * this.dpr;
      this.textCanvas.style.width = `${this.width}px`;
      this.textCanvas.style.height = `${this.height}px`;
    }
  }

  /**
   * 渲染
   */
  render(data: RenderData): void {
    if (!this.initialized) return;

    this.viewport = { ...data.viewport };

    // 更新 LOD 策略
    this.lodManager.updateZoom(data.viewport.zoom);
    const textStrategy = this.lodManager.getStrategy();

    // 渲染图形（WebGPU）
    this.renderGraphics(data);

    // 渲染文字（Canvas 2D）
    this.renderTexts(data.texts, textStrategy);
  }

  /**
   * 仅渲染图形
   */
  renderGraphicsOnly(data: RenderData): void {
    if (!this.initialized) return;

    this.viewport = { ...data.viewport };
    this.renderGraphics(data);

    // 清空文字层
    if (this.textCtx && this.textCanvas) {
      this.textCtx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
    }
  }

  /**
   * 获取当前 LOD 级别
   */
  getLODLevel(): string {
    return this.lodManager.getLevel();
  }

  // ============================================================
  // 私有方法
  // ============================================================

  private renderGraphics(data: RenderData): void {
    this.webgpuGraphics.setViewport(this.viewport);
    this.webgpuGraphics.clear();

    // 渲染顺序：路径 → 矩形 → 圆形 → 三角形
    this.webgpuGraphics.renderPaths(data.paths);
    this.webgpuGraphics.renderRects(data.rects);
    this.webgpuGraphics.renderCircles(data.circles);
    this.webgpuGraphics.renderTriangles(data.triangles);
  }

  private renderTexts(texts: RenderText[], strategy: TextStrategy): void {
    if (!this.textCtx || !this.textCanvas) return;
    if (strategy.method === 'hidden') {
      this.textCtx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
      return;
    }

    const ctx = this.textCtx;
    ctx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
    ctx.save();

    ctx.scale(this.dpr, this.dpr);
    ctx.translate(this.viewport.x, this.viewport.y);
    ctx.scale(this.viewport.zoom, this.viewport.zoom);

    for (const text of texts) {
      if (this.shouldRenderText(text, strategy)) {
        this.renderText(ctx, text, strategy);
      }
    }

    ctx.restore();
  }

  private shouldRenderText(text: RenderText, strategy: TextStrategy): boolean {
    const id = text.id;
    if (id.includes('title')) return strategy.showTitle;
    if (id.includes('label')) return strategy.showLabels;
    if (id.includes('type')) return strategy.showTypes;
    if (id.includes('summary')) return strategy.showSummary;
    return strategy.showTitle;
  }

  private renderText(ctx: CanvasRenderingContext2D, text: RenderText, strategy: TextStrategy): void {
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const fontSize = (text.fontSize ?? 12) * strategy.fontScale;
    ctx.font = `${fontSize}px ${text.fontFamily ?? 'system-ui, sans-serif'}`;
    ctx.fillStyle = text.color ?? '#ffffff';
    ctx.textAlign = (text.align ?? 'left') as CanvasTextAlign;
    ctx.textBaseline = (text.baseline ?? 'top') as CanvasTextBaseline;
    ctx.fillText(text.text, text.x, text.y);

    ctx.restore();
  }
}
