/**
 * WebGLContentRenderer - WebGL 内容层渲染器
 * 
 * 使用 WebGL 渲染图形，Canvas 2D 渲染文字。
 * 两层叠加合成最终效果。
 * 
 * 层级结构：
 * - 底层：WebGL Canvas（图形）
 * - 顶层：Canvas 2D（文字，透明背景）
 */

import type {
  RenderData,
  RenderText,
  Viewport,
} from '../../../core/RenderData';
import type { IContentRenderer } from './IContentRenderer';
import { WebGLGraphics } from './WebGLGraphics';
import { TextLODManager, type TextStrategy } from './TextLOD';
import { tokens } from '../../shared/styles';

/**
 * WebGL 内容层渲染器
 */
export class WebGLContentRenderer implements IContentRenderer {
  private container: HTMLElement | null = null;
  private webglCanvas: HTMLCanvasElement | null = null;
  private textCanvas: HTMLCanvasElement | null = null;
  private textCtx: CanvasRenderingContext2D | null = null;
  private webglGraphics: WebGLGraphics;
  private lodManager: TextLODManager;
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private width: number = 0;
  private height: number = 0;
  private dpr: number = 1;
  private initialized: boolean = false;

  constructor() {
    this.webglGraphics = new WebGLGraphics();
    this.lodManager = new TextLODManager();
  }

  /**
   * 初始化
   */
  async init(container: HTMLElement): Promise<boolean> {
    this.container = container;
    this.dpr = window.devicePixelRatio || 1;
    
    // 创建 WebGL Canvas（底层）
    this.webglCanvas = document.createElement('canvas');
    this.webglCanvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
    `;
    container.appendChild(this.webglCanvas);
    
    // 初始化 WebGL
    if (!(await this.webglGraphics.init(this.webglCanvas))) {
      console.warn('WebGL initialization failed, falling back to Canvas 2D');
      this.webglCanvas.remove();
      this.webglCanvas = null;
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
    this.webglGraphics.dispose();
    
    if (this.webglCanvas) {
      this.webglCanvas.remove();
      this.webglCanvas = null;
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
    
    // 更新 WebGL Canvas
    if (this.webglCanvas) {
      this.webglCanvas.width = this.width * this.dpr;
      this.webglCanvas.height = this.height * this.dpr;
      this.webglCanvas.style.width = `${this.width}px`;
      this.webglCanvas.style.height = `${this.height}px`;
      this.webglGraphics.resize(this.width, this.height);
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
    
    // 渲染图形（WebGL）
    this.renderGraphics(data);
    
    // 渲染文字（Canvas 2D）
    this.renderTexts(data.texts, textStrategy);
  }

  /**
   * 仅渲染图形（用于拖拽/缩放优化）
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
    this.webglGraphics.setViewport(this.viewport);
    this.webglGraphics.clear();
    
    // 渲染顺序：路径 → 矩形 → 圆形 → 三角形
    this.webglGraphics.renderPaths(data.paths);
    this.webglGraphics.renderRects(data.rects);
    this.webglGraphics.renderCircles(data.circles);
    this.webglGraphics.renderTriangles(data.triangles);
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
    
    // 应用 DPI 缩放
    ctx.scale(this.dpr, this.dpr);
    
    // 应用视口变换
    ctx.translate(this.viewport.x, this.viewport.y);
    ctx.scale(this.viewport.zoom, this.viewport.zoom);
    
    // 渲染文字
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
    ctx.font = `${fontSize}px ${text.fontFamily ?? tokens.text.fontFamily}`;
    ctx.fillStyle = text.color ?? tokens.text.title.color;
    ctx.textAlign = (text.align ?? 'left') as CanvasTextAlign;
    ctx.textBaseline = (text.baseline ?? 'top') as CanvasTextBaseline;
    ctx.fillText(text.text, text.x, text.y);
    
    ctx.restore();
  }
}
