/**
 * ContentRenderer - 内容层渲染器
 * 
 * 负责渲染节点、连线、端口、文字。
 * 分离图形渲染和文字渲染，支持 LOD 策略。
 * 
 * 设计：
 * - 图形渲染：可切换 Canvas 2D / WebGL / WebGPU
 * - 文字渲染：始终使用 Canvas 2D + LOD 策略
 * - 样式：从 Design Tokens 统一获取
 */

import type {
  RenderData,
  RenderRect,
  RenderText,
  RenderPath,
  RenderCircle,
  RenderTriangle,
} from '../../../core/RenderData';
import type { IContentRenderer } from './IContentRenderer';
import { TextLODManager, type TextStrategy } from './TextLOD';
import { tokens, LAYOUT, TEXT } from '../../shared/styles';

/** 图形渲染器类型 */
export type GraphicsBackend = 'canvas2d' | 'webgl' | 'webgpu';

/** ContentRenderer 配置 */
export interface ContentRendererConfig {
  /** 图形后端类型 */
  graphicsBackend: GraphicsBackend;
  /** 是否启用文字缓存 */
  enableTextCache: boolean;
}

const DEFAULT_CONFIG: ContentRendererConfig = {
  graphicsBackend: 'canvas2d',
  enableTextCache: false,
};

/**
 * 内容层渲染器（Canvas 2D 实现）
 */
export class ContentRenderer implements IContentRenderer {
  private container: HTMLElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private lodManager: TextLODManager;
  private dpr: number = 1;
  private width: number = 0;
  private height: number = 0;
  private initialized: boolean = false;
  
  // 配置（预留给未来扩展）
  readonly graphicsBackend: GraphicsBackend;
  readonly enableTextCache: boolean;

  constructor(config: Partial<ContentRendererConfig> = {}) {
    const merged = { ...DEFAULT_CONFIG, ...config };
    this.graphicsBackend = merged.graphicsBackend;
    this.enableTextCache = merged.enableTextCache;
    this.lodManager = new TextLODManager();
  }

  /**
   * 初始化渲染器
   */
  async init(container: HTMLElement): Promise<boolean> {
    this.container = container;
    this.dpr = window.devicePixelRatio || 1;

    // 创建 Canvas
    this.canvas = document.createElement('canvas');
    this.canvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
    `;
    container.appendChild(this.canvas);

    this.ctx = this.canvas.getContext('2d');
    if (!this.ctx) {
      console.warn('Canvas 2D context not available');
      this.canvas.remove();
      this.canvas = null;
      return false;
    }

    this.initialized = true;
    this.updateSize();
    return true;
  }

  /**
   * 销毁渲染器
   */
  dispose(): void {
    if (this.canvas) {
      this.canvas.remove();
      this.canvas = null;
    }
    this.ctx = null;
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
    if (!this.container || !this.canvas) return;

    const rect = this.container.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    this.dpr = window.devicePixelRatio || 1;

    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
  }

  /**
   * 绑定到 Canvas（兼容旧 API）
   */
  bind(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.initialized = true;
  }

  /**
   * 解绑（兼容旧 API）
   */
  unbind(): void {
    this.canvas = null;
    this.ctx = null;
    this.initialized = false;
  }

  /**
   * 更新 DPR
   */
  setDPR(dpr: number): void {
    this.dpr = dpr;
  }

  /**
   * 渲染内容层
   */
  render(data: RenderData): void {
    if (!this.ctx || !this.canvas) return;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.save();

    // 应用 DPI 缩放
    ctx.scale(this.dpr, this.dpr);

    // 应用视口变换
    ctx.translate(data.viewport.x, data.viewport.y);
    ctx.scale(data.viewport.zoom, data.viewport.zoom);

    // 更新 LOD 策略
    this.lodManager.updateZoom(data.viewport.zoom);
    const textStrategy = this.lodManager.getStrategy();

    // 渲染顺序：连线 → 节点 → 端口 → 文字
    this.renderPaths(ctx, data.paths);
    this.renderRects(ctx, data.rects);
    this.renderCircles(ctx, data.circles);
    this.renderTriangles(ctx, data.triangles);
    this.renderTexts(ctx, data.texts, textStrategy);

    ctx.restore();
  }

  /**
   * 仅渲染图形（不含文字）
   * 用于拖拽/缩放过程中的快速渲染
   */
  renderGraphicsOnly(data: RenderData): void {
    if (!this.ctx || !this.canvas) return;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.save();

    ctx.scale(this.dpr, this.dpr);
    ctx.translate(data.viewport.x, data.viewport.y);
    ctx.scale(data.viewport.zoom, data.viewport.zoom);

    this.renderPaths(ctx, data.paths);
    this.renderRects(ctx, data.rects);
    this.renderCircles(ctx, data.circles);
    this.renderTriangles(ctx, data.triangles);

    ctx.restore();
  }

  /**
   * 获取当前 LOD 策略
   */
  getTextStrategy(): TextStrategy {
    return this.lodManager.getStrategy();
  }

  /**
   * 获取当前 LOD 级别
   */
  getLODLevel(): string {
    return this.lodManager.getLevel();
  }

  // ============================================================
  // 图形渲染方法
  // ============================================================

  private renderPaths(ctx: CanvasRenderingContext2D, paths: RenderPath[]): void {
    for (const path of paths) {
      this.renderPath(ctx, path);
    }
  }

  private renderPath(ctx: CanvasRenderingContext2D, path: RenderPath): void {
    if (path.points.length < 2) return;

    ctx.save();
    ctx.strokeStyle = path.color ?? '#ffffff';
    ctx.lineWidth = path.width ?? 2;

    if (path.dashed && path.dashPattern) {
      ctx.setLineDash(path.dashPattern);
    }

    ctx.beginPath();

    if (path.points.length === 4) {
      // 贝塞尔曲线
      ctx.moveTo(path.points[0].x, path.points[0].y);
      ctx.bezierCurveTo(
        path.points[1].x, path.points[1].y,
        path.points[2].x, path.points[2].y,
        path.points[3].x, path.points[3].y
      );
    } else {
      // 折线
      ctx.moveTo(path.points[0].x, path.points[0].y);
      for (let i = 1; i < path.points.length; i++) {
        ctx.lineTo(path.points[i].x, path.points[i].y);
      }
    }

    ctx.stroke();

    // 箭头
    if (path.arrowEnd && path.points.length >= 2) {
      const lastIdx = path.points.length - 1;
      const end = path.points[lastIdx];
      const prev = path.points[lastIdx - 1];
      this.renderArrow(ctx, prev.x, prev.y, end.x, end.y, path.color ?? '#ffffff');
    }

    ctx.restore();
  }

  private renderRects(ctx: CanvasRenderingContext2D, rects: RenderRect[]): void {
    // 按 zIndex 排序
    const sorted = [...rects].sort((a, b) => (a.zIndex ?? 0) - (b.zIndex ?? 0));
    for (const rect of sorted) {
      this.renderRect(ctx, rect);
    }
  }

  private renderRect(ctx: CanvasRenderingContext2D, rect: RenderRect): void {
    ctx.save();

    if (rect.fillColor && rect.fillColor !== 'transparent') {
      ctx.fillStyle = rect.fillColor;
      this.roundRect(ctx, rect.x, rect.y, rect.width, rect.height, rect.borderRadius ?? LAYOUT.borderRadius);
      ctx.fill();
    }

    if (rect.borderWidth && rect.borderWidth > 0 && rect.borderColor && rect.borderColor !== 'transparent') {
      ctx.strokeStyle = rect.borderColor;
      ctx.lineWidth = rect.borderWidth;
      this.roundRect(ctx, rect.x, rect.y, rect.width, rect.height, rect.borderRadius ?? LAYOUT.borderRadius);
      ctx.stroke();
    }

    // 选中高亮 - 使用 tokens 颜色
    if (rect.selected) {
      ctx.strokeStyle = tokens.node.selected.borderColor;
      ctx.lineWidth = tokens.node.selected.borderWidth;
      const selectionRadius = typeof rect.borderRadius === 'number'
        ? (rect.borderRadius ?? 0) + 2
        : {
            topLeft: (rect.borderRadius?.topLeft ?? 0) + 2,
            topRight: (rect.borderRadius?.topRight ?? 0) + 2,
            bottomLeft: (rect.borderRadius?.bottomLeft ?? 0) + 2,
            bottomRight: (rect.borderRadius?.bottomRight ?? 0) + 2,
          };
      this.roundRect(ctx, rect.x - 2, rect.y - 2, rect.width + 4, rect.height + 4, selectionRadius);
      ctx.stroke();
    }

    ctx.restore();
  }

  private renderCircles(ctx: CanvasRenderingContext2D, circles: RenderCircle[]): void {
    for (const circle of circles) {
      this.renderCircle(ctx, circle);
    }
  }

  private renderCircle(ctx: CanvasRenderingContext2D, circle: RenderCircle): void {
    ctx.save();
    ctx.beginPath();
    ctx.arc(circle.x, circle.y, circle.radius, 0, Math.PI * 2);

    if (circle.fillColor && circle.fillColor !== 'transparent') {
      ctx.fillStyle = circle.fillColor;
      ctx.fill();
    }

    if (circle.borderWidth && circle.borderWidth > 0 && circle.borderColor) {
      ctx.strokeStyle = circle.borderColor;
      ctx.lineWidth = circle.borderWidth;
      ctx.stroke();
    }

    ctx.restore();
  }

  private renderTriangles(ctx: CanvasRenderingContext2D, triangles: RenderTriangle[]): void {
    for (const triangle of triangles) {
      this.renderTriangle(ctx, triangle);
    }
  }

  private renderTriangle(ctx: CanvasRenderingContext2D, triangle: RenderTriangle): void {
    ctx.save();
    ctx.beginPath();

    const { x, y, size, direction } = triangle;

    if (direction === 'right') {
      ctx.moveTo(x - size * 0.5, y - size * 0.6);
      ctx.lineTo(x - size * 0.5, y + size * 0.6);
      ctx.lineTo(x + size * 0.7, y);
    } else {
      ctx.moveTo(x + size * 0.5, y - size * 0.6);
      ctx.lineTo(x + size * 0.5, y + size * 0.6);
      ctx.lineTo(x - size * 0.7, y);
    }

    ctx.closePath();

    if (triangle.fillColor && triangle.fillColor !== 'transparent') {
      ctx.fillStyle = triangle.fillColor;
      ctx.fill();
    }

    if (triangle.borderWidth && triangle.borderWidth > 0 && triangle.borderColor) {
      ctx.strokeStyle = triangle.borderColor;
      ctx.lineWidth = triangle.borderWidth;
      ctx.stroke();
    }

    ctx.restore();
  }

  // ============================================================
  // 文字渲染方法（带 LOD）
  // ============================================================

  private renderTexts(ctx: CanvasRenderingContext2D, texts: RenderText[], strategy: TextStrategy): void {
    if (strategy.method === 'hidden') return;

    for (const text of texts) {
      if (this.shouldRenderText(text, strategy)) {
        this.renderText(ctx, text, strategy);
      }
    }
  }

  private shouldRenderText(text: RenderText, strategy: TextStrategy): boolean {
    // 根据文字 ID 前缀判断类型
    const id = text.id;
    if (id.includes('title')) return strategy.showTitle;
    if (id.includes('label')) return strategy.showLabels;
    if (id.includes('type')) return strategy.showTypes;
    if (id.includes('summary')) return strategy.showSummary;
    // 默认显示
    return strategy.showTitle;
  }

  private renderText(ctx: CanvasRenderingContext2D, text: RenderText, strategy: TextStrategy): void {
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    const fontSize = (text.fontSize ?? TEXT.labelSize) * strategy.fontScale;
    ctx.font = `${fontSize}px ${text.fontFamily ?? TEXT.fontFamily}`;
    ctx.fillStyle = text.color ?? TEXT.labelColor;
    ctx.textAlign = (text.align ?? 'left') as CanvasTextAlign;
    ctx.textBaseline = (text.baseline ?? 'top') as CanvasTextBaseline;
    ctx.fillText(text.text, text.x, text.y);

    ctx.restore();
  }

  // ============================================================
  // 辅助方法
  // ============================================================

  private roundRect(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, width: number, height: number,
    radius: number | { topLeft: number; topRight: number; bottomLeft: number; bottomRight: number }
  ): void {
    let tl: number, tr: number, bl: number, br: number;
    if (typeof radius === 'number') {
      tl = tr = bl = br = radius;
    } else {
      tl = radius.topLeft;
      tr = radius.topRight;
      bl = radius.bottomLeft;
      br = radius.bottomRight;
    }

    if (tl === 0 && tr === 0 && bl === 0 && br === 0) {
      ctx.rect(x, y, width, height);
      return;
    }

    ctx.beginPath();
    ctx.moveTo(x + tl, y);
    ctx.lineTo(x + width - tr, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + tr);
    ctx.lineTo(x + width, y + height - br);
    ctx.quadraticCurveTo(x + width, y + height, x + width - br, y + height);
    ctx.lineTo(x + bl, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - bl);
    ctx.lineTo(x, y + tl);
    ctx.quadraticCurveTo(x, y, x + tl, y);
    ctx.closePath();
  }

  private renderArrow(
    ctx: CanvasRenderingContext2D,
    fromX: number, fromY: number, toX: number, toY: number, color: string
  ): void {
    const headLength = 10;
    const angle = Math.atan2(toY - fromY, toX - fromX);

    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(
      toX - headLength * Math.cos(angle - Math.PI / 6),
      toY - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      toX - headLength * Math.cos(angle + Math.PI / 6),
      toY - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}
