/**
 * Canvas 2D 渲染后端
 * 
 * 使用 HTML5 Canvas 2D API 渲染图形。
 * 适用于需要高性能渲染大量节点的场景。
 * 
 * 特点：
 * - 完全自定义渲染，不依赖 DOM 节点
 * - 支持高 DPI 显示
 * - 需要 OverlayManager 来处理属性编辑器
 * - 样式从 StyleSystem 统一获取
 */

import type { IRenderer } from './IRenderer';
import type {
  RenderData,
  Viewport,
  RenderRect,
  RenderText,
  RenderPath,
  RenderCircle,
  RenderTriangle,
  InteractionHint,
} from './types';
import type { RawInputCallback, MouseButton } from './input';
import { createPointerInput, createWheelInput, createKeyInput, extractModifiers } from './input';
import { StyleSystem } from '../../core/StyleSystem';

/**
 * Canvas 2D 渲染后端
 */
export class CanvasRenderer implements IRenderer {
  private container: HTMLElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private inputCallback: RawInputCallback | null = null;
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private width: number = 0;
  private height: number = 0;
  private dpr: number = 1;

  // 事件处理器引用（用于清理）
  private boundHandlePointerDown: ((e: PointerEvent) => void) | null = null;
  private boundHandlePointerMove: ((e: PointerEvent) => void) | null = null;
  private boundHandlePointerUp: ((e: PointerEvent) => void) | null = null;
  private boundHandleWheel: ((e: WheelEvent) => void) | null = null;
  private boundHandleKeyDown: ((e: KeyboardEvent) => void) | null = null;
  private boundHandleKeyUp: ((e: KeyboardEvent) => void) | null = null;
  private boundHandleDragOver: ((e: DragEvent) => void) | null = null;
  private boundHandleDrop: ((e: DragEvent) => void) | null = null;

  // 拖放回调
  private dropCallback: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null = null;

  // UI 渲染回调（用于在图内容之上渲染 Canvas UI 组件）
  private uiRenderCallback: ((ctx: CanvasRenderingContext2D) => void) | null = null;

  // IRenderer 接口实现

  mount(container: HTMLElement): void {
    this.container = container;
    
    // 创建 canvas 元素
    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.display = 'block';
    this.canvas.tabIndex = 0;
    this.canvas.style.outline = 'none';
    
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.updateSize();
    this.bindEvents();
  }

  unmount(): void {
    this.unbindEvents();
    if (this.canvas && this.container) {
      this.container.removeChild(this.canvas);
    }
    this.canvas = null;
    this.ctx = null;
    this.container = null;
  }

  resize(): void {
    this.updateSize();
  }

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
    
    // 按 zIndex 排序
    const sortedRects = [...data.rects].sort((a, b) => (a.zIndex ?? 0) - (b.zIndex ?? 0));
    
    // 渲染边
    for (const path of data.paths) {
      this.renderPath(ctx, path);
    }
    
    // 渲染节点
    for (const rect of sortedRects) {
      this.renderRect(ctx, rect);
    }
    
    // 渲染文字
    for (const text of data.texts) {
      this.renderText(ctx, text);
    }
    
    // 渲染端口
    for (const circle of data.circles) {
      this.renderCircle(ctx, circle);
    }
    
    // 渲染执行引脚（三角形）
    for (const triangle of data.triangles) {
      this.renderTriangle(ctx, triangle);
    }
    
    // 渲染交互提示
    this.renderHint(ctx, data.hint);
    
    ctx.restore();
    
    // 渲染 UI 层（在视口变换之外，使用屏幕坐标）
    if (this.uiRenderCallback) {
      ctx.save();
      ctx.scale(this.dpr, this.dpr);
      this.uiRenderCallback(ctx);
      ctx.restore();
    }
    
    this.viewport = { ...data.viewport };
  }

  onInput(callback: RawInputCallback): void {
    this.inputCallback = callback;
  }

  getName(): string {
    return 'Canvas2D';
  }

  isAvailable(): boolean {
    return typeof document !== 'undefined' && 
           typeof document.createElement('canvas').getContext === 'function';
  }

  getViewport(): Viewport {
    return { ...this.viewport };
  }

  setViewport(viewport: Viewport): void {
    this.viewport = { ...viewport };
  }

  // 私有方法

  private updateSize(): void {
    if (!this.canvas || !this.container) return;
    
    const rect = this.container.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    this.dpr = window.devicePixelRatio || 1;
    
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
  }

  private bindEvents(): void {
    if (!this.canvas) return;
    
    this.boundHandlePointerDown = this.handlePointerDown.bind(this);
    this.boundHandlePointerMove = this.handlePointerMove.bind(this);
    this.boundHandlePointerUp = this.handlePointerUp.bind(this);
    this.boundHandleWheel = this.handleWheel.bind(this);
    this.boundHandleKeyDown = this.handleKeyDown.bind(this);
    this.boundHandleKeyUp = this.handleKeyUp.bind(this);
    this.boundHandleDragOver = this.handleDragOver.bind(this);
    this.boundHandleDrop = this.handleDrop.bind(this);
    
    this.canvas.addEventListener('pointerdown', this.boundHandlePointerDown);
    this.canvas.addEventListener('pointermove', this.boundHandlePointerMove);
    this.canvas.addEventListener('pointerup', this.boundHandlePointerUp);
    this.canvas.addEventListener('wheel', this.boundHandleWheel, { passive: false });
    this.canvas.addEventListener('keydown', this.boundHandleKeyDown);
    this.canvas.addEventListener('keyup', this.boundHandleKeyUp);
    this.canvas.addEventListener('dragover', this.boundHandleDragOver);
    this.canvas.addEventListener('drop', this.boundHandleDrop);
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  private unbindEvents(): void {
    if (!this.canvas) return;
    
    if (this.boundHandlePointerDown) {
      this.canvas.removeEventListener('pointerdown', this.boundHandlePointerDown);
    }
    if (this.boundHandlePointerMove) {
      this.canvas.removeEventListener('pointermove', this.boundHandlePointerMove);
    }
    if (this.boundHandlePointerUp) {
      this.canvas.removeEventListener('pointerup', this.boundHandlePointerUp);
    }
    if (this.boundHandleWheel) {
      this.canvas.removeEventListener('wheel', this.boundHandleWheel);
    }
    if (this.boundHandleKeyDown) {
      this.canvas.removeEventListener('keydown', this.boundHandleKeyDown);
    }
    if (this.boundHandleKeyUp) {
      this.canvas.removeEventListener('keyup', this.boundHandleKeyUp);
    }
    if (this.boundHandleDragOver) {
      this.canvas.removeEventListener('dragover', this.boundHandleDragOver);
    }
    if (this.boundHandleDrop) {
      this.canvas.removeEventListener('drop', this.boundHandleDrop);
    }
  }

  /**
   * 设置拖放回调
   */
  setDropCallback(callback: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null): void {
    this.dropCallback = callback;
  }

  /**
   * 设置 UI 渲染回调（用于在图内容之上渲染 Canvas UI 组件）
   */
  setUIRenderCallback(callback: ((ctx: CanvasRenderingContext2D) => void) | null): void {
    this.uiRenderCallback = callback;
  }

  private handleDragOver(e: DragEvent): void {
    e.preventDefault();
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = 'copy';
    }
  }

  private handleDrop(e: DragEvent): void {
    e.preventDefault();
    if (!this.canvas || !e.dataTransfer) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    
    // 转换为画布坐标
    const canvasX = (screenX - this.viewport.x) / this.viewport.zoom;
    const canvasY = (screenY - this.viewport.y) / this.viewport.zoom;
    
    this.dropCallback?.(canvasX, canvasY, e.dataTransfer);
  }


  // 事件处理 - 传递屏幕坐标给 Controller

  private handlePointerDown(e: PointerEvent): void {
    if (!this.canvas) return;
    this.canvas.focus();
    
    const rect = this.canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    
    const input = createPointerInput(
      'down',
      screenX,
      screenY,
      e.button as MouseButton,
      extractModifiers(e)
    );
    this.inputCallback?.(input);
  }

  private handlePointerMove(e: PointerEvent): void {
    if (!this.canvas) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    
    const input = createPointerInput(
      'move',
      screenX,
      screenY,
      e.button as MouseButton,
      extractModifiers(e)
    );
    this.inputCallback?.(input);
  }

  private handlePointerUp(e: PointerEvent): void {
    if (!this.canvas) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    
    const input = createPointerInput(
      'up',
      screenX,
      screenY,
      e.button as MouseButton,
      extractModifiers(e)
    );
    this.inputCallback?.(input);
  }

  private handleWheel(e: WheelEvent): void {
    e.preventDefault();
    if (!this.canvas) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    
    const input = createWheelInput(
      e.deltaX,
      e.deltaY,
      screenX,
      screenY,
      extractModifiers(e)
    );
    this.inputCallback?.(input);
  }

  private handleKeyDown(e: KeyboardEvent): void {
    const input = createKeyInput('down', e.key, e.code, extractModifiers(e));
    this.inputCallback?.(input);
  }

  private handleKeyUp(e: KeyboardEvent): void {
    const input = createKeyInput('up', e.key, e.code, extractModifiers(e));
    this.inputCallback?.(input);
  }

  // 渲染方法

  private renderRect(ctx: CanvasRenderingContext2D, rect: RenderRect): void {
    ctx.save();
    const nodeStyle = StyleSystem.getNodeStyle();
    
    if (rect.fillColor && rect.fillColor !== 'transparent') {
      ctx.fillStyle = rect.fillColor;
      this.roundRect(ctx, rect.x, rect.y, rect.width, rect.height, rect.borderRadius ?? nodeStyle.borderRadius);
      ctx.fill();
    }
    
    if (rect.borderWidth && rect.borderWidth > 0 && rect.borderColor && rect.borderColor !== 'transparent') {
      ctx.strokeStyle = rect.borderColor;
      ctx.lineWidth = rect.borderWidth;
      this.roundRect(ctx, rect.x, rect.y, rect.width, rect.height, rect.borderRadius ?? nodeStyle.borderRadius);
      ctx.stroke();
    }
    
    if (rect.selected) {
      ctx.strokeStyle = nodeStyle.selectedBorderColor;
      ctx.lineWidth = nodeStyle.selectedBorderWidth;
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

  private renderText(ctx: CanvasRenderingContext2D, text: RenderText): void {
    ctx.save();
    const textStyle = StyleSystem.getTextStyle();
    // 启用高质量文字渲染
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.font = `${text.fontSize ?? textStyle.labelFontSize}px ${text.fontFamily ?? textStyle.fontFamily}`;
    ctx.fillStyle = text.color ?? textStyle.labelColor;
    ctx.textAlign = (text.align ?? 'left') as CanvasTextAlign;
    ctx.textBaseline = (text.baseline ?? 'top') as CanvasTextBaseline;
    ctx.fillText(text.text, text.x, text.y);
    ctx.restore();
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
      ctx.moveTo(path.points[0].x, path.points[0].y);
      ctx.bezierCurveTo(
        path.points[1].x, path.points[1].y,
        path.points[2].x, path.points[2].y,
        path.points[3].x, path.points[3].y
      );
    } else {
      ctx.moveTo(path.points[0].x, path.points[0].y);
      for (let i = 1; i < path.points.length; i++) {
        ctx.lineTo(path.points[i].x, path.points[i].y);
      }
    }
    
    ctx.stroke();
    
    if (path.arrowEnd && path.points.length >= 2) {
      const lastIdx = path.points.length - 1;
      const end = path.points[lastIdx];
      const prev = path.points[lastIdx - 1];
      this.renderArrow(ctx, prev.x, prev.y, end.x, end.y, path.color ?? '#ffffff');
    }
    
    ctx.restore();
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

  private renderTriangle(ctx: CanvasRenderingContext2D, triangle: RenderTriangle): void {
    ctx.save();
    ctx.beginPath();
    
    const { x, y, size, direction } = triangle;
    
    if (direction === 'right') {
      // 向右指的三角形（执行输出引脚）
      // 三个顶点：左上、左下、右中
      ctx.moveTo(x - size * 0.5, y - size * 0.6);
      ctx.lineTo(x - size * 0.5, y + size * 0.6);
      ctx.lineTo(x + size * 0.7, y);
    } else {
      // 向左指的三角形（执行输入引脚）
      // 三个顶点：右上、右下、左中
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

  private renderHint(ctx: CanvasRenderingContext2D, hint: InteractionHint): void {
    if (hint.connectionPreview) {
      this.renderPath(ctx, hint.connectionPreview);
    }
    if (hint.selectionBox) {
      this.renderRect(ctx, hint.selectionBox);
    }
    if (hint.dragPreview) {
      ctx.save();
      ctx.globalAlpha = 0.5;
      this.renderRect(ctx, hint.dragPreview);
      ctx.restore();
    }
  }

  private roundRect(
    ctx: CanvasRenderingContext2D,
    x: number, y: number, width: number, height: number, 
    radius: number | { topLeft: number; topRight: number; bottomLeft: number; bottomRight: number }
  ): void {
    // 解析圆角配置
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

export default CanvasRenderer;
