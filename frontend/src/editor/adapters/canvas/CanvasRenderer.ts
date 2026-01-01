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
 * - 样式从 Design Tokens 统一获取
 */

import type { IRenderer } from './IRenderer';
import type {
  RenderData,
  Viewport,
  RenderRect,
  RenderPath,
  InteractionHint,
} from './types';
import type { RawInputCallback, MouseButton } from './input';
import { createPointerInput, createWheelInput, createKeyInput, extractModifiers } from './input';
import { tokens, LAYOUT } from '../shared/styles';
import type { LayoutBox, CornerRadius } from '../../core/layout/types';
import { layoutConfig } from '../../core/layout/LayoutConfig';

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
    
    // 渲染边
    for (const path of data.paths) {
      this.renderPath(ctx, path);
    }
    
    // 使用 LayoutBox 系统渲染节点
    if (data.layoutBoxes && data.layoutBoxes.size > 0) {
      this.renderWithLayoutBoxes(ctx, data);
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

  /**
   * 使用 LayoutBox 系统渲染节点
   */
  private renderWithLayoutBoxes(ctx: CanvasRenderingContext2D, data: RenderData): void {
    if (!data.layoutBoxes) return;
    
    // 构建节点信息映射（仅需要 selected 和 zIndex）
    const nodeInfoMap = new Map<string, {
      selected: boolean;
      zIndex: number;
    }>();
    
    for (const rect of data.rects) {
      if (rect.id.startsWith('rect-')) {
        const nodeId = rect.id.slice(5);
        nodeInfoMap.set(nodeId, {
          selected: rect.selected,
          zIndex: rect.zIndex,
        });
      }
    }
    
    // 按 zIndex 排序
    const sortedEntries = [...data.layoutBoxes.entries()].sort((a, b) => {
      const aInfo = nodeInfoMap.get(a[0]);
      const bInfo = nodeInfoMap.get(b[0]);
      return (aInfo?.zIndex ?? 0) - (bInfo?.zIndex ?? 0);
    });
    
    for (const [nodeId, layoutBox] of sortedEntries) {
      const nodeInfo = nodeInfoMap.get(nodeId);
      const selected = nodeInfo?.selected ?? false;
      
      this.renderLayoutBoxTree(ctx, layoutBox, 0, 0, selected);
    }
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

  private renderPath(ctx: CanvasRenderingContext2D, path: RenderPath): void {
    if (path.points.length < 2) return;
    
    ctx.save();
    ctx.strokeStyle = path.color ?? tokens.edge.exec.color;
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
      this.renderArrow(ctx, prev.x, prev.y, end.x, end.y, path.color ?? tokens.edge.exec.color);
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
      ctx.beginPath();
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

  // ============================================================================
  // LayoutBox 渲染方法
  // ============================================================================

  /**
   * 渲染 LayoutBox 树
   * @deprecated 内部使用 renderLayoutBoxTree
   */
  renderLayoutBox(
    ctx: CanvasRenderingContext2D,
    box: LayoutBox,
    offsetX: number,
    offsetY: number,
    selected: boolean = false
  ): void {
    this.renderLayoutBoxTree(ctx, box, offsetX, offsetY, selected);
  }

  /**
   * 渲染 LayoutBox 树
   * @param ctx - Canvas 2D 上下文
   * @param box - LayoutBox
   * @param offsetX - X 偏移
   * @param offsetY - Y 偏移
   * @param selected - 是否选中
   */
  private renderLayoutBoxTree(
    ctx: CanvasRenderingContext2D,
    box: LayoutBox,
    offsetX: number,
    offsetY: number,
    selected: boolean
  ): void {
    const absX = offsetX + box.x;
    const absY = offsetY + box.y;

    ctx.save();

    // 1. 渲染背景和边框
    if (box.style || (selected && box.type === 'node')) {
      this.renderLayoutBoxBackground(ctx, box, absX, absY, selected);
    }

    // 2. 渲染特殊元素（Handle、TypeLabel 等）
    this.renderLayoutBoxSpecial(ctx, box, absX, absY);

    // 3. 渲染文本
    if (box.text) {
      this.renderLayoutBoxText(ctx, box, absX, absY);
    }

    // 4. 递归渲染子节点
    for (const child of box.children) {
      this.renderLayoutBoxTree(ctx, child, absX, absY, false);
    }

    ctx.restore();
  }

  /**
   * 渲染 LayoutBox 背景
   */
  private renderLayoutBoxBackground(
    ctx: CanvasRenderingContext2D,
    box: LayoutBox,
    x: number,
    y: number,
    selected: boolean
  ): void {
    const style = box.style;
    const radius = this.normalizeCornerRadius(style?.cornerRadius);

    // 所有元素统一使用 style 中的颜色（包括 headerContent）
    if (style) {
      // 填充背景
      if (style.fill && style.fill !== 'transparent') {
        ctx.fillStyle = style.fill;
        this.roundRect(ctx, x, y, box.width, box.height, radius);
        ctx.fill();
      }

      // 绘制边框
      if (style.stroke && style.strokeWidth && style.strokeWidth > 0) {
        ctx.strokeStyle = style.stroke;
        ctx.lineWidth = style.strokeWidth;
        this.roundRect(ctx, x, y, box.width, box.height, radius);
        ctx.stroke();
      }
    }

    // 选中状态边框（仅 node 类型）
    if (selected && box.type === 'node') {
      const selectedConfig = layoutConfig.node.selected;
      if (selectedConfig) {
        ctx.strokeStyle = selectedConfig.stroke ?? '#60a5fa';
        ctx.lineWidth = selectedConfig.strokeWidth ?? 2;
        const selectionRadius = this.expandCornerRadius(radius, 2);
        this.roundRect(ctx, x - 2, y - 2, box.width + 4, box.height + 4, selectionRadius);
        ctx.stroke();
      }
    }
  }

  /**
   * 渲染特殊元素
   */
  private renderLayoutBoxSpecial(
    ctx: CanvasRenderingContext2D,
    box: LayoutBox,
    x: number,
    y: number
  ): void {
    const id = box.interactive?.id;

    // Handle 渲染
    if (box.type === 'handle' && id) {
      this.renderHandle(ctx, box, x, y, id);
      return;
    }

    // TypeLabel 背景渲染
    if (box.type === 'typeLabel') {
      this.renderTypeLabel(ctx, box, x, y);
      return;
    }
  }

  /**
   * 渲染 Handle
   */
  private renderHandle(
    ctx: CanvasRenderingContext2D,
    box: LayoutBox,
    x: number,
    y: number,
    id: string
  ): void {
    const handleConfig = layoutConfig.handle;
    const size = typeof handleConfig.width === 'number' ? handleConfig.width : 12;
    const strokeWidth = handleConfig.strokeWeight ?? 2;
    const centerX = x + box.width / 2;
    const centerY = y + box.height / 2;

    // 判断是执行端口还是数据端口
    const isExec = id.includes('exec');

    if (isExec) {
      // 执行端口：三角形（白色）
      this.renderExecHandle(ctx, centerX, centerY, size, 'right', strokeWidth);
    } else {
      // 数据端口：圆形，从 LayoutBox.interactive.pinColor 获取颜色
      const color = box.interactive?.pinColor ?? layoutConfig.nodeType.operation;
      this.renderDataHandle(ctx, centerX, centerY, size / 2, color, strokeWidth);
    }
  }

  /**
   * 渲染 TypeLabel
   */
  private renderTypeLabel(
    ctx: CanvasRenderingContext2D,
    box: LayoutBox,
    x: number,
    y: number
  ): void {
    const config = layoutConfig.typeLabel;
    const radius = this.normalizeCornerRadius(config.cornerRadius);

    // 从 LayoutBox.interactive.pinColor 获取颜色
    const pinColor = box.interactive?.pinColor;
    
    // 背景颜色：使用 Handle 颜色的半透明版本，或默认灰色
    const bgColor = pinColor 
      ? this.colorWithAlpha(pinColor, 0.3)
      : (config.fill ?? 'rgba(100, 100, 100, 0.5)');

    ctx.fillStyle = bgColor;
    this.roundRect(ctx, x, y, box.width, box.height, radius);
    ctx.fill();

    // 文本
    if (box.text) {
      this.renderLayoutBoxText(ctx, box, x, y);
    }
  }

  /**
   * 将颜色转换为带透明度的版本
   */
  private colorWithAlpha(color: string, alpha: number): string {
    // 处理 hex 颜色
    if (color.startsWith('#')) {
      const hex = color.slice(1);
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    // 处理 rgb/rgba
    if (color.startsWith('rgb')) {
      const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (match) {
        return `rgba(${match[1]}, ${match[2]}, ${match[3]}, ${alpha})`;
      }
    }
    return color;
  }

  /**
   * 渲染执行端口（三角形）
   * 使用与 renderTriangle 相同的尺寸计算逻辑
   */
  private renderExecHandle(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    size: number,
    direction: 'left' | 'right',
    strokeWidth: number
  ): void {
    ctx.save();
    ctx.beginPath();

    // 与 renderTriangle 和 WebGL/WebGPU 渲染器一致的尺寸
    const scaledSize = size * 0.6;
    if (direction === 'right') {
      ctx.moveTo(x - scaledSize * 0.5, y - scaledSize * 0.6);
      ctx.lineTo(x - scaledSize * 0.5, y + scaledSize * 0.6);
      ctx.lineTo(x + scaledSize * 0.7, y);
    } else {
      ctx.moveTo(x + scaledSize * 0.5, y - scaledSize * 0.6);
      ctx.lineTo(x + scaledSize * 0.5, y + scaledSize * 0.6);
      ctx.lineTo(x - scaledSize * 0.7, y);
    }
    ctx.closePath();

    // 白色填充和边框
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = strokeWidth;
    ctx.stroke();

    ctx.restore();
  }

  /**
   * 渲染数据端口（圆形）
   */
  private renderDataHandle(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    radius: number,
    color: string,
    strokeWidth: number
  ): void {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);

    // 填充颜色
    ctx.fillStyle = color;
    ctx.fill();

    // 边框：使用节点背景色，与 GPURenderer 一致
    ctx.strokeStyle = tokens.node.bg;
    ctx.lineWidth = strokeWidth;
    ctx.stroke();

    ctx.restore();
  }

  /**
   * 渲染 LayoutBox 文本
   */
  private renderLayoutBoxText(
    ctx: CanvasRenderingContext2D,
    box: LayoutBox,
    x: number,
    y: number
  ): void {
    const text = box.text!;
    const fontFamily = text.fontFamily ?? layoutConfig.text.fontFamily;
    const fontWeight = text.fontWeight ?? 400;

    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.font = `${fontWeight} ${text.fontSize}px ${fontFamily}`;
    ctx.fillStyle = text.fill;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    // 文本位置：考虑 padding（如果有）
    const textX = x;
    const textY = y;

    // 处理 ellipsis 截断
    let displayText = text.content;
    if (box.style?.textOverflow === 'ellipsis' && box.width > 0) {
      displayText = this.truncateTextWithEllipsis(ctx, text.content, box.width);
    }

    ctx.fillText(displayText, textX, textY);
    ctx.restore();
  }

  /**
   * 截断文本并添加省略号
   */
  private truncateTextWithEllipsis(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
    const fullWidth = ctx.measureText(text).width;
    if (fullWidth <= maxWidth) {
      return text;
    }

    const ellipsis = '…';
    const ellipsisWidth = ctx.measureText(ellipsis).width;
    const availableWidth = maxWidth - ellipsisWidth;

    if (availableWidth <= 0) {
      return ellipsis;
    }

    // 二分查找截断位置
    let low = 0;
    let high = text.length;

    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      const width = ctx.measureText(text.slice(0, mid)).width;

      if (width <= availableWidth) {
        low = mid;
      } else {
        high = mid - 1;
      }
    }

    return low === 0 ? ellipsis : text.slice(0, low) + ellipsis;
  }

  /**
   * 规范化圆角配置
   */
  private normalizeCornerRadius(
    radius: CornerRadius | undefined
  ): number | { topLeft: number; topRight: number; bottomLeft: number; bottomRight: number } {
    if (radius === undefined) return 0;
    if (typeof radius === 'number') return radius;
    // [topLeft, topRight, bottomRight, bottomLeft]
    return {
      topLeft: radius[0],
      topRight: radius[1],
      bottomRight: radius[2],
      bottomLeft: radius[3],
    };
  }

  /**
   * 扩展圆角（用于选中边框）
   */
  private expandCornerRadius(
    radius: number | { topLeft: number; topRight: number; bottomLeft: number; bottomRight: number },
    expand: number
  ): number | { topLeft: number; topRight: number; bottomLeft: number; bottomRight: number } {
    if (typeof radius === 'number') {
      return radius + expand;
    }
    return {
      topLeft: radius.topLeft + expand,
      topRight: radius.topRight + expand,
      bottomLeft: radius.bottomLeft + expand,
      bottomRight: radius.bottomRight + expand,
    };
  }
}

export default CanvasRenderer;
