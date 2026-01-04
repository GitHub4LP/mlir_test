/**
 * GPU 渲染器
 * 
 * 实现 IRenderer 接口，内部管理 WebGL/WebGPU 后端。
 * 自动选择最佳后端：WebGPU 优先，WebGL 2.0 降级。
 * 
 * 支持两种文字渲染模式：
 * - GPU 模式：文字通过纹理图集在 GPU 渲染
 * - Canvas 模式：文字通过 Canvas 2D 渲染（混合模式）
 */

import type { IRenderer } from '../canvas/IRenderer';
import type { RenderData } from '../canvas/types';
import type { RawInputCallback } from '../canvas/input';
import { createPointerInput, createWheelInput, createKeyInput, extractModifiers } from '../canvas/input';
import type { MouseButton } from '../canvas/input';
import type { IGPUBackend, BackendType } from './backends/IGPUBackend';
import { isWebGPUSupported, isWebGL2Supported } from './backends/IGPUBackend';
import type { Viewport } from '../shared/Viewport';
import { NodeBatchManager } from './geometry/NodeBatch';
import { EdgeBatchManager } from './geometry/EdgeBatch';
import { CircleBatchManager } from './geometry/CircleBatch';
import { TriangleBatchManager } from './geometry/TriangleBatch';
import { TextBatchManager } from './geometry/TextBatch';
import { tokens } from '../shared/styles';
import type { LayoutBox, CornerRadius } from '../../core/layout/types';
import { layoutConfig } from '../../core/layout/LayoutConfig';

/** 渲染模式（GPU 或 Canvas 2D） */
export type RenderMode = 'gpu' | 'canvas';

/** 文字渲染模式 */
export type TextRenderMode = RenderMode;

/** 边渲染模式 */
export type EdgeRenderMode = RenderMode;

/**
 * GPU 渲染器 - 直接实现 IRenderer，不继承 BaseRenderer
 */
export class GPURenderer implements IRenderer {
  private container: HTMLElement | null = null;
  private backend: IGPUBackend | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private uiCanvas: HTMLCanvasElement | null = null;  // UI 层 Canvas
  private uiCtx: CanvasRenderingContext2D | null = null;
  private textCanvas: HTMLCanvasElement | null = null;  // 文字层 Canvas（混合模式）
  private textCtx: CanvasRenderingContext2D | null = null;
  private canvasId: string = '';  // Canvas 唯一标识，用于 unmount 时识别
  private preferWebGPU: boolean = true;
  private _viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private width: number = 0;
  private height: number = 0;
  private dpr: number = 1;
  private inputCallback: RawInputCallback | null = null;
  
  // 文字渲染模式
  private textRenderMode: TextRenderMode = 'gpu';
  
  // 边渲染模式
  private edgeRenderMode: EdgeRenderMode = 'gpu';
  
  // 批次管理器
  private nodeBatch: NodeBatchManager = new NodeBatchManager();
  private edgeBatch: EdgeBatchManager = new EdgeBatchManager();
  private circleBatch: CircleBatchManager = new CircleBatchManager();
  private triangleBatch: TriangleBatchManager = new TriangleBatchManager();
  private textBatch: TextBatchManager = new TextBatchManager();
  
  // 缓存的渲染数据（用于模式切换后重新渲染）
  private lastRenderData: RenderData | null = null;
  
  // 事件处理器引用
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
  
  // UI 渲染回调
  private uiRenderCallback: ((ctx: CanvasRenderingContext2D) => void) | null = null;

  // 就绪状态
  private ready: boolean = false;
  private readyCallbacks: Array<() => void> = [];

  constructor(preferWebGPU: boolean = true) {
    this.preferWebGPU = preferWebGPU;
  }

  getBackendType(): BackendType | null {
    return this.backend?.name ?? null;
  }

  getName(): string {
    const backendName = this.backend?.name ?? 'GPU';
    const textMode = this.textRenderMode === 'canvas' ? '+Canvas文字' : '';
    return (backendName === 'webgpu' ? 'WebGPU' : 'WebGL') + textMode;
  }

  /**
   * 获取当前文字渲染模式
   */
  getTextRenderMode(): TextRenderMode {
    return this.textRenderMode;
  }

  /**
   * 设置文字渲染模式
   */
  setTextRenderMode(mode: TextRenderMode): void {
    if (this.textRenderMode === mode) return;
    this.textRenderMode = mode;
    
    // 更新文字层 Canvas 可见性
    if (this.textCanvas) {
      this.textCanvas.style.display = (mode === 'canvas' || this.edgeRenderMode === 'canvas') ? 'block' : 'none';
    }
    
    // 重新渲染
    if (this.lastRenderData) {
      this.render(this.lastRenderData);
    }
  }

  /**
   * 获取当前边渲染模式
   */
  getEdgeRenderMode(): EdgeRenderMode {
    return this.edgeRenderMode;
  }

  /**
   * 设置边渲染模式
   */
  setEdgeRenderMode(mode: EdgeRenderMode): void {
    if (this.edgeRenderMode === mode) return;
    this.edgeRenderMode = mode;
    
    // 更新 Canvas 可见性（边和文字共用 textCanvas）
    if (this.textCanvas) {
      this.textCanvas.style.display = (mode === 'canvas' || this.textRenderMode === 'canvas') ? 'block' : 'none';
    }
    
    // 重新渲染
    if (this.lastRenderData) {
      this.render(this.lastRenderData);
    }
  }

  isAvailable(): boolean {
    return isWebGPUSupported() || isWebGL2Supported();
  }

  getViewport(): Viewport {
    return this._viewport;
  }

  waitForReady(): Promise<void> {
    if (this.ready) return Promise.resolve();
    return new Promise(resolve => {
      this.readyCallbacks.push(resolve);
    });
  }

  isReady(): boolean {
    return this.ready;
  }

  // ============================================================
  // IRenderer 接口实现
  // ============================================================

  mount(container: HTMLElement): void {
    this.container = container;
    
    // 生成唯一 ID
    this.canvasId = `gpu-canvas-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    
    // 同步创建 GPU canvas
    this.canvas = document.createElement('canvas');
    this.canvas.id = this.canvasId;
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.display = 'block';
    this.canvas.style.position = 'absolute';
    this.canvas.style.top = '0';
    this.canvas.style.left = '0';
    this.canvas.style.zIndex = '1';  // 底层
    container.appendChild(this.canvas);
    
    // 创建文字层 Canvas（混合模式时使用）
    this.textCanvas = document.createElement('canvas');
    this.textCanvas.id = `${this.canvasId}-text`;
    this.textCanvas.style.width = '100%';
    this.textCanvas.style.height = '100%';
    this.textCanvas.style.display = this.textRenderMode === 'canvas' ? 'block' : 'none';
    this.textCanvas.style.position = 'absolute';
    this.textCanvas.style.top = '0';
    this.textCanvas.style.left = '0';
    this.textCanvas.style.pointerEvents = 'none';
    this.textCanvas.style.zIndex = '2';  // 在 GPU canvas 之上
    container.appendChild(this.textCanvas);
    this.textCtx = this.textCanvas.getContext('2d');
    
    // 创建 UI Canvas 层（叠加在 GPU canvas 上）
    this.uiCanvas = document.createElement('canvas');
    this.uiCanvas.id = `${this.canvasId}-ui`;
    this.uiCanvas.style.width = '100%';
    this.uiCanvas.style.height = '100%';
    this.uiCanvas.style.display = 'block';
    this.uiCanvas.style.position = 'absolute';
    this.uiCanvas.style.top = '0';
    this.uiCanvas.style.left = '0';
    this.uiCanvas.style.pointerEvents = 'none';  // 事件穿透到 GPU canvas
    this.uiCanvas.style.zIndex = '3';  // 在文字层之上
    this.uiCanvas.tabIndex = -1;
    container.appendChild(this.uiCanvas);
    this.uiCtx = this.uiCanvas.getContext('2d');
    
    // GPU canvas 接收事件
    this.canvas.tabIndex = 0;
    this.canvas.style.outline = 'none';
    
    // 同步更新尺寸
    this.updateSize();
    
    // 同步绑定事件
    this.bindEvents();
    
    // 异步初始化后端
    this.initBackend().then(() => {
      // 确保 backend 尺寸与 canvas 同步
      if (this.backend && this.canvas) {
        this.backend.resize(this.canvas.width, this.canvas.height);
      }
      this.ready = true;
      for (const cb of this.readyCallbacks) {
        cb();
      }
      this.readyCallbacks = [];
    }).catch(err => {
      console.error('GPU backend init failed:', err);
    });
  }

  unmount(): void {
    this.unbindEvents();
    
    if (this.backend) {
      this.backend.dispose();
      this.backend = null;
    }
    
    // 删除 UI canvas
    if (this.uiCanvas && this.uiCanvas.parentNode) {
      this.uiCanvas.parentNode.removeChild(this.uiCanvas);
    }
    this.uiCanvas = null;
    this.uiCtx = null;
    
    // 删除文字层 canvas
    if (this.textCanvas && this.textCanvas.parentNode) {
      this.textCanvas.parentNode.removeChild(this.textCanvas);
    }
    this.textCanvas = null;
    this.textCtx = null;
    
    // 只删除自己创建的 canvas（通过 ID 匹配）
    if (this.canvas && this.canvas.id === this.canvasId && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
    this.canvas = null;
    this.container = null;
    this.ready = false;
    this.readyCallbacks = [];
    this.lastRenderData = null;
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.updateSize();
  }

  onInput(callback: RawInputCallback): void {
    this.inputCallback = callback;
  }

  /**
   * 设置拖放回调
   */
  setDropCallback(callback: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null): void {
    this.dropCallback = callback;
  }

  /**
   * 设置 UI 渲染回调（用于在 GPU 渲染后绘制 UI 层）
   */
  setUIRenderCallback(callback: ((ctx: CanvasRenderingContext2D) => void) | null): void {
    this.uiRenderCallback = callback;
  }

  render(data: RenderData): void {
    if (!this.backend) {
      // backend 未就绪时静默跳过（初始化是异步的）
      return;
    }
    
    // 缓存渲染数据（用于模式切换后重新渲染）
    this.lastRenderData = data;
    
    // 判断是否需要 Canvas 层
    const needsCanvasLayer = this.textRenderMode === 'canvas' || this.edgeRenderMode === 'canvas';
    
    // 管理 textCanvas 可见性
    if (this.textCanvas) {
      this.textCanvas.style.display = needsCanvasLayer ? 'block' : 'none';
    }
    
    // 清空 Canvas 层（如果需要）
    if (needsCanvasLayer && this.textCtx && this.textCanvas) {
      this.textCtx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
    }
    
    this.backend.beginFrame();
    
    // 设置视口变换矩阵
    const matrix = this.createViewMatrix(data.viewport);
    this.backend.setViewTransform(matrix);
    
    // 渲染边（总是先渲染，节点在上层）
    if (this.edgeRenderMode === 'gpu') {
      this.edgeBatch.updateFromPaths(data.paths);
      this.backend.renderEdges(this.edgeBatch.getBatch());
    }
    
    // 使用 LayoutBox 系统渲染节点
    if (data.layoutBoxes && data.layoutBoxes.size > 0) {
      this.renderNodesFromLayoutBoxes(data);
    }
    
    this.backend.endFrame();
    this._viewport = { ...data.viewport };
    
    // Canvas 模式渲染边（在 GPU 渲染之后）
    if (this.edgeRenderMode === 'canvas' && this.textCtx && this.textCanvas) {
      const ctx = this.textCtx;
      ctx.save();
      ctx.scale(this.dpr, this.dpr);
      ctx.translate(data.viewport.x, data.viewport.y);
      ctx.scale(data.viewport.zoom, data.viewport.zoom);
      this.renderEdgesToCanvas(ctx, data);
      ctx.restore();
    }
    
    // 渲染交互提示（连接预览线、选择框等）- 使用 textCanvas
    if (data.hint && this.textCtx && this.textCanvas) {
      const ctx = this.textCtx;
      ctx.save();
      ctx.scale(this.dpr, this.dpr);
      ctx.translate(data.viewport.x, data.viewport.y);
      ctx.scale(data.viewport.zoom, data.viewport.zoom);
      this.renderHint(ctx, data.hint);
      ctx.restore();
    }
    
    // 渲染 UI 层
    if (this.uiCtx && this.uiRenderCallback) {
      this.uiCtx.clearRect(0, 0, this.uiCanvas!.width, this.uiCanvas!.height);
      this.uiCtx.save();
      this.uiCtx.scale(this.dpr, this.dpr);
      this.uiRenderCallback(this.uiCtx);
      this.uiCtx.restore();
    }
  }

  /**
   * 从 LayoutBox 提取图元渲染节点（GPU 渲染）
   */
  private renderNodesFromLayoutBoxes(data: RenderData): void {
    if (!data.layoutBoxes || !this.backend) return;
    
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
    
    // 收集图元
    const rects: import('../../core/RenderData').RenderRect[] = [];
    const texts: import('../../core/RenderData').RenderText[] = [];
    const circles: import('../../core/RenderData').RenderCircle[] = [];
    const triangles: import('../../core/RenderData').RenderTriangle[] = [];
    
    // 按 zIndex 排序
    const sortedEntries = [...data.layoutBoxes.entries()].sort((a, b) => {
      const aInfo = nodeInfoMap.get(a[0]);
      const bInfo = nodeInfoMap.get(b[0]);
      return (aInfo?.zIndex ?? 0) - (bInfo?.zIndex ?? 0);
    });
    
    // 从 LayoutBox 提取图元
    for (const [nodeId, layoutBox] of sortedEntries) {
      const nodeInfo = nodeInfoMap.get(nodeId);
      const selected = nodeInfo?.selected ?? false;
      const zIndex = nodeInfo?.zIndex ?? 0;
      
      // LayoutBox 的 x, y 已经是画布坐标（computeNodeLayoutBox 设置的）
      // 所以 offsetX, offsetY 应该是 0
      this.extractPrimitivesFromLayoutBox(
        layoutBox, 0, 0, selected, nodeId, zIndex,
        rects, texts, circles, triangles
      );
    }
    
    // 使用 GPU batch 渲染
    this.nodeBatch.updateFromRects(rects);
    this.backend.renderNodes(this.nodeBatch.getBatch());
    
    // 渲染文字
    if (this.textRenderMode === 'gpu') {
      this.textBatch.updateFromTexts(texts);
      if (this.textBatch.needsTextureUpdate()) {
        this.backend.updateTextTexture(this.textBatch.getAtlasCanvas());
        this.textBatch.clearTextureUpdateFlag();
      }
      this.backend.renderText(this.textBatch.getBatch());
    } else if (this.textCtx) {
      // Canvas 模式渲染文字（textCanvas 已在 render() 中清空）
      const ctx = this.textCtx;
      ctx.save();
      ctx.scale(this.dpr, this.dpr);
      ctx.translate(data.viewport.x, data.viewport.y);
      ctx.scale(data.viewport.zoom, data.viewport.zoom);
      this.renderTextToCanvas(ctx, { ...data, texts });
      ctx.restore();
    }
    
    // 渲染圆形（数据端口）
    this.circleBatch.updateFromCircles(circles);
    this.backend.renderCircles(this.circleBatch.getBatch());
    
    // 渲染三角形（执行引脚）
    if (triangles.length > 0) {
      this.triangleBatch.updateFromTriangles(triangles);
      this.backend.renderTriangles(this.triangleBatch.getBatch());
    }
  }
  
  /**
   * 从 LayoutBox 树提取图元
   */
  private extractPrimitivesFromLayoutBox(
    box: LayoutBox,
    offsetX: number,
    offsetY: number,
    selected: boolean,
    nodeId: string,
    zIndex: number,
    rects: import('../../core/RenderData').RenderRect[],
    texts: import('../../core/RenderData').RenderText[],
    circles: import('../../core/RenderData').RenderCircle[],
    triangles: import('../../core/RenderData').RenderTriangle[]
  ): void {
    const absX = offsetX + box.x;
    const absY = offsetY + box.y;
    
    // 提取矩形（背景）- 通用逻辑：只看 style.fill
    if (box.style?.fill && box.style.fill !== 'transparent') {
      const style = box.style;
      const radius = this.normalizeCornerRadius(style.cornerRadius);
      
      // typeLabel 特殊处理：使用 pinColor 的半透明版本
      let fillColor: string = style.fill!;
      if (box.type === 'typeLabel' && box.interactive?.pinColor) {
        fillColor = this.colorWithAlpha(box.interactive.pinColor, 0.3);
      }
      
      // 节点背景（不传递 selected，选中边框单独生成）
      rects.push({
        id: `lb-${box.type}-${nodeId}-${absX}-${absY}`,
        x: absX,
        y: absY,
        width: box.width,
        height: box.height,
        fillColor: fillColor,
        borderColor: style.stroke ?? 'transparent',
        borderWidth: style.strokeWidth ?? 0,
        borderRadius: radius,
        selected: false,
        zIndex: zIndex,
      });
    }
    
    // node 类型：生成选中边框（4 个填充矩形）
    // 注意：选中边框独立于 fill 条件，因为 node 容器可能没有 fill
    if (box.type === 'node' && selected) {
      const borderWidth = layoutConfig.node.selected?.strokeWidth ?? 2;
      const borderColor = layoutConfig.node.selected?.stroke ?? '#60a5fa';
      const offset = 2;
      const x = absX - offset;
      const y = absY - offset;
      const w = box.width + offset * 2;
      const h = box.height + offset * 2;
      const selectionZIndex = zIndex + 100;
      
      // 上边框
      rects.push({
        id: `selection-top-${nodeId}`,
        x: x, y: y, width: w, height: borderWidth,
        fillColor: borderColor, borderColor: 'transparent', borderWidth: 0,
        borderRadius: 0, selected: false, zIndex: selectionZIndex,
      });
      // 下边框
      rects.push({
        id: `selection-bottom-${nodeId}`,
        x: x, y: y + h - borderWidth, width: w, height: borderWidth,
        fillColor: borderColor, borderColor: 'transparent', borderWidth: 0,
        borderRadius: 0, selected: false, zIndex: selectionZIndex,
      });
      // 左边框
      rects.push({
        id: `selection-left-${nodeId}`,
        x: x, y: y, width: borderWidth, height: h,
        fillColor: borderColor, borderColor: 'transparent', borderWidth: 0,
        borderRadius: 0, selected: false, zIndex: selectionZIndex,
      });
      // 右边框
      rects.push({
        id: `selection-right-${nodeId}`,
        x: x + w - borderWidth, y: y, width: borderWidth, height: h,
        fillColor: borderColor, borderColor: 'transparent', borderWidth: 0,
        borderRadius: 0, selected: false, zIndex: selectionZIndex,
      });
    }
    
    // 提取 Handle
    if (box.type === 'handle' && box.interactive?.id) {
      const id = box.interactive.id;
      const handleConfig = layoutConfig.handle;
      const size = typeof handleConfig.width === 'number' ? handleConfig.width : 12;
      const centerX = absX + box.width / 2;
      const centerY = absY + box.height / 2;
      
      const isExec = id.includes('exec');
      
      if (isExec) {
        // 执行端口：三角形
        triangles.push({
          id: `lb-${id}-${nodeId}`,
          x: centerX,
          y: centerY,
          size: size * 0.6,  // 与 WebGL/WebGPU 渲染器一致
          fillColor: '#ffffff',
          borderColor: '#ffffff',
          borderWidth: 1,
          direction: 'right',
        });
      } else {
        // 数据端口：圆形，从 LayoutBox.interactive.pinColor 获取颜色
        const color = box.interactive.pinColor ?? layoutConfig.nodeType.operation;
        
        circles.push({
          id: `lb-${id}-${nodeId}`,
          x: centerX,
          y: centerY,
          radius: size / 2,
          fillColor: color,
          borderColor: tokens.node.bg,
          borderWidth: 2,
        });
      }
    }
    
    // 提取文本 - 直接使用布局引擎计算的位置
    if (box.text) {
      const text = box.text;
      const hasEllipsis = box.style?.textOverflow === 'ellipsis';
      texts.push({
        id: `lb-text-${nodeId}-${absX}-${absY}`,
        text: text.content,
        x: absX,
        y: absY,
        fontSize: text.fontSize,
        fontFamily: text.fontFamily ?? layoutConfig.text.fontFamily,
        color: text.fill,
        align: 'left',
        baseline: 'top',
        ellipsis: hasEllipsis,
        maxWidth: hasEllipsis ? box.width : undefined,
      });
    }
    
    // 递归处理子节点（子节点不需要选中框）
    for (const child of box.children) {
      this.extractPrimitivesFromLayoutBox(
        child, absX, absY, false, nodeId, zIndex,
        rects, texts, circles, triangles
      );
    }
  }

  /**
   * 使用 Canvas 2D 渲染边
   */
  private renderEdgesToCanvas(ctx: CanvasRenderingContext2D, data: RenderData): void {
    for (const path of data.paths) {
      if (path.points.length < 2) continue;
      
      ctx.beginPath();
      ctx.strokeStyle = path.color ?? '#888888';
      ctx.lineWidth = path.width ?? 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      
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
    }
  }

  /**
   * 使用 Canvas 2D 渲染文字
   */
  private renderTextToCanvas(ctx: CanvasRenderingContext2D, data: RenderData): void {
    // 启用高质量文字渲染
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // 渲染所有文字
    for (const text of data.texts) {
      ctx.save();
      ctx.font = `${text.fontSize ?? 12}px ${text.fontFamily ?? tokens.text.fontFamily}`;
      ctx.fillStyle = text.color ?? '#ffffff';
      ctx.textAlign = (text.align ?? 'left') as CanvasTextAlign;
      ctx.textBaseline = (text.baseline ?? 'top') as CanvasTextBaseline;
      
      // 处理 ellipsis 截断
      let displayText = text.text;
      if (text.ellipsis && text.maxWidth !== undefined && text.maxWidth > 0) {
        displayText = this.truncateTextForCanvas(ctx, text.text, text.maxWidth);
      }
      
      ctx.fillText(displayText, text.x, text.y);
      ctx.restore();
    }
  }
  
  /**
   * Canvas 模式下截断文本并添加省略号
   */
  private truncateTextForCanvas(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
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

  // ============================================================
  // 私有方法
  // ============================================================

  private async initBackend(): Promise<void> {
    if (!this.canvas) return;
    const canvas = this.canvas;
    
    // 尝试 WebGPU
    if (this.preferWebGPU && isWebGPUSupported()) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const module = await import('./backends/WebGPUBackend') as any;
        const backend = new module.WebGPUBackend() as IGPUBackend;
        await backend.init(canvas);
        this.backend = backend;
        return;
      } catch (e) {
        console.warn('WebGPU initialization failed, falling back to WebGL:', e);
      }
    }
    
    // 尝试 WebGL 2.0
    if (isWebGL2Supported()) {
      try {
        const { WebGLBackend } = await import('./backends/WebGLBackend');
        this.backend = new WebGLBackend();
        await this.backend.init(canvas);
        return;
      } catch (e) {
        console.error('WebGL initialization failed:', e);
      }
    }
    
    throw new Error('No GPU backend available.');
  }

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
    
    // 同步更新文字层 Canvas 尺寸
    if (this.textCanvas) {
      this.textCanvas.width = this.width * this.dpr;
      this.textCanvas.height = this.height * this.dpr;
      this.textCanvas.style.width = `${this.width}px`;
      this.textCanvas.style.height = `${this.height}px`;
    }
    
    // 同步更新 UI Canvas 尺寸
    if (this.uiCanvas) {
      this.uiCanvas.width = this.width * this.dpr;
      this.uiCanvas.height = this.height * this.dpr;
      this.uiCanvas.style.width = `${this.width}px`;
      this.uiCanvas.style.height = `${this.height}px`;
    }
    
    this.backend?.resize(this.canvas.width, this.canvas.height);
  }

  private createViewMatrix(viewport: Viewport): Float32Array {
    const matrix = new Float32Array(9);
    matrix[0] = viewport.zoom * this.dpr;
    matrix[1] = 0;
    matrix[2] = 0;
    matrix[3] = 0;
    matrix[4] = viewport.zoom * this.dpr;
    matrix[5] = 0;
    matrix[6] = viewport.x * this.dpr;
    matrix[7] = viewport.y * this.dpr;
    matrix[8] = 1;
    return matrix;
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

  private emitInput(input: Parameters<RawInputCallback>[0]): void {
    this.inputCallback?.(input);
  }

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
    this.emitInput(input);
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
    this.emitInput(input);
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
    this.emitInput(input);
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
    this.emitInput(input);
  }

  private handleKeyDown(e: KeyboardEvent): void {
    const input = createKeyInput('down', e.key, e.code, extractModifiers(e));
    this.emitInput(input);
  }

  private handleKeyUp(e: KeyboardEvent): void {
    const input = createKeyInput('up', e.key, e.code, extractModifiers(e));
    this.emitInput(input);
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
    const canvasX = (screenX - this._viewport.x) / this._viewport.zoom;
    const canvasY = (screenY - this._viewport.y) / this._viewport.zoom;
    
    this.dropCallback?.(canvasX, canvasY, e.dataTransfer);
  }

  // ============================================================================
  // LayoutBox 辅助方法
  // ============================================================================

  /**
   * 规范化圆角配置
   */
  private normalizeCornerRadius(
    radius: CornerRadius | undefined
  ): number | { topLeft: number; topRight: number; bottomLeft: number; bottomRight: number } {
    if (radius === undefined) return 0;
    if (typeof radius === 'number') return radius;
    return {
      topLeft: radius[0],
      topRight: radius[1],
      bottomRight: radius[2],
      bottomLeft: radius[3],
    };
  }

  /**
   * 将颜色转换为带透明度的版本
   */
  private colorWithAlpha(color: string, alpha: number): string {
    if (color.startsWith('#')) {
      const hex = color.slice(1);
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    if (color.startsWith('rgb')) {
      const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
      if (match) {
        return `rgba(${match[1]}, ${match[2]}, ${match[3]}, ${alpha})`;
      }
    }
    return color;
  }

  // ============================================================================
  // 交互提示渲染
  // ============================================================================

  /**
   * 渲染交互提示（连接预览线、选择框等）
   */
  private renderHint(ctx: CanvasRenderingContext2D, hint: import('./types').InteractionHint): void {
    // 渲染连接预览线
    if (hint.connectionPreview) {
      this.renderPath(ctx, hint.connectionPreview);
    }
    
    // 渲染选择框
    if (hint.selectionBox) {
      this.renderSelectionBox(ctx, hint.selectionBox);
    }
    
    // 渲染拖拽预览
    if (hint.dragPreview) {
      ctx.save();
      ctx.globalAlpha = 0.5;
      this.renderSelectionBox(ctx, hint.dragPreview);
      ctx.restore();
    }
  }

  /**
   * 渲染路径（用于连接预览线）
   */
  private renderPath(ctx: CanvasRenderingContext2D, path: import('./types').RenderPath): void {
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
    ctx.restore();
  }

  /**
   * 渲染选择框
   */
  private renderSelectionBox(ctx: CanvasRenderingContext2D, rect: import('./types').RenderRect): void {
    ctx.save();
    
    if (rect.fillColor && rect.fillColor !== 'transparent') {
      ctx.fillStyle = rect.fillColor;
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    }
    
    if (rect.borderWidth && rect.borderWidth > 0 && rect.borderColor && rect.borderColor !== 'transparent') {
      ctx.strokeStyle = rect.borderColor;
      ctx.lineWidth = rect.borderWidth;
      ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    }
    
    ctx.restore();
  }

}


export default GPURenderer;
