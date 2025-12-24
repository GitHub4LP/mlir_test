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
        console.log('GPURenderer: backend initialized, size:', this.canvas.width, 'x', this.canvas.height);
      }
      this.ready = true;
      console.log('GPURenderer: ready, calling', this.readyCallbacks.length, 'callbacks');
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
      console.warn('GPURenderer.render: backend is null, skipping render');
      return;
    }
    
    // 缓存渲染数据（用于模式切换后重新渲染）
    this.lastRenderData = data;
    
    this.backend.beginFrame();
    
    // 设置视口变换矩阵
    const matrix = this.createViewMatrix(data.viewport);
    this.backend.setViewTransform(matrix);
    
    // 根据模式渲染边（先渲染边，节点在上层）
    if (this.edgeRenderMode === 'gpu') {
      this.edgeBatch.updateFromPaths(data.paths);
      this.backend.renderEdges(this.edgeBatch.getBatch());
    }
    
    // 更新并渲染节点
    this.nodeBatch.updateFromRects(data.rects);
    this.backend.renderNodes(this.nodeBatch.getBatch());
    
    // 根据模式渲染文字
    if (this.textRenderMode === 'gpu') {
      // GPU 模式：通过纹理图集渲染
      this.textBatch.updateFromTexts(data.texts);
      if (this.textBatch.needsTextureUpdate()) {
        this.backend.updateTextTexture(this.textBatch.getAtlasCanvas());
        this.textBatch.clearTextureUpdateFlag();
      }
      this.backend.renderText(this.textBatch.getBatch());
    }
    
    // 更新并渲染圆形（数据端口）
    this.circleBatch.updateFromCircles(data.circles);
    this.backend.renderCircles(this.circleBatch.getBatch());
    
    // 更新并渲染三角形（执行引脚）
    if (data.triangles && data.triangles.length > 0) {
      this.triangleBatch.updateFromTriangles(data.triangles);
      this.backend.renderTriangles(this.triangleBatch.getBatch());
    }
    
    this.backend.endFrame();
    this._viewport = { ...data.viewport };
    
    // Canvas 模式渲染（边和文字共用 textCanvas）
    const needCanvasRender = this.edgeRenderMode === 'canvas' || this.textRenderMode === 'canvas';
    if (needCanvasRender && this.textCtx && this.textCanvas) {
      this.renderToCanvas(data);
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
   * 使用 Canvas 2D 渲染边和/或文字（混合模式）
   */
  private renderToCanvas(data: RenderData): void {
    if (!this.textCtx || !this.textCanvas) return;
    
    const ctx = this.textCtx;
    const viewport = data.viewport;
    
    ctx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
    ctx.save();
    
    // 应用 DPR 和视口变换
    ctx.scale(this.dpr, this.dpr);
    ctx.translate(viewport.x, viewport.y);
    ctx.scale(viewport.zoom, viewport.zoom);
    
    // Canvas 模式渲染边
    if (this.edgeRenderMode === 'canvas') {
      this.renderEdgesToCanvas(ctx, data);
    }
    
    // Canvas 模式渲染文字
    if (this.textRenderMode === 'canvas') {
      this.renderTextToCanvas(ctx, data);
    }
    
    ctx.restore();
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
    }
  }

  /**
   * 使用 Canvas 2D 渲染文字
   */
  private renderTextToCanvas(ctx: CanvasRenderingContext2D, data: RenderData): void {
    // 启用高质量文字渲染
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // 渲染所有文字（与 CanvasRenderer.renderText 保持一致）
    for (const text of data.texts) {
      ctx.save();
      ctx.font = `${text.fontSize ?? 12}px ${text.fontFamily ?? 'system-ui, sans-serif'}`;
      ctx.fillStyle = text.color ?? '#ffffff';
      ctx.textAlign = (text.align ?? 'left') as CanvasTextAlign;
      ctx.textBaseline = (text.baseline ?? 'top') as CanvasTextBaseline;
      ctx.fillText(text.text, text.x, text.y);
      ctx.restore();
    }
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
        console.log('GPU Renderer: Using WebGPU backend');
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
        console.log('GPU Renderer: Using WebGL 2.0 backend');
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
}


export default GPURenderer;
