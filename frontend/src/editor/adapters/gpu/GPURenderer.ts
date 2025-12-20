/**
 * GPU 渲染器
 * 
 * 实现 IRenderer 接口，内部管理 WebGL/WebGPU 后端。
 * 自动选择最佳后端：WebGPU 优先，WebGL 2.0 降级。
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

/**
 * GPU 渲染器 - 直接实现 IRenderer，不继承 BaseRenderer
 */
export class GPURenderer implements IRenderer {
  private container: HTMLElement | null = null;
  private backend: IGPUBackend | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private canvasId: string = '';  // Canvas 唯一标识，用于 unmount 时识别
  private preferWebGPU: boolean = true;
  private _viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private width: number = 0;
  private height: number = 0;
  private dpr: number = 1;
  private inputCallback: RawInputCallback | null = null;
  
  // 批次管理器
  private nodeBatch: NodeBatchManager = new NodeBatchManager();
  private edgeBatch: EdgeBatchManager = new EdgeBatchManager();
  private circleBatch: CircleBatchManager = new CircleBatchManager();
  private triangleBatch: TriangleBatchManager = new TriangleBatchManager();
  private textBatch: TextBatchManager = new TextBatchManager();
  
  // 事件处理器引用
  private boundHandlePointerDown: ((e: PointerEvent) => void) | null = null;
  private boundHandlePointerMove: ((e: PointerEvent) => void) | null = null;
  private boundHandlePointerUp: ((e: PointerEvent) => void) | null = null;
  private boundHandleWheel: ((e: WheelEvent) => void) | null = null;
  private boundHandleKeyDown: ((e: KeyboardEvent) => void) | null = null;
  private boundHandleKeyUp: ((e: KeyboardEvent) => void) | null = null;

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
    return backendName === 'webgpu' ? 'WebGPU' : 'WebGL';
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
    
    // 同步创建 canvas
    this.canvas = document.createElement('canvas');
    this.canvas.id = this.canvasId;
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.display = 'block';
    this.canvas.tabIndex = 0;
    this.canvas.style.outline = 'none';
    container.appendChild(this.canvas);
    
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
    
    // 只删除自己创建的 canvas（通过 ID 匹配）
    if (this.canvas && this.canvas.id === this.canvasId && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
    this.canvas = null;
    this.container = null;
    this.ready = false;
    this.readyCallbacks = [];
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.updateSize();
  }

  onInput(callback: RawInputCallback): void {
    this.inputCallback = callback;
  }

  render(data: RenderData): void {
    if (!this.backend) return;
    
    this.backend.beginFrame();
    
    // 设置视口变换矩阵
    const matrix = this.createViewMatrix(data.viewport);
    this.backend.setViewTransform(matrix);
    
    // 更新并渲染边（先渲染边，节点在上层）
    this.edgeBatch.updateFromPaths(data.paths);
    this.backend.renderEdges(this.edgeBatch.getBatch());
    
    // 更新并渲染节点
    this.nodeBatch.updateFromRects(data.rects);
    this.backend.renderNodes(this.nodeBatch.getBatch());
    
    // 更新并渲染文字
    this.textBatch.updateFromTexts(data.texts);
    if (this.textBatch.needsTextureUpdate()) {
      this.backend.updateTextTexture(this.textBatch.getAtlasCanvas());
      this.textBatch.clearTextureUpdateFlag();
    }
    this.backend.renderText(this.textBatch.getBatch());
    
    // 更新并渲染圆形（数据端口）
    // 同时将三角形（执行引脚）临时转换为小圆形渲染
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
    
    this.canvas.addEventListener('pointerdown', this.boundHandlePointerDown);
    this.canvas.addEventListener('pointermove', this.boundHandlePointerMove);
    this.canvas.addEventListener('pointerup', this.boundHandlePointerUp);
    this.canvas.addEventListener('wheel', this.boundHandleWheel, { passive: false });
    this.canvas.addEventListener('keydown', this.boundHandleKeyDown);
    this.canvas.addEventListener('keyup', this.boundHandleKeyUp);
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
}

export default GPURenderer;
