/**
 * MultiLayerCanvasRenderer - 多层 Canvas 渲染器
 * 
 * 整合 LayerManager、ContentRenderer、InteractionRenderer、UIRenderer，
 * 实现完整的多层 Canvas 渲染架构。
 * 
 * 层级结构：
 * - Content Layer: 节点、连线、端口、文字（支持 LOD）
 * - Interaction Layer: 连线预览、选择框、hover 高亮
 * - UI Layer: 类型选择器、属性编辑器
 * 
 * 支持三种图形后端：
 * - Canvas 2D（默认，兼容性最好）
 * - WebGL（性能更好）
 * - WebGPU（最新技术，性能最佳）
 */

import type { IRenderer } from '../../core/IRenderer';
import type { RenderData, Viewport } from '../../core/RenderData';
import type { RawInputCallback } from '../../core/input';
import { createPointerInput, createWheelInput, createKeyInput, extractModifiers, type MouseButton } from './input';
import { LayerManager } from './layers/LayerManager';
import type { IContentRenderer, ContentBackendType } from './layers/IContentRenderer';
import { ContentRenderer } from './layers/ContentRenderer';
import { WebGLContentRenderer } from './layers/WebGLContentRenderer';
import { WebGPUContentRenderer } from './layers/WebGPUContentRenderer';
import { InteractionRenderer } from './layers/InteractionRenderer';
import { UIRenderer } from './layers/UIRenderer';
import { TextCache } from './layers/TextCache';
import { TypeSelector, type TypeOption, type ConstraintData } from './ui/TypeSelector';
import { AttributeEditor, type AttributeDef, type AttributeValue } from './ui/AttributeEditor';
import { detectBestRendererSync } from './layers/RendererDetector';

/** 渲染器配置 */
export interface MultiLayerCanvasConfig {
  /** 是否启用文字缓存 */
  enableTextCache: boolean;
  /** 是否启用拖拽/缩放优化 */
  enableDragOptimization: boolean;
  /** 图形后端类型（默认自动检测） */
  graphicsBackend: ContentBackendType | 'auto';
}

const DEFAULT_CONFIG: MultiLayerCanvasConfig = {
  enableTextCache: true,
  enableDragOptimization: true,
  graphicsBackend: 'auto',
};

/**
 * 多层 Canvas 渲染器
 */
export class MultiLayerCanvasRenderer implements IRenderer {
  private container: HTMLElement | null = null;
  private config: MultiLayerCanvasConfig;
  private inputCallback: RawInputCallback | null = null;
  
  // 层管理
  private layerManager: LayerManager;
  private contentRenderer: IContentRenderer | null = null;
  private currentBackend: ContentBackendType = 'canvas2d';
  private interactionRenderer: InteractionRenderer;
  private uiRenderer: UIRenderer;
  private textCache: TextCache | null = null;
  
  // 当前状态
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  private isDragging: boolean = false;
  private isZooming: boolean = false;
  private dragEndTimer: number | null = null;
  
  // UI 组件
  private typeSelector: TypeSelector | null = null;
  private attributeEditor: AttributeEditor | null = null;
  
  // 回调
  private onTypeSelect: ((nodeId: string, handleId: string, type: string) => void) | null = null;
  private onAttributeChange: ((nodeId: string, name: string, value: unknown) => void) | null = null;

  constructor(config: Partial<MultiLayerCanvasConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.layerManager = new LayerManager();
    this.interactionRenderer = new InteractionRenderer();
    this.uiRenderer = new UIRenderer();
    
    if (this.config.enableTextCache) {
      this.textCache = new TextCache();
    }
  }

  // ============================================================
  // IRenderer 接口实现
  // ============================================================

  mount(container: HTMLElement): void {
    this.container = container;
    
    // 设置容器样式
    container.style.position = 'relative';
    container.style.overflow = 'hidden';
    
    // 挂载层管理器
    this.layerManager.mount(container);
    
    // 初始化内容渲染器
    this.initContentRenderer(container);
    
    // 绑定交互层和 UI 层
    const interactionCanvas = this.layerManager.getCanvas('interaction');
    const uiCanvas = this.layerManager.getCanvas('ui');
    
    if (interactionCanvas) {
      this.interactionRenderer.bind(interactionCanvas);
    }
    if (uiCanvas) {
      this.uiRenderer.bind(uiCanvas);
    }
    
    // 绑定事件
    this.bindEvents();
  }

  /**
   * 初始化内容渲染器（异步）
   */
  private async initContentRenderer(container: HTMLElement): Promise<void> {
    // 确定要使用的后端
    let targetBackend: ContentBackendType;
    if (this.config.graphicsBackend === 'auto') {
      targetBackend = detectBestRendererSync();
    } else {
      targetBackend = this.config.graphicsBackend;
    }

    // 尝试初始化，失败则降级
    const success = await this.tryInitBackend(targetBackend, container);
    if (!success) {
      console.warn(`Failed to init ${targetBackend}, falling back`);
      await this.fallbackInit(targetBackend, container);
    }
  }

  /**
   * 尝试初始化指定后端
   */
  private async tryInitBackend(backend: ContentBackendType, container: HTMLElement): Promise<boolean> {
    // 清理现有渲染器
    if (this.contentRenderer) {
      this.contentRenderer.dispose();
      this.contentRenderer = null;
    }

    let renderer: IContentRenderer;
    switch (backend) {
      case 'webgpu':
        renderer = new WebGPUContentRenderer();
        break;
      case 'webgl':
        renderer = new WebGLContentRenderer();
        break;
      case 'canvas2d':
      default:
        renderer = new ContentRenderer();
        break;
    }

    const success = await renderer.init(container);
    if (success) {
      this.contentRenderer = renderer;
      this.currentBackend = backend;
      console.log(`Content renderer initialized: ${backend}`);
      return true;
    }

    renderer.dispose();
    return false;
  }

  /**
   * 降级初始化
   */
  private async fallbackInit(failedBackend: ContentBackendType, container: HTMLElement): Promise<void> {
    const fallbackOrder: ContentBackendType[] = ['webgpu', 'webgl', 'canvas2d'];
    const startIndex = fallbackOrder.indexOf(failedBackend) + 1;

    for (let i = startIndex; i < fallbackOrder.length; i++) {
      const backend = fallbackOrder[i];
      const success = await this.tryInitBackend(backend, container);
      if (success) {
        return;
      }
    }

    console.error('All renderers failed to initialize');
  }

  unmount(): void {
    this.unbindEvents();
    
    // 清理 UI 组件
    this.hideTypeSelector();
    this.hideAttributeEditor();
    
    // 解绑渲染器
    this.contentRenderer?.dispose();
    this.contentRenderer = null;
    this.interactionRenderer.unbind();
    this.uiRenderer.unbind();
    
    // 卸载层管理器
    this.layerManager.unmount();
    
    // 清理缓存
    this.textCache?.dispose();
    
    this.container = null;
  }

  resize(): void {
    this.layerManager.resize();
    
    const { dpr } = this.layerManager.getSize();
    this.contentRenderer?.resize();
    this.interactionRenderer.setDPR(dpr);
    this.uiRenderer.setDPR(dpr);
  }

  render(data: RenderData): void {
    this.viewport = { ...data.viewport };
    this.layerManager.setViewport(this.viewport);
    this.interactionRenderer.setViewport(this.viewport);
    
    // 根据拖拽/缩放状态选择渲染策略
    if (this.contentRenderer) {
      if (this.config.enableDragOptimization && (this.isDragging || this.isZooming)) {
        // 拖拽/缩放中：只渲染图形，跳过文字
        this.contentRenderer.renderGraphicsOnly(data);
      } else {
        // 正常渲染：完整渲染
        this.contentRenderer.render(data);
      }
    }
    
    // 渲染交互层
    this.interactionRenderer.render(data.hint);
    
    // 渲染 UI 层
    if (this.uiRenderer.hasVisibleComponents()) {
      this.uiRenderer.render();
    }
  }

  onInput(callback: RawInputCallback): void {
    this.inputCallback = callback;
  }

  getName(): string {
    return 'MultiLayerCanvas';
  }

  isAvailable(): boolean {
    return typeof document !== 'undefined' &&
           typeof document.createElement('canvas').getContext === 'function';
  }

  // ============================================================
  // 类型选择器
  // ============================================================

  /**
   * 显示类型选择器
   */
  showTypeSelector(
    nodeId: string,
    handleId: string,
    screenX: number,
    screenY: number,
    options: TypeOption[],
    currentType?: string,
    constraintData?: ConstraintData
  ): void {
    // 隐藏现有的
    this.hideTypeSelector();
    
    // 创建新的类型选择器
    this.typeSelector = new TypeSelector(`type-selector-${nodeId}-${handleId}`);
    
    // 设置约束数据（优先使用新 API）
    if (constraintData) {
      this.typeSelector.setConstraintData(constraintData);
      this.typeSelector.setCurrentType(currentType || '');
    } else {
      // 兼容旧 API
      this.typeSelector.setOptions(options);
    }
    
    this.typeSelector.setPosition(screenX, screenY);
    this.typeSelector.setOnSelect((type) => {
      this.onTypeSelect?.(nodeId, handleId, type);
      this.hideTypeSelector();
    });
    this.typeSelector.setOnClose(() => {
      this.hideTypeSelector();
    });
    
    // 挂载到容器
    if (this.container) {
      this.typeSelector.mount(this.container);
    }
    
    // 添加到 UI 层
    this.uiRenderer.addComponent(this.typeSelector);
    this.typeSelector.show();
    this.uiRenderer.render();
  }

  /**
   * 隐藏类型选择器
   */
  hideTypeSelector(): void {
    if (this.typeSelector) {
      this.uiRenderer.removeComponent(this.typeSelector.id);
      this.typeSelector = null;
      this.uiRenderer.render();
    }
  }

  /**
   * 设置类型选择回调
   */
  setOnTypeSelect(callback: (nodeId: string, handleId: string, type: string) => void): void {
    this.onTypeSelect = callback;
  }

  // ============================================================
  // 属性编辑器
  // ============================================================

  /**
   * 显示属性编辑器
   */
  showAttributeEditor(
    nodeId: string,
    screenX: number,
    screenY: number,
    title: string,
    attributes: AttributeDef[],
    values: AttributeValue[]
  ): void {
    // 隐藏现有的
    this.hideAttributeEditor();
    
    // 创建新的属性编辑器
    this.attributeEditor = new AttributeEditor(`attr-editor-${nodeId}`);
    this.attributeEditor.setTitle(title);
    this.attributeEditor.setAttributes(attributes);
    this.attributeEditor.setValues(values);
    this.attributeEditor.setPosition(screenX, screenY);
    this.attributeEditor.setOnChange((name, value) => {
      this.onAttributeChange?.(nodeId, name, value);
    });
    this.attributeEditor.setOnClose(() => {
      this.hideAttributeEditor();
    });
    
    // 挂载到容器
    if (this.container) {
      this.attributeEditor.mount(this.container);
    }
    
    // 添加到 UI 层
    this.uiRenderer.addComponent(this.attributeEditor);
    this.uiRenderer.render();
  }

  /**
   * 隐藏属性编辑器
   */
  hideAttributeEditor(): void {
    if (this.attributeEditor) {
      this.uiRenderer.removeComponent(this.attributeEditor.id);
      this.attributeEditor = null;
      this.uiRenderer.render();
    }
  }

  /**
   * 设置属性变更回调
   */
  setOnAttributeChange(callback: (nodeId: string, name: string, value: unknown) => void): void {
    this.onAttributeChange = callback;
  }

  // ============================================================
  // 拖拽/缩放优化
  // ============================================================

  /**
   * 通知开始拖拽
   */
  notifyDragStart(): void {
    this.isDragging = true;
    this.cancelDragEndTimer();
  }

  /**
   * 通知结束拖拽
   */
  notifyDragEnd(): void {
    // 延迟结束，避免频繁切换
    this.cancelDragEndTimer();
    this.dragEndTimer = window.setTimeout(() => {
      this.isDragging = false;
      this.dragEndTimer = null;
    }, 100);
  }

  /**
   * 通知开始缩放
   */
  notifyZoomStart(): void {
    this.isZooming = true;
    this.cancelDragEndTimer();
  }

  /**
   * 通知结束缩放
   */
  notifyZoomEnd(): void {
    this.cancelDragEndTimer();
    this.dragEndTimer = window.setTimeout(() => {
      this.isZooming = false;
      this.dragEndTimer = null;
    }, 100);
  }

  private cancelDragEndTimer(): void {
    if (this.dragEndTimer !== null) {
      clearTimeout(this.dragEndTimer);
      this.dragEndTimer = null;
    }
  }

  // ============================================================
  // 事件处理
  // ============================================================

  private boundHandlePointerDown: ((e: PointerEvent) => void) | null = null;
  private boundHandlePointerMove: ((e: PointerEvent) => void) | null = null;
  private boundHandlePointerUp: ((e: PointerEvent) => void) | null = null;
  private boundHandleWheel: ((e: WheelEvent) => void) | null = null;
  private boundHandleKeyDown: ((e: KeyboardEvent) => void) | null = null;
  private boundHandleKeyUp: ((e: KeyboardEvent) => void) | null = null;

  private bindEvents(): void {
    const uiCanvas = this.layerManager.getCanvas('ui');
    if (!uiCanvas) return;
    
    this.boundHandlePointerDown = this.handlePointerDown.bind(this);
    this.boundHandlePointerMove = this.handlePointerMove.bind(this);
    this.boundHandlePointerUp = this.handlePointerUp.bind(this);
    this.boundHandleWheel = this.handleWheel.bind(this);
    this.boundHandleKeyDown = this.handleKeyDown.bind(this);
    this.boundHandleKeyUp = this.handleKeyUp.bind(this);
    
    uiCanvas.addEventListener('pointerdown', this.boundHandlePointerDown);
    uiCanvas.addEventListener('pointermove', this.boundHandlePointerMove);
    uiCanvas.addEventListener('pointerup', this.boundHandlePointerUp);
    uiCanvas.addEventListener('wheel', this.boundHandleWheel, { passive: false });
    uiCanvas.addEventListener('keydown', this.boundHandleKeyDown);
    uiCanvas.addEventListener('keyup', this.boundHandleKeyUp);
    uiCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
    
    uiCanvas.tabIndex = 0;
    uiCanvas.style.outline = 'none';
  }

  private unbindEvents(): void {
    const uiCanvas = this.layerManager.getCanvas('ui');
    if (!uiCanvas) return;
    
    if (this.boundHandlePointerDown) {
      uiCanvas.removeEventListener('pointerdown', this.boundHandlePointerDown);
    }
    if (this.boundHandlePointerMove) {
      uiCanvas.removeEventListener('pointermove', this.boundHandlePointerMove);
    }
    if (this.boundHandlePointerUp) {
      uiCanvas.removeEventListener('pointerup', this.boundHandlePointerUp);
    }
    if (this.boundHandleWheel) {
      uiCanvas.removeEventListener('wheel', this.boundHandleWheel);
    }
    if (this.boundHandleKeyDown) {
      uiCanvas.removeEventListener('keydown', this.boundHandleKeyDown);
    }
    if (this.boundHandleKeyUp) {
      uiCanvas.removeEventListener('keyup', this.boundHandleKeyUp);
    }
  }

  private handlePointerDown(e: PointerEvent): void {
    const uiCanvas = this.layerManager.getCanvas('ui');
    if (!uiCanvas) return;
    uiCanvas.focus();
    
    const rect = uiCanvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    
    // 先检查 UI 层
    const uiEvent = { x: screenX, y: screenY, button: e.button };
    if (this.uiRenderer.handleMouseDown(uiEvent)) {
      this.uiRenderer.render();
      return;
    }
    
    // 传递给控制器
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
    const uiCanvas = this.layerManager.getCanvas('ui');
    if (!uiCanvas) return;
    
    const rect = uiCanvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    
    // 先检查 UI 层
    const uiEvent = { x: screenX, y: screenY, button: e.button };
    if (this.uiRenderer.handleMouseMove(uiEvent)) {
      this.uiRenderer.render();
    }
    
    // 传递给控制器
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
    const uiCanvas = this.layerManager.getCanvas('ui');
    if (!uiCanvas) return;
    
    const rect = uiCanvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    
    // 先检查 UI 层
    const uiEvent = { x: screenX, y: screenY, button: e.button };
    if (this.uiRenderer.handleMouseUp(uiEvent)) {
      this.uiRenderer.render();
      return;
    }
    
    // 传递给控制器
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
    const uiCanvas = this.layerManager.getCanvas('ui');
    if (!uiCanvas) return;
    
    const rect = uiCanvas.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    
    // 先检查 UI 层
    const uiEvent = { x: screenX, y: screenY, deltaX: e.deltaX, deltaY: e.deltaY };
    if (this.uiRenderer.handleWheel(uiEvent)) {
      this.uiRenderer.render();
      return;
    }
    
    // 通知缩放开始
    this.notifyZoomStart();
    
    // 传递给控制器
    const input = createWheelInput(
      e.deltaX,
      e.deltaY,
      screenX,
      screenY,
      extractModifiers(e)
    );
    this.inputCallback?.(input);
    
    // 通知缩放结束
    this.notifyZoomEnd();
  }

  private handleKeyDown(e: KeyboardEvent): void {
    // 先检查 UI 层
    const uiEvent = {
      key: e.key,
      code: e.code,
      ctrlKey: e.ctrlKey,
      shiftKey: e.shiftKey,
      altKey: e.altKey,
    };
    if (this.uiRenderer.handleKeyDown(uiEvent)) {
      this.uiRenderer.render();
      return;
    }
    
    // 传递给控制器
    const input = createKeyInput('down', e.key, e.code, extractModifiers(e));
    this.inputCallback?.(input);
  }

  private handleKeyUp(e: KeyboardEvent): void {
    // 先检查 UI 层
    const uiEvent = {
      key: e.key,
      code: e.code,
      ctrlKey: e.ctrlKey,
      shiftKey: e.shiftKey,
      altKey: e.altKey,
    };
    if (this.uiRenderer.handleKeyUp(uiEvent)) {
      this.uiRenderer.render();
      return;
    }
    
    // 传递给控制器
    const input = createKeyInput('up', e.key, e.code, extractModifiers(e));
    this.inputCallback?.(input);
  }

  // ============================================================
  // 辅助方法
  // ============================================================

  /**
   * 获取当前 LOD 级别
   */
  getLODLevel(): string {
    return this.contentRenderer?.getLODLevel() ?? 'unknown';
  }

  /**
   * 获取视口
   */
  getViewport(): Viewport {
    return { ...this.viewport };
  }

  /**
   * 屏幕坐标转画布坐标
   */
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number } {
    return this.layerManager.screenToCanvas(screenX, screenY);
  }

  /**
   * 画布坐标转屏幕坐标
   */
  canvasToScreen(canvasX: number, canvasY: number): { x: number; y: number } {
    return this.layerManager.canvasToScreen(canvasX, canvasY);
  }

  // ============================================================
  // 渲染器切换
  // ============================================================

  /**
   * 获取当前图形后端
   */
  getCurrentBackend(): ContentBackendType {
    return this.currentBackend;
  }

  /**
   * 切换图形后端
   */
  async switchGraphicsBackend(backend: ContentBackendType): Promise<boolean> {
    if (!this.container) {
      console.warn('Cannot switch backend: not mounted');
      return false;
    }

    if (backend === this.currentBackend) {
      return true;
    }

    const success = await this.tryInitBackend(backend, this.container);
    if (!success) {
      console.warn(`Failed to switch to ${backend}`);
      return false;
    }

    console.log(`Switched to ${backend}`);
    return true;
  }
}
