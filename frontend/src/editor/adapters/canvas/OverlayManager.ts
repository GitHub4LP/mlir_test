/**
 * OverlayManager - Canvas/GPU 渲染器的 HTML 覆盖层管理器
 * 
 * 设计原则：
 * - Canvas/GPU 渲染节点主体，HTML 覆盖层渲染交互 UI（类型选择器、属性编辑器）
 * - 覆盖层位置跟随画布变换（平移、缩放）
 * - 支持 React 组件渲染到覆盖层
 * 
 * 使用场景：
 * - 点击节点端口的类型标签 → 显示类型选择器
 * - 点击节点属性区域 → 显示属性编辑器
 * - 点击参数/返回值名 → 显示可编辑名称
 * - 点击 Traits 按钮 → 显示 Traits 编辑器
 * 
 * 注意：对于 React 应用，推荐使用 overlays/OverlayContainer 组件
 */

import type { Viewport } from '../../core/RenderData';

/** 覆盖层类型 */
export type OverlayType = 
  | 'type-selector' 
  | 'attribute-editor'
  | 'editable-name'
  | 'traits-editor';

/** 覆盖层配置 */
export interface OverlayConfig {
  /** 覆盖层类型 */
  type: OverlayType;
  /** 关联的节点 ID */
  nodeId: string;
  /** 关联的端口 ID（类型选择器用） */
  portId?: string;
  /** 关联的属性名（属性编辑器用） */
  attributeName?: string;
  /** 画布坐标 X */
  canvasX: number;
  /** 画布坐标 Y */
  canvasY: number;
  /** 宽度（可选，默认自适应） */
  width?: number;
  /** 高度（可选，默认自适应） */
  height?: number;
  /** 额外数据（传递给渲染组件） */
  data?: Record<string, unknown>;
}

/** 活动覆盖层状态 */
export interface ActiveOverlay extends OverlayConfig {
  /** 唯一 ID */
  id: string;
  /** 屏幕坐标 X（根据视口计算） */
  screenX: number;
  /** 屏幕坐标 Y（根据视口计算） */
  screenY: number;
}

/** 覆盖层渲染回调 */
export type OverlayRenderCallback = (overlay: ActiveOverlay) => void;

/**
 * OverlayManager 类
 * 
 * 管理 Canvas/GPU 渲染器上的 HTML 覆盖层
 */
export class OverlayManager {
  /** 容器元素 */
  private container: HTMLElement | null = null;
  
  /** 覆盖层容器 */
  private overlayContainer: HTMLDivElement | null = null;
  
  /** 当前视口 */
  private viewport: Viewport = { x: 0, y: 0, zoom: 1 };
  
  /** 活动覆盖层列表 */
  private activeOverlays: Map<string, ActiveOverlay> = new Map();
  
  /** 覆盖层变化回调 */
  private onOverlayChange: OverlayRenderCallback | null = null;
  
  /** ID 计数器 */
  private idCounter: number = 0;

  /**
   * 挂载到容器
   */
  mount(container: HTMLElement): void {
    this.container = container;
    
    // 创建覆盖层容器
    this.overlayContainer = document.createElement('div');
    this.overlayContainer.className = 'canvas-overlay-container';
    this.overlayContainer.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      overflow: hidden;
      z-index: 100;
    `;
    
    container.appendChild(this.overlayContainer);
  }

  /**
   * 卸载
   */
  unmount(): void {
    if (this.overlayContainer && this.container) {
      this.container.removeChild(this.overlayContainer);
    }
    this.overlayContainer = null;
    this.container = null;
    this.activeOverlays.clear();
  }

  /**
   * 更新视口
   */
  updateViewport(viewport: Viewport): void {
    this.viewport = { ...viewport };
    this.updateOverlayPositions();
  }

  /**
   * 显示覆盖层
   */
  show(config: OverlayConfig): string {
    const id = `overlay-${++this.idCounter}`;
    const screenPos = this.canvasToScreen(config.canvasX, config.canvasY);
    
    const overlay: ActiveOverlay = {
      ...config,
      id,
      screenX: screenPos.x,
      screenY: screenPos.y,
    };
    
    this.activeOverlays.set(id, overlay);
    this.onOverlayChange?.(overlay);
    
    return id;
  }

  /**
   * 隐藏覆盖层
   */
  hide(id: string): void {
    this.activeOverlays.delete(id);
  }

  /**
   * 隐藏所有覆盖层
   */
  hideAll(): void {
    this.activeOverlays.clear();
  }

  /**
   * 隐藏指定节点的所有覆盖层
   */
  hideByNode(nodeId: string): void {
    for (const [id, overlay] of this.activeOverlays) {
      if (overlay.nodeId === nodeId) {
        this.activeOverlays.delete(id);
      }
    }
  }

  /**
   * 获取所有活动覆盖层
   */
  getActiveOverlays(): ActiveOverlay[] {
    return Array.from(this.activeOverlays.values());
  }

  /**
   * 获取覆盖层容器
   */
  getContainer(): HTMLDivElement | null {
    return this.overlayContainer;
  }

  /**
   * 设置覆盖层变化回调
   */
  setOnOverlayChange(callback: OverlayRenderCallback | null): void {
    this.onOverlayChange = callback;
  }

  /**
   * 画布坐标转屏幕坐标
   */
  private canvasToScreen(canvasX: number, canvasY: number): { x: number; y: number } {
    return {
      x: canvasX * this.viewport.zoom + this.viewport.x,
      y: canvasY * this.viewport.zoom + this.viewport.y,
    };
  }

  /**
   * 更新所有覆盖层位置
   */
  private updateOverlayPositions(): void {
    for (const [id, overlay] of this.activeOverlays) {
      const screenPos = this.canvasToScreen(overlay.canvasX, overlay.canvasY);
      overlay.screenX = screenPos.x;
      overlay.screenY = screenPos.y;
      this.activeOverlays.set(id, overlay);
    }
  }
}

/** 单例实例 */
let overlayManagerInstance: OverlayManager | null = null;

/**
 * 获取 OverlayManager 单例
 */
export function getOverlayManager(): OverlayManager {
  if (!overlayManagerInstance) {
    overlayManagerInstance = new OverlayManager();
  }
  return overlayManagerInstance;
}

/**
 * 重置 OverlayManager（用于测试）
 */
export function resetOverlayManager(): void {
  if (overlayManagerInstance) {
    overlayManagerInstance.unmount();
    overlayManagerInstance = null;
  }
}
