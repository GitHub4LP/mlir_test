/**
 * UIManager - Canvas UI 管理器
 * 
 * 统一管理所有 Canvas UI 组件（TypeSelector、AttributeEditor 等）。
 * 负责：
 * - UI 组件的生命周期管理
 * - 事件路由（UI 组件优先）
 * - 渲染协调
 */

import type { UIMouseEvent, UIKeyEvent, UIWheelEvent } from './ui/UIComponent';
import { TypeSelector, type TypeOption } from './ui/TypeSelector';

export interface TypeSelectorState {
  visible: boolean;
  nodeId: string;
  handleId: string;
  screenX: number;
  screenY: number;
  options: TypeOption[];
  currentType?: string;
}

export interface UIManagerCallbacks {
  onTypeSelect?: (nodeId: string, handleId: string, type: string) => void;
  onTypeSelectorClose?: () => void;
}

/**
 * Canvas UI 管理器
 */
export class UIManager {
  private typeSelector: TypeSelector;
  private typeSelectorState: TypeSelectorState = {
    visible: false,
    nodeId: '',
    handleId: '',
    screenX: 0,
    screenY: 0,
    options: [],
  };
  
  private callbacks: UIManagerCallbacks = {};
  
  constructor() {
    this.typeSelector = new TypeSelector('type-selector');
    this.typeSelector.visible = false;
    
    // 设置回调
    this.typeSelector.setOnSelect((type) => {
      const { nodeId, handleId } = this.typeSelectorState;
      this.callbacks.onTypeSelect?.(nodeId, handleId, type);
      this.hideTypeSelector();
    });
    
    this.typeSelector.setOnClose(() => {
      this.hideTypeSelector();
    });
  }
  
  /**
   * 挂载到容器（用于隐藏 input 等 DOM 元素）
   */
  mount(container: HTMLElement): void {
    this.typeSelector.mount(container);
  }
  
  /**
   * 卸载
   */
  unmount(): void {
    this.typeSelector.unmount();
  }
  
  /**
   * 设置回调
   */
  setCallbacks(callbacks: UIManagerCallbacks): void {
    this.callbacks = callbacks;
  }
  
  /**
   * 显示类型选择器
   */
  showTypeSelector(
    nodeId: string,
    handleId: string,
    screenX: number,
    screenY: number,
    options: TypeOption[],
    currentType?: string
  ): void {
    this.typeSelectorState = {
      visible: true,
      nodeId,
      handleId,
      screenX,
      screenY,
      options,
      currentType,
    };
    
    this.typeSelector.setOptions(options);
    this.typeSelector.setPosition(screenX, screenY);
    this.typeSelector.show();
  }
  
  /**
   * 隐藏类型选择器
   */
  hideTypeSelector(): void {
    this.typeSelectorState.visible = false;
    this.typeSelector.hide();
    this.callbacks.onTypeSelectorClose?.();
  }
  
  /**
   * 类型选择器是否可见
   */
  isTypeSelectorVisible(): boolean {
    return this.typeSelectorState.visible;
  }
  
  /**
   * 获取类型选择器状态
   */
  getTypeSelectorState(): TypeSelectorState {
    return { ...this.typeSelectorState };
  }
  
  /**
   * 渲染所有 UI 组件
   */
  render(ctx: CanvasRenderingContext2D): void {
    if (this.typeSelector.visible) {
      this.typeSelector.render(ctx);
    }
  }
  
  /**
   * 处理鼠标按下事件
   * @returns true 如果事件被 UI 组件处理
   */
  handleMouseDown(event: UIMouseEvent): boolean {
    if (this.typeSelector.visible) {
      // 点击外部关闭
      if (!this.typeSelector.hitTest(event.x, event.y)) {
        this.hideTypeSelector();
        return true; // 消费事件
      }
      return this.typeSelector.onMouseDown?.(event) ?? false;
    }
    return false;
  }
  
  /**
   * 处理鼠标移动事件
   */
  handleMouseMove(event: UIMouseEvent): boolean {
    if (this.typeSelector.visible) {
      return this.typeSelector.onMouseMove?.(event) ?? false;
    }
    return false;
  }
  
  /**
   * 处理鼠标抬起事件
   */
  handleMouseUp(event: UIMouseEvent): boolean {
    if (this.typeSelector.visible) {
      return this.typeSelector.onMouseUp?.(event) ?? false;
    }
    return false;
  }
  
  /**
   * 处理滚轮事件
   */
  handleWheel(event: UIWheelEvent): boolean {
    if (this.typeSelector.visible && this.typeSelector.hitTest(event.x, event.y)) {
      return this.typeSelector.onWheel?.(event) ?? false;
    }
    return false;
  }
  
  /**
   * 处理键盘按下事件
   */
  handleKeyDown(event: UIKeyEvent): boolean {
    if (this.typeSelector.visible) {
      return this.typeSelector.onKeyDown?.(event) ?? false;
    }
    return false;
  }
  

  
  /**
   * 销毁
   */
  dispose(): void {
    this.typeSelector.dispose();
  }
}
