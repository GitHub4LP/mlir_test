/**
 * UIRenderer - UI 层渲染器
 * 
 * 负责渲染 Canvas UI 组件：
 * - 类型选择器
 * - 属性编辑器
 * - 右键菜单
 * - tooltip
 * 
 * UI 组件在屏幕坐标系中渲染（不受视口变换影响）。
 */

import type { UIComponent, UIMouseEvent, UIKeyEvent, UIWheelEvent } from '../ui/UIComponent';

/**
 * UI 层渲染器
 */
export class UIRenderer {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private dpr: number = 1;
  private components: UIComponent[] = [];
  private focusedComponent: UIComponent | null = null;

  /**
   * 绑定到 Canvas
   */
  bind(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
  }

  /**
   * 解绑
   */
  unbind(): void {
    this.disposeAll();
    this.canvas = null;
    this.ctx = null;
  }

  /**
   * 更新 DPR
   */
  setDPR(dpr: number): void {
    this.dpr = dpr;
  }

  /**
   * 添加组件
   */
  addComponent(component: UIComponent): void {
    this.components.push(component);
  }

  /**
   * 移除组件
   */
  removeComponent(id: string): void {
    const index = this.components.findIndex(c => c.id === id);
    if (index >= 0) {
      const component = this.components[index];
      if (this.focusedComponent === component) {
        this.focusedComponent = null;
      }
      component.dispose();
      this.components.splice(index, 1);
    }
  }

  /**
   * 获取组件
   */
  getComponent(id: string): UIComponent | undefined {
    return this.components.find(c => c.id === id);
  }

  /**
   * 清空所有组件
   */
  clear(): void {
    this.disposeAll();
    this.components = [];
    this.focusedComponent = null;
  }

  /**
   * 是否有可见组件
   */
  hasVisibleComponents(): boolean {
    return this.components.some(c => c.visible);
  }

  /**
   * 清空画布
   */
  clearCanvas(): void {
    if (!this.ctx || !this.canvas) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * 渲染所有组件
   */
  render(): void {
    if (!this.ctx || !this.canvas) return;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.save();

    // 应用 DPI 缩放
    this.ctx.scale(this.dpr, this.dpr);

    // 渲染所有可见组件
    for (const component of this.components) {
      if (component.visible) {
        component.render(this.ctx);
      }
    }

    this.ctx.restore();
  }

  // ============================================================
  // 事件处理
  // ============================================================

  /**
   * 命中测试
   */
  hitTest(x: number, y: number): UIComponent | null {
    // 从后往前检查（后添加的在上层）
    for (let i = this.components.length - 1; i >= 0; i--) {
      const component = this.components[i];
      if (component.visible && component.hitTest(x, y)) {
        return component;
      }
    }
    return null;
  }

  /**
   * 处理鼠标按下
   * @returns 是否被 UI 层处理
   */
  handleMouseDown(event: UIMouseEvent): boolean {
    // 从后往前检查
    for (let i = this.components.length - 1; i >= 0; i--) {
      const component = this.components[i];
      if (component.visible && component.hitTest(event.x, event.y)) {
        // 设置焦点
        if (this.focusedComponent !== component) {
          this.focusedComponent?.onBlur?.();
          this.focusedComponent = component;
          component.onFocus?.();
        }
        
        if (component.onMouseDown?.(event)) {
          return true;
        }
      }
    }

    // 点击空白处，清除焦点
    if (this.focusedComponent) {
      this.focusedComponent.onBlur?.();
      this.focusedComponent = null;
    }

    return false;
  }

  /**
   * 处理鼠标移动
   * @returns 是否需要重绘
   */
  handleMouseMove(event: UIMouseEvent): boolean {
    let needsRedraw = false;
    
    for (let i = this.components.length - 1; i >= 0; i--) {
      const component = this.components[i];
      if (component.visible && component.onMouseMove?.(event)) {
        needsRedraw = true;
      }
    }

    return needsRedraw;
  }

  /**
   * 处理鼠标抬起
   * @returns 是否被 UI 层处理
   */
  handleMouseUp(event: UIMouseEvent): boolean {
    for (let i = this.components.length - 1; i >= 0; i--) {
      const component = this.components[i];
      if (component.visible && component.onMouseUp?.(event)) {
        return true;
      }
    }
    return false;
  }

  /**
   * 处理滚轮
   * @returns 是否被 UI 层处理
   */
  handleWheel(event: UIWheelEvent): boolean {
    for (let i = this.components.length - 1; i >= 0; i--) {
      const component = this.components[i];
      if (component.visible && component.hitTest(event.x, event.y) && component.onWheel?.(event)) {
        return true;
      }
    }
    return false;
  }

  /**
   * 处理键盘按下
   * @returns 是否被 UI 层处理
   */
  handleKeyDown(event: UIKeyEvent): boolean {
    if (this.focusedComponent?.onKeyDown?.(event)) {
      return true;
    }
    return false;
  }

  /**
   * 处理键盘抬起
   * @returns 是否被 UI 层处理
   */
  handleKeyUp(event: UIKeyEvent): boolean {
    if (this.focusedComponent?.onKeyUp?.(event)) {
      return true;
    }
    return false;
  }

  // ============================================================
  // 私有方法
  // ============================================================

  private disposeAll(): void {
    for (const component of this.components) {
      component.dispose();
    }
  }
}
