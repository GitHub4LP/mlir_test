/**
 * UIComponent - Canvas UI 组件接口
 * 
 * 所有 Canvas UI 组件的基础接口。
 * UI 组件在屏幕坐标系中渲染（不受视口变换影响）。
 */

/** 组件边界 */
export interface Bounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** 鼠标事件 */
export interface UIMouseEvent {
  /** 屏幕坐标 X */
  x: number;
  /** 屏幕坐标 Y */
  y: number;
  /** 按钮：0=左键, 1=中键, 2=右键 */
  button: number;
}

/** 键盘事件 */
export interface UIKeyEvent {
  key: string;
  code: string;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey: boolean;
}

/** 滚轮事件 */
export interface UIWheelEvent {
  x: number;
  y: number;
  deltaX: number;
  deltaY: number;
}

/**
 * UI 组件接口
 */
export interface UIComponent {
  /** 组件 ID */
  id: string;

  /** 是否可见 */
  visible: boolean;

  /** 获取边界 */
  getBounds(): Bounds;

  /** 设置位置 */
  setPosition(x: number, y: number): void;

  /** 设置尺寸 */
  setSize(width: number, height: number): void;

  /** 渲染 */
  render(ctx: CanvasRenderingContext2D): void;

  /** 命中测试 */
  hitTest(x: number, y: number): boolean;

  /** 鼠标按下 */
  onMouseDown?(event: UIMouseEvent): boolean;

  /** 鼠标移动 */
  onMouseMove?(event: UIMouseEvent): boolean;

  /** 鼠标抬起 */
  onMouseUp?(event: UIMouseEvent): boolean;

  /** 鼠标滚轮 */
  onWheel?(event: UIWheelEvent): boolean;

  /** 键盘按下 */
  onKeyDown?(event: UIKeyEvent): boolean;

  /** 键盘抬起 */
  onKeyUp?(event: UIKeyEvent): boolean;

  /** 获取焦点 */
  onFocus?(): void;

  /** 失去焦点 */
  onBlur?(): void;

  /** 销毁 */
  dispose(): void;
}

/**
 * UI 组件基类
 */
export abstract class BaseUIComponent implements UIComponent {
  id: string;
  visible: boolean = true;
  protected x: number = 0;
  protected y: number = 0;
  protected width: number = 0;
  protected height: number = 0;

  constructor(id: string) {
    this.id = id;
  }

  getBounds(): Bounds {
    return { x: this.x, y: this.y, width: this.width, height: this.height };
  }

  setPosition(x: number, y: number): void {
    this.x = x;
    this.y = y;
  }

  setSize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  hitTest(x: number, y: number): boolean {
    return (
      x >= this.x &&
      x <= this.x + this.width &&
      y >= this.y &&
      y <= this.y + this.height
    );
  }

  abstract render(ctx: CanvasRenderingContext2D): void;

  dispose(): void {
    // 子类可覆盖
  }
}

/**
 * 容器组件基类
 */
export abstract class ContainerComponent extends BaseUIComponent {
  protected children: UIComponent[] = [];

  addChild(child: UIComponent): void {
    this.children.push(child);
  }

  removeChild(id: string): void {
    const index = this.children.findIndex(c => c.id === id);
    if (index >= 0) {
      this.children[index].dispose();
      this.children.splice(index, 1);
    }
  }

  getChild(id: string): UIComponent | undefined {
    return this.children.find(c => c.id === id);
  }

  render(ctx: CanvasRenderingContext2D): void {
    if (!this.visible) return;
    this.renderSelf(ctx);
    for (const child of this.children) {
      if (child.visible) {
        child.render(ctx);
      }
    }
  }

  protected abstract renderSelf(ctx: CanvasRenderingContext2D): void;

  hitTest(x: number, y: number): boolean {
    if (!this.visible) return false;
    // 先检查子组件
    for (let i = this.children.length - 1; i >= 0; i--) {
      if (this.children[i].visible && this.children[i].hitTest(x, y)) {
        return true;
      }
    }
    // 再检查自身
    return super.hitTest(x, y);
  }

  onMouseDown(event: UIMouseEvent): boolean {
    for (let i = this.children.length - 1; i >= 0; i--) {
      const child = this.children[i];
      if (child.visible && child.hitTest(event.x, event.y) && child.onMouseDown?.(event)) {
        return true;
      }
    }
    return false;
  }

  onMouseMove(event: UIMouseEvent): boolean {
    for (let i = this.children.length - 1; i >= 0; i--) {
      const child = this.children[i];
      if (child.visible && child.onMouseMove?.(event)) {
        return true;
      }
    }
    return false;
  }

  onMouseUp(event: UIMouseEvent): boolean {
    for (let i = this.children.length - 1; i >= 0; i--) {
      const child = this.children[i];
      if (child.visible && child.onMouseUp?.(event)) {
        return true;
      }
    }
    return false;
  }

  onWheel(event: UIWheelEvent): boolean {
    for (let i = this.children.length - 1; i >= 0; i--) {
      const child = this.children[i];
      if (child.visible && child.hitTest(event.x, event.y) && child.onWheel?.(event)) {
        return true;
      }
    }
    return false;
  }

  dispose(): void {
    for (const child of this.children) {
      child.dispose();
    }
    this.children = [];
  }
}
