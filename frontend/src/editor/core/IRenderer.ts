/**
 * 渲染器接口
 * 
 * 所有渲染器（Canvas、WebGL、WebGPU、React Flow、Vue Flow）都实现此接口。
 * 控制器通过此接口与渲染器交互，无需关心具体实现。
 * 
 * 设计原则：
 * - 接口尽可能简单，便于实现新渲染器
 * - 渲染器不包含任何业务逻辑
 * - 渲染器不直接访问应用状态（stores）
 * - 只负责：显示 RenderData、收集 RawInput
 */

import type { RenderData } from './RenderData';
import type { RawInput, RawInputCallback } from './input';

/**
 * 渲染器接口
 */
export interface IRenderer {
  // ============================================================
  // 生命周期
  // ============================================================

  /**
   * 挂载到 DOM 容器
   * @param container - 渲染器将在此容器内创建渲染元素
   */
  mount(container: HTMLElement): void;

  /**
   * 卸载
   * 清理所有资源，移除事件监听器
   */
  unmount(): void;

  /**
   * 容器尺寸变化
   * 渲染器应从容器获取实际尺寸，参数为可选提示
   * @param width - 新宽度（像素，可选）
   * @param height - 新高度（像素，可选）
   */
  resize(width?: number, height?: number): void;

  // ============================================================
  // 渲染
  // ============================================================

  /**
   * 渲染
   * @param data - 预计算的渲染数据
   */
  render(data: RenderData): void;

  // ============================================================
  // 输入
  // ============================================================

  /**
   * 注册原始输入回调
   * 渲染器收集用户输入后，通过此回调通知控制器
   * @param callback - 输入回调函数
   */
  onInput(callback: RawInputCallback): void;

  // ============================================================
  // 元信息
  // ============================================================

  /**
   * 获取渲染器名称
   * @returns 渲染器名称（如 'Canvas', 'WebGL', 'ReactFlow'）
   */
  getName(): string;

  /**
   * 检查渲染器是否可用
   * 某些渲染器可能需要特定浏览器支持（如 WebGPU）
   * @returns 是否可用
   */
  isAvailable(): boolean;
}

/**
 * 渲染器基类（可选）
 * 提供一些通用实现，具体渲染器可以继承此类
 */
export abstract class BaseRenderer implements IRenderer {
  protected container: HTMLElement | null = null;
  protected inputCallback: RawInputCallback | null = null;
  protected width: number = 0;
  protected height: number = 0;

  mount(container: HTMLElement): void {
    this.container = container;
    this.width = container.clientWidth;
    this.height = container.clientHeight;
    this.onMount();
  }

  unmount(): void {
    this.onUnmount();
    this.container = null;
    this.inputCallback = null;
  }

  resize(width?: number, height?: number): void {
    if (width !== undefined) this.width = width;
    if (height !== undefined) this.height = height;
    this.onResize(this.width, this.height);
  }

  onInput(callback: RawInputCallback): void {
    this.inputCallback = callback;
  }

  /** 发送原始输入到控制器 */
  protected emitInput(input: RawInput): void {
    this.inputCallback?.(input);
  }

  // 子类实现
  abstract render(data: RenderData): void;
  abstract getName(): string;
  abstract isAvailable(): boolean;

  /** 挂载时调用（子类实现） */
  protected abstract onMount(): void;

  /** 卸载时调用（子类实现） */
  protected abstract onUnmount(): void;

  /** 尺寸变化时调用（子类实现） */
  protected abstract onResize(width: number, height: number): void;
}
