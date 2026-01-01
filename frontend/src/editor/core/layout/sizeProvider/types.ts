/**
 * SizeProvider 接口定义
 * 统一尺寸获取接口，支持 DOM 测量和配置计算两种模式
 */

import type { Size } from '../types';

/**
 * 尺寸提供者接口
 * Layout Engine 通过此接口获取交互组件尺寸
 */
export interface SizeProvider {
  /**
   * 获取组件尺寸
   * @param type - 组件类型（如 'editableName', 'typeLabel', 'button'）
   * @param content - 文本内容（可选，用于计算文本宽度）
   * @returns 组件尺寸
   */
  getSize(type: string, content?: string): Size;

  /**
   * 清除缓存（可选）
   * 用于在配置变更或字体加载后重新计算尺寸
   */
  clearCache?(): void;
}

/**
 * DOM 尺寸提供者接口（扩展）
 * 支持 DOM 元素的创建、测量和复用
 */
export interface DOMSizeProvider extends SizeProvider {
  /**
   * 获取已创建的 DOM 元素
   * @param id - 元素唯一标识
   * @returns DOM 元素或 null
   */
  getElement(id: string): HTMLElement | null;

  /**
   * 将元素移动到目标容器
   * @param id - 元素唯一标识
   * @param container - 目标容器
   */
  moveElementTo(id: string, container: HTMLElement): void;

  /**
   * 注册元素工厂函数
   * @param type - 组件类型
   * @param factory - 创建 DOM 元素的工厂函数
   */
  registerFactory(type: string, factory: (content?: string) => HTMLElement): void;

  /**
   * 销毁所有缓存的元素
   */
  dispose(): void;
}

/**
 * 交互组件类型
 */
export type InteractiveComponentType = 
  | 'editableName'
  | 'typeLabel'
  | 'button'
  | 'handle';
