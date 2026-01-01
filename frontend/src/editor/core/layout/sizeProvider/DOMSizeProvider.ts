/**
 * DOMSizeProvider
 * 先创建后测量模式，用于 DOM 渲染器
 */

import type { DOMSizeProvider as IDOMSizeProvider } from './types';
import type { Size } from '../types';
import { getContainerConfig } from '../tokens';

/** 元素缓存项 */
interface CacheEntry {
  element: HTMLElement;
  size: Size;
}

/**
 * DOMSizeProvider 实现
 * 在 offscreen 容器中创建 DOM 元素，测量后缓存尺寸
 */
export class DOMSizeProvider implements IDOMSizeProvider {
  /** offscreen 容器 */
  private container: HTMLDivElement | null = null;
  
  /** 元素缓存 */
  private cache = new Map<string, CacheEntry>();
  
  /** 元素工厂函数 */
  private factories = new Map<string, (content?: string) => HTMLElement>();

  constructor() {
    this.createContainer();
    this.registerDefaultFactories();
  }

  /**
   * 创建 offscreen 容器
   */
  private createContainer(): void {
    if (typeof document === 'undefined') return;
    
    this.container = document.createElement('div');
    this.container.style.cssText = `
      position: absolute;
      visibility: hidden;
      pointer-events: none;
      top: -9999px;
      left: -9999px;
    `;
    document.body.appendChild(this.container);
  }

  /**
   * 注册默认元素工厂函数
   * 从配置中获取 className 并应用
   */
  private registerDefaultFactories(): void {
    // editableName 工厂
    this.registerFactory('editableName', (content) => {
      const config = getContainerConfig('editableName');
      const el = document.createElement('input');
      el.type = 'text';
      el.value = content ?? '';
      if ('className' in config && config.className) {
        el.className = config.className as string;
      }
      return el;
    });

    // typeLabel 工厂
    this.registerFactory('typeLabel', (content) => {
      const config = getContainerConfig('typeLabel');
      const el = document.createElement('span');
      el.textContent = content ?? '';
      if ('className' in config && config.className) {
        el.className = config.className as string;
      }
      return el;
    });

    // button 工厂
    this.registerFactory('button', () => {
      const config = getContainerConfig('button');
      const el = document.createElement('button');
      if ('className' in config && config.className) {
        el.className = config.className as string;
      }
      return el;
    });

    // handle 工厂
    this.registerFactory('handle', () => {
      const config = getContainerConfig('handle');
      const el = document.createElement('div');
      if ('className' in config && config.className) {
        el.className = config.className as string;
      }
      return el;
    });
  }

  /**
   * 注册元素工厂函数
   */
  registerFactory(type: string, factory: (content?: string) => HTMLElement): void {
    this.factories.set(type, factory);
  }

  /**
   * 获取组件尺寸
   * 如果缓存中没有，则创建元素并测量
   */
  getSize(type: string, content?: string): Size {
    const cacheKey = `${type}|${content ?? ''}`;
    
    // 检查缓存
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!.size;
    }
    
    // 创建元素
    const element = this.createElement(type, content);
    if (!element || !this.container) {
      // 返回默认尺寸
      return { width: 0, height: 0 };
    }
    
    // 添加到 offscreen 容器
    this.container.appendChild(element);
    
    // 测量尺寸
    const rect = element.getBoundingClientRect();
    const size: Size = {
      width: rect.width,
      height: rect.height,
    };
    
    // 缓存
    this.cache.set(cacheKey, { element, size });
    
    return size;
  }

  /**
   * 创建 DOM 元素
   */
  private createElement(type: string, content?: string): HTMLElement | null {
    const factory = this.factories.get(type);
    if (!factory) {
      console.warn(`DOMSizeProvider: No factory registered for type "${type}"`);
      return null;
    }
    return factory(content);
  }

  /**
   * 获取已创建的 DOM 元素
   */
  getElement(id: string): HTMLElement | null {
    const entry = this.cache.get(id);
    return entry?.element ?? null;
  }

  /**
   * 将元素移动到目标容器
   */
  moveElementTo(id: string, container: HTMLElement): void {
    const entry = this.cache.get(id);
    if (entry?.element) {
      container.appendChild(entry.element);
    }
  }

  /**
   * 清除缓存
   */
  clearCache(): void {
    // 移除所有缓存的元素
    for (const entry of this.cache.values()) {
      entry.element.remove();
    }
    this.cache.clear();
  }

  /**
   * 销毁
   */
  dispose(): void {
    this.clearCache();
    this.container?.remove();
    this.container = null;
    this.factories.clear();
  }
}

/** 默认 DOMSizeProvider 实例 */
export const domSizeProvider = new DOMSizeProvider();
