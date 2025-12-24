/**
 * IContentRenderer - 内容层渲染器接口
 * 
 * 定义内容层渲染器的统一接口，支持：
 * - Canvas 2D (ContentRenderer)
 * - WebGL (WebGLContentRenderer)
 * - WebGPU (WebGPUContentRenderer)
 */

import type { RenderData } from '../../../core/RenderData';

/**
 * 内容层渲染器接口
 */
export interface IContentRenderer {
  /**
   * 初始化渲染器
   * @param container 容器元素
   * @returns 是否初始化成功
   */
  init(container: HTMLElement): Promise<boolean>;

  /**
   * 销毁渲染器
   */
  dispose(): void;

  /**
   * 是否已初始化
   */
  isInitialized(): boolean;

  /**
   * 调整尺寸
   */
  resize(): void;

  /**
   * 渲染完整内容（图形 + 文字）
   */
  render(data: RenderData): void;

  /**
   * 仅渲染图形（用于拖拽/缩放优化）
   */
  renderGraphicsOnly(data: RenderData): void;

  /**
   * 获取当前 LOD 级别
   */
  getLODLevel(): string;
}

/** 渲染后端类型 */
export type ContentBackendType = 'canvas2d' | 'webgl' | 'webgpu';

/**
 * 检查渲染后端是否可用
 */
export function isBackendAvailable(type: ContentBackendType): boolean {
  switch (type) {
    case 'canvas2d':
      return true;
    case 'webgl':
      return typeof document !== 'undefined' &&
             !!document.createElement('canvas').getContext('webgl');
    case 'webgpu':
      return typeof navigator !== 'undefined' && 'gpu' in navigator;
    default:
      return false;
  }
}

/**
 * 获取最佳可用后端
 */
export function getBestAvailableBackend(): ContentBackendType {
  // 优先级：WebGPU > WebGL > Canvas 2D
  if (isBackendAvailable('webgpu')) return 'webgpu';
  if (isBackendAvailable('webgl')) return 'webgl';
  return 'canvas2d';
}
