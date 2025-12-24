/**
 * IGraphicsRenderer - GPU 图形渲染器接口
 * 
 * 定义 GPU 图形渲染器的统一接口，支持：
 * - WebGL (WebGLGraphics)
 * - WebGPU (WebGPUGraphics)
 */

import type {
  RenderRect,
  RenderCircle,
  RenderTriangle,
  RenderPath,
  Viewport,
} from '../../../core/RenderData';

/**
 * GPU 图形渲染器接口
 */
export interface IGraphicsRenderer {
  /**
   * 初始化渲染器
   * @param canvas Canvas 元素
   * @returns 是否初始化成功
   */
  init(canvas: HTMLCanvasElement): Promise<boolean>;

  /**
   * 销毁渲染器
   */
  dispose(): void;

  /**
   * 是否已初始化
   */
  isInitialized(): boolean;

  /**
   * 设置视口
   */
  setViewport(viewport: Viewport): void;

  /**
   * 调整尺寸
   */
  resize(width: number, height: number): void;

  /**
   * 清空画布
   */
  clear(): void;

  /**
   * 渲染矩形
   */
  renderRects(rects: RenderRect[]): void;

  /**
   * 渲染圆形
   */
  renderCircles(circles: RenderCircle[]): void;

  /**
   * 渲染三角形
   */
  renderTriangles(triangles: RenderTriangle[]): void;

  /**
   * 渲染路径（连线）
   */
  renderPaths(paths: RenderPath[]): void;
}
