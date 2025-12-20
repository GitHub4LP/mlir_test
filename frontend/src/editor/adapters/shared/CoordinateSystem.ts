/**
 * 统一坐标系统模块
 * 
 * 所有渲染器使用相同的坐标转换公式，确保与 React Flow 一致。
 * 坐标系统：原点在左上角，X 轴向右，Y 轴向下。
 */

import type { Viewport } from './Viewport';

/** 2D 点 */
export interface Point {
  x: number;
  y: number;
}

/**
 * 屏幕坐标转画布坐标（世界坐标）
 * 
 * 公式：canvasX = (screenX - viewport.x) / viewport.zoom
 * 
 * @param screenX 屏幕坐标 X（相对于渲染器容器左上角）
 * @param screenY 屏幕坐标 Y（相对于渲染器容器左上角）
 * @param viewport 视口状态
 * @returns 画布坐标
 */
export function screenToCanvas(
  screenX: number,
  screenY: number,
  viewport: Viewport
): Point {
  return {
    x: (screenX - viewport.x) / viewport.zoom,
    y: (screenY - viewport.y) / viewport.zoom,
  };
}

/**
 * 画布坐标（世界坐标）转屏幕坐标
 * 
 * 公式：screenX = canvasX * viewport.zoom + viewport.x
 * 
 * @param canvasX 画布坐标 X
 * @param canvasY 画布坐标 Y
 * @param viewport 视口状态
 * @returns 屏幕坐标（相对于渲染器容器左上角）
 */
export function canvasToScreen(
  canvasX: number,
  canvasY: number,
  viewport: Viewport
): Point {
  return {
    x: canvasX * viewport.zoom + viewport.x,
    y: canvasY * viewport.zoom + viewport.y,
  };
}


/**
 * 从 DOM 事件获取相对于元素的屏幕坐标
 * 
 * @param event 鼠标/指针事件
 * @param element 目标元素（通常是 canvas）
 * @returns 相对于元素左上角的屏幕坐标
 */
export function getScreenCoordinates(
  event: MouseEvent | PointerEvent,
  element: HTMLElement
): Point {
  const rect = element.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

/**
 * 从 DOM 事件获取画布坐标
 * 
 * @param event 鼠标/指针事件
 * @param element 目标元素
 * @param viewport 视口状态
 * @returns 画布坐标
 */
export function getCanvasCoordinates(
  event: MouseEvent | PointerEvent,
  element: HTMLElement,
  viewport: Viewport
): Point {
  const screen = getScreenCoordinates(event, element);
  return screenToCanvas(screen.x, screen.y, viewport);
}

/**
 * 计算缩放后的新视口位置
 * 保持指定屏幕坐标点对应的画布坐标不变
 * 
 * @param viewport 当前视口
 * @param newZoom 新的缩放比例
 * @param screenX 缩放中心屏幕坐标 X
 * @param screenY 缩放中心屏幕坐标 Y
 * @returns 新的视口状态
 */
export function zoomAtPoint(
  viewport: Viewport,
  newZoom: number,
  screenX: number,
  screenY: number
): Viewport {
  // 计算缩放中心的画布坐标
  const canvasX = (screenX - viewport.x) / viewport.zoom;
  const canvasY = (screenY - viewport.y) / viewport.zoom;
  
  // 计算新的视口偏移，保持画布坐标不变
  return {
    x: screenX - canvasX * newZoom,
    y: screenY - canvasY * newZoom,
    zoom: newZoom,
  };
}

/**
 * 限制缩放比例在有效范围内
 * 
 * @param zoom 缩放比例
 * @param min 最小值（默认 0.1）
 * @param max 最大值（默认 4）
 * @returns 限制后的缩放比例
 */
export function clampZoom(zoom: number, min = 0.1, max = 4): number {
  return Math.max(min, Math.min(max, zoom));
}
