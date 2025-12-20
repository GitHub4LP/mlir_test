/**
 * 统一视口接口
 * 
 * 所有渲染器使用相同的视口语义，确保切换时状态保持一致。
 */

/** 视口状态 */
export interface Viewport {
  /** 视口偏移 X（屏幕像素） */
  x: number;
  /** 视口偏移 Y（屏幕像素） */
  y: number;
  /** 缩放比例 (0.1 ~ 4.0) */
  zoom: number;
}

/** 默认视口状态 */
export const defaultViewport: Viewport = {
  x: 0,
  y: 0,
  zoom: 1,
};

/**
 * 创建视口副本
 */
export function cloneViewport(viewport: Viewport): Viewport {
  return {
    x: viewport.x,
    y: viewport.y,
    zoom: viewport.zoom,
  };
}

/**
 * 比较两个视口是否相等
 */
export function viewportsEqual(a: Viewport, b: Viewport): boolean {
  return a.x === b.x && a.y === b.y && a.zoom === b.zoom;
}

/**
 * 从 React Flow 视口转换
 * React Flow 的 viewport 语义与我们相同，直接复制即可
 */
export function fromReactFlowViewport(rfViewport: {
  x: number;
  y: number;
  zoom: number;
}): Viewport {
  return {
    x: rfViewport.x,
    y: rfViewport.y,
    zoom: rfViewport.zoom,
  };
}

/**
 * 转换为 React Flow 视口格式
 */
export function toReactFlowViewport(viewport: Viewport): {
  x: number;
  y: number;
  zoom: number;
} {
  return {
    x: viewport.x,
    y: viewport.y,
    zoom: viewport.zoom,
  };
}
