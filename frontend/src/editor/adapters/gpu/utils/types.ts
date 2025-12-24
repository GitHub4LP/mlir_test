/**
 * GPU 渲染共用类型定义
 */

/** RGBA 颜色（0-1 范围） */
export interface RGBA {
  r: number;
  g: number;
  b: number;
  a: number;
}

/** 2D 点 */
export interface Point {
  x: number;
  y: number;
}
