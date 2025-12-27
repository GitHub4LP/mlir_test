/**
 * Figma Auto Layout 类型定义
 * 直接从 @figma/plugin-typings 引用，确保与 Figma API 完全一致
 */

/// <reference types="@figma/plugin-typings" />

// ============================================================================
// 从 @figma/plugin-typings 重导出的类型
// ============================================================================

// 注意：@figma/plugin-typings 是全局类型，不能直接 import
// 我们提取需要的类型子集，并添加类型兼容性检查

/** Figma 布局模式 */
export type FigmaLayoutMode = 'NONE' | 'HORIZONTAL' | 'VERTICAL';

/** Figma 尺寸模式 */
export type FigmaSizingMode = 'FIXED' | 'AUTO';

/** Figma 主轴对齐 */
export type FigmaPrimaryAxisAlignItems = 'MIN' | 'CENTER' | 'MAX' | 'SPACE_BETWEEN';

/** Figma 交叉轴对齐 */
export type FigmaCounterAxisAlignItems = 'MIN' | 'CENTER' | 'MAX' | 'BASELINE';

/** Figma 颜色 (0-1 范围) */
export type FigmaColor = RGBA;

/** Figma 纯色填充 */
export type FigmaSolidPaint = SolidPaint;

/** Figma 填充类型 */
export type FigmaFill = Paint;


// ============================================================================
// 类型兼容性断言
// 确保我们使用的类型值与 Figma 官方 API 一致
// 如果 Figma 更新 API，TypeScript 编译会报错
// ============================================================================

// 使用 void 表达式消除 "未使用" 警告，同时保留编译时类型检查
// eslint-disable-next-line @typescript-eslint/no-unused-vars
type AssertExtends<T, _U extends T> = true;

// 检查我们的类型是否与 Figma 官方类型兼容
type _TypeChecks = [
  AssertExtends<FrameNode['layoutMode'], FigmaLayoutMode>,
  AssertExtends<FrameNode['primaryAxisSizingMode'], FigmaSizingMode>,
  AssertExtends<FrameNode['primaryAxisAlignItems'], FigmaPrimaryAxisAlignItems>,
  AssertExtends<FrameNode['counterAxisAlignItems'], FigmaCounterAxisAlignItems>,
];

// 导出一个虚拟常量以使用类型检查（避免 unused 警告）
export const _figmaTypeChecks: _TypeChecks = [true, true, true, true];

// ============================================================================
// 布局配置接口（使用 Figma 原生属性名）
// ============================================================================

/**
 * Figma 布局配置
 * 属性名与 Figma Plugin API 的 FrameNode 属性完全一致
 */
export interface FigmaLayoutConfig {
  // === 布局属性（来自 FrameNode）===
  layoutMode?: FigmaLayoutMode;
  itemSpacing?: number;

  // === 内边距（来自 FrameNode）===
  paddingTop?: number;
  paddingRight?: number;
  paddingBottom?: number;
  paddingLeft?: number;

  // === 尺寸模式（来自 FrameNode）===
  primaryAxisSizingMode?: FigmaSizingMode;
  counterAxisSizingMode?: FigmaSizingMode;
  layoutGrow?: number;

  // === 固定尺寸（来自 FrameNode）===
  width?: number;
  height?: number;
  minWidth?: number;
  maxWidth?: number;
  minHeight?: number;
  maxHeight?: number;

  // === 对齐（来自 FrameNode）===
  primaryAxisAlignItems?: FigmaPrimaryAxisAlignItems;
  counterAxisAlignItems?: FigmaCounterAxisAlignItems;

  // === 圆角（来自 RectangleCornerMixin）===
  cornerRadius?: number;
  topLeftRadius?: number;
  topRightRadius?: number;
  bottomLeftRadius?: number;
  bottomRightRadius?: number;

  // === 填充和描边（来自 GeometryMixin）===
  fills?: readonly Paint[];
  strokes?: readonly Paint[];
  strokeWeight?: number;

  // === 字体（来自 TextNode，扩展用于文本样式）===
  fontSize?: number;
  fontWeight?: number;
  lineHeight?: number | string;
}


// ============================================================================
// 辅助函数：创建 Figma 颜色
// ============================================================================

/**
 * 从 hex 颜色创建 Figma 颜色
 * @param hex - 十六进制颜色，如 '#ff0000' 或 'ff0000'
 * @param alpha - 透明度 0-1，默认 1
 */
export function hexToFigmaColor(hex: string, alpha = 1): RGBA {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16) / 255;
  const g = parseInt(h.substring(2, 4), 16) / 255;
  const b = parseInt(h.substring(4, 6), 16) / 255;
  return { r, g, b, a: alpha };
}

/**
 * 从 RGB 值创建 Figma 颜色
 * @param r - 红色 0-255
 * @param g - 绿色 0-255
 * @param b - 蓝色 0-255
 * @param a - 透明度 0-1，默认 1
 */
export function rgbToFigmaColor(r: number, g: number, b: number, a = 1): RGBA {
  return { r: r / 255, g: g / 255, b: b / 255, a };
}

/**
 * 创建 Figma 纯色填充
 */
export function createSolidPaint(color: RGBA): SolidPaint {
  return {
    type: 'SOLID',
    color: { r: color.r, g: color.g, b: color.b },
    opacity: color.a,
    visible: true,
  };
}

