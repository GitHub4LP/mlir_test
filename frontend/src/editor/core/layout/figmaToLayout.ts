/**
 * Figma 布局配置 → Canvas/GPU 布局数据转换
 * 用于 Canvas 2D、WebGL、WebGPU 渲染器
 */

import type {
  FigmaLayoutConfig,
  FigmaLayoutMode,
  FigmaPrimaryAxisAlignItems,
  FigmaCounterAxisAlignItems,
} from './figmaTypes';

// ============================================================================
// Canvas/GPU 渲染器用的布局数据类型
// ============================================================================

/** 布局方向 */
export type LayoutDirection = 'horizontal' | 'vertical' | 'none';

/** 主轴对齐 */
export type LayoutJustify = 'start' | 'center' | 'end' | 'space-between';

/** 交叉轴对齐 */
export type LayoutAlign = 'start' | 'center' | 'end' | 'baseline';

/** 归一化的 RGBA 颜色 (0-1 范围) */
export interface LayoutColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

/** 内边距 */
export interface LayoutPadding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/** 圆角 */
export interface LayoutCornerRadius {
  topLeft: number;
  topRight: number;
  bottomRight: number;
  bottomLeft: number;
}


/** Canvas/GPU 渲染器用的布局数据 */
export interface LayoutData {
  // 布局
  direction: LayoutDirection;
  gap: number;
  padding: LayoutPadding;
  justify: LayoutJustify;
  align: LayoutAlign;

  // 尺寸
  width: number | 'auto';
  height: number | 'auto';
  minWidth?: number;
  maxWidth?: number;
  minHeight?: number;
  maxHeight?: number;
  flexGrow: number;

  // 圆角
  cornerRadius: LayoutCornerRadius;

  // 颜色
  fill?: LayoutColor;
  stroke?: LayoutColor;
  strokeWidth: number;

  // 字体
  fontSize?: number;
  fontWeight?: number;
  lineHeight?: number;
}

// ============================================================================
// 映射函数
// ============================================================================

function mapLayoutMode(mode: FigmaLayoutMode | undefined): LayoutDirection {
  switch (mode) {
    case 'HORIZONTAL':
      return 'horizontal';
    case 'VERTICAL':
      return 'vertical';
    default:
      return 'none';
  }
}

function mapPrimaryAlign(align: FigmaPrimaryAxisAlignItems | undefined): LayoutJustify {
  switch (align) {
    case 'MIN':
      return 'start';
    case 'CENTER':
      return 'center';
    case 'MAX':
      return 'end';
    case 'SPACE_BETWEEN':
      return 'space-between';
    default:
      return 'start';
  }
}

function mapCounterAlign(align: FigmaCounterAxisAlignItems | undefined): LayoutAlign {
  switch (align) {
    case 'MIN':
      return 'start';
    case 'CENTER':
      return 'center';
    case 'MAX':
      return 'end';
    case 'BASELINE':
      return 'baseline';
    default:
      return 'start';
  }
}


/**
 * 从 Figma fills 提取第一个可见纯色
 */
function extractFillColor(fills: readonly Paint[] | undefined): LayoutColor | undefined {
  if (!fills) return undefined;
  for (const fill of fills) {
    if (fill.type === 'SOLID' && fill.visible !== false) {
      const { color, opacity = 1 } = fill;
      return { r: color.r, g: color.g, b: color.b, a: opacity };
    }
  }
  return undefined;
}

/**
 * 获取圆角
 */
function getCornerRadius(config: FigmaLayoutConfig): LayoutCornerRadius {
  const base = config.cornerRadius ?? 0;
  return {
    topLeft: config.topLeftRadius ?? base,
    topRight: config.topRightRadius ?? base,
    bottomRight: config.bottomRightRadius ?? base,
    bottomLeft: config.bottomLeftRadius ?? base,
  };
}

// ============================================================================
// 主转换函数
// ============================================================================

/**
 * Figma 布局配置 → Canvas/GPU 布局数据
 */
export function figmaToLayout(config: FigmaLayoutConfig): LayoutData {
  return {
    // 布局
    direction: mapLayoutMode(config.layoutMode),
    gap: config.itemSpacing ?? 0,
    padding: {
      top: config.paddingTop ?? 0,
      right: config.paddingRight ?? 0,
      bottom: config.paddingBottom ?? 0,
      left: config.paddingLeft ?? 0,
    },
    justify: mapPrimaryAlign(config.primaryAxisAlignItems),
    align: mapCounterAlign(config.counterAxisAlignItems),

    // 尺寸
    width: config.primaryAxisSizingMode === 'AUTO' ? 'auto' : (config.width ?? 'auto'),
    height: config.counterAxisSizingMode === 'AUTO' ? 'auto' : (config.height ?? 'auto'),
    minWidth: config.minWidth,
    maxWidth: config.maxWidth,
    minHeight: config.minHeight,
    maxHeight: config.maxHeight,
    flexGrow: config.layoutGrow ?? 0,

    // 圆角
    cornerRadius: getCornerRadius(config),

    // 颜色
    fill: extractFillColor(config.fills),
    stroke: extractFillColor(config.strokes),
    strokeWidth: config.strokeWeight ?? 0,

    // 字体
    fontSize: config.fontSize,
    fontWeight: config.fontWeight,
    lineHeight: typeof config.lineHeight === 'number' ? config.lineHeight : undefined,
  };
}
