/**
 * Figma 布局配置 → CSS 样式转换
 * 用于 ReactFlow、VueFlow 等 DOM 渲染器
 * 
 * Figma Auto Layout 与 CSS Flexbox 的核心映射：
 * 
 * | Figma 概念 | CSS 等价 |
 * |-----------|---------|
 * | layoutMode: HORIZONTAL | flex-direction: row |
 * | layoutMode: VERTICAL | flex-direction: column |
 * | primaryAxisSizingMode: AUTO | 主轴方向不设置尺寸（hug contents） |
 * | primaryAxisSizingMode: FIXED | 主轴方向设置固定尺寸 |
 * | counterAxisSizingMode: AUTO | 交叉轴方向不设置尺寸 |
 * | counterAxisSizingMode: FIXED | 交叉轴方向设置固定尺寸 |
 * | layoutGrow > 0 | flex-grow + flex-basis: 0（关键！） |
 * | itemSpacing | gap |
 * 
 * 关键点：
 * 1. layoutGrow 需要配合 flex-basis: 0 才能正确分配空间
 * 2. 父容器需要有明确宽度，子元素的 flex-grow 才能生效
 */

import type { CSSProperties } from 'react';
import type {
  FigmaLayoutConfig,
  FigmaLayoutMode,
  FigmaPrimaryAxisAlignItems,
  FigmaCounterAxisAlignItems,
  FigmaSizingMode,
} from './figmaTypes';

// ============================================================================
// 颜色转换
// ============================================================================

/**
 * Figma RGBA 颜色转 CSS 字符串
 * Figma 使用 0-1 范围，CSS 使用 0-255
 */
export function figmaColorToCSS(color: RGBA): string {
  const r = Math.round(color.r * 255);
  const g = Math.round(color.g * 255);
  const b = Math.round(color.b * 255);
  if (color.a === 1) {
    return `rgb(${r}, ${g}, ${b})`;
  }
  return `rgba(${r}, ${g}, ${b}, ${color.a})`;
}

/**
 * Figma Paint 转 CSS 背景/颜色
 * 目前只支持 SOLID 类型
 */
export function figmaPaintToCSS(paint: Paint): string | undefined {
  if (paint.type === 'SOLID' && paint.visible !== false) {
    const { color, opacity = 1 } = paint;
    return figmaColorToCSS({ ...color, a: opacity });
  }
  // 渐变等其他类型暂不支持
  return undefined;
}

/**
 * Figma fills 数组转 CSS 背景
 * 取第一个可见的纯色填充
 */
export function figmaFillsToCSS(fills: readonly Paint[] | undefined): string | undefined {
  if (!fills) return undefined;
  for (const fill of fills) {
    const css = figmaPaintToCSS(fill);
    if (css) return css;
  }
  return undefined;
}


// ============================================================================
// 布局模式转换
// ============================================================================

/**
 * Figma layoutMode → CSS flexDirection
 */
function mapLayoutModeToFlexDirection(mode: FigmaLayoutMode | undefined): CSSProperties['flexDirection'] {
  switch (mode) {
    case 'HORIZONTAL':
      return 'row';
    case 'VERTICAL':
      return 'column';
    case 'NONE':
    default:
      return undefined;
  }
}

/**
 * Figma primaryAxisAlignItems → CSS justifyContent
 */
function mapPrimaryAlignToJustify(
  align: FigmaPrimaryAxisAlignItems | undefined
): CSSProperties['justifyContent'] {
  switch (align) {
    case 'MIN':
      return 'flex-start';
    case 'CENTER':
      return 'center';
    case 'MAX':
      return 'flex-end';
    case 'SPACE_BETWEEN':
      return 'space-between';
    default:
      return undefined;
  }
}

/**
 * Figma counterAxisAlignItems → CSS alignItems
 */
function mapCounterAlignToAlignItems(
  align: FigmaCounterAxisAlignItems | undefined
): CSSProperties['alignItems'] {
  switch (align) {
    case 'MIN':
      return 'flex-start';
    case 'CENTER':
      return 'center';
    case 'MAX':
      return 'flex-end';
    case 'BASELINE':
      return 'baseline';
    default:
      return undefined;
  }
}

/**
 * Figma sizingMode → CSS width/height 值
 * AUTO = hug contents (不设置，让 flexbox 自动计算)
 * FIXED = 使用固定值
 */
function mapSizingModeToCSS(
  mode: FigmaSizingMode | undefined,
  fixedValue: number | undefined
): string | number | undefined {
  switch (mode) {
    case 'AUTO':
      // AUTO 表示 hug contents，不需要设置 width/height
      // 让 flexbox 自动计算尺寸
      return undefined;
    case 'FIXED':
      return fixedValue;
    default:
      return undefined;
  }
}

// ============================================================================
// 圆角转换
// ============================================================================

/**
 * 获取 CSS borderRadius
 * 支持统一圆角和四角独立圆角
 */
function getBorderRadius(config: FigmaLayoutConfig): string | number | undefined {
  const { cornerRadius, topLeftRadius, topRightRadius, bottomLeftRadius, bottomRightRadius } = config;

  // 如果有独立圆角，使用四角格式
  if (
    topLeftRadius !== undefined ||
    topRightRadius !== undefined ||
    bottomLeftRadius !== undefined ||
    bottomRightRadius !== undefined
  ) {
    const tl = topLeftRadius ?? cornerRadius ?? 0;
    const tr = topRightRadius ?? cornerRadius ?? 0;
    const br = bottomRightRadius ?? cornerRadius ?? 0;
    const bl = bottomLeftRadius ?? cornerRadius ?? 0;
    return `${tl}px ${tr}px ${br}px ${bl}px`;
  }

  // 统一圆角
  return cornerRadius;
}


// ============================================================================
// 主转换函数
// ============================================================================

/**
 * Figma 布局配置 → CSS 样式
 * 用于 ReactFlow、VueFlow 等 DOM 渲染器
 */
export function figmaToCSS(config: FigmaLayoutConfig): CSSProperties {
  const style: CSSProperties = {};

  // 布局模式
  if (config.layoutMode && config.layoutMode !== 'NONE') {
    style.display = 'flex';
    style.flexDirection = mapLayoutModeToFlexDirection(config.layoutMode);
  }

  // 间距
  if (config.itemSpacing !== undefined) {
    style.gap = config.itemSpacing;
  }

  // 内边距
  if (config.paddingTop !== undefined) style.paddingTop = config.paddingTop;
  if (config.paddingRight !== undefined) style.paddingRight = config.paddingRight;
  if (config.paddingBottom !== undefined) style.paddingBottom = config.paddingBottom;
  if (config.paddingLeft !== undefined) style.paddingLeft = config.paddingLeft;

  // 对齐
  if (config.primaryAxisAlignItems) {
    style.justifyContent = mapPrimaryAlignToJustify(config.primaryAxisAlignItems);
  }
  if (config.counterAxisAlignItems) {
    style.alignItems = mapCounterAlignToAlignItems(config.counterAxisAlignItems);
  }

  // 尺寸 - Figma 的 primaryAxis 对应主轴方向
  // HORIZONTAL: primaryAxis = width, counterAxis = height
  // VERTICAL: primaryAxis = height, counterAxis = width
  const isVertical = config.layoutMode === 'VERTICAL';
  
  if (isVertical) {
    // VERTICAL 布局：主轴是 height，交叉轴是 width
    const height = mapSizingModeToCSS(config.primaryAxisSizingMode, config.height);
    const width = mapSizingModeToCSS(config.counterAxisSizingMode, config.width);
    if (width !== undefined) style.width = width;
    if (height !== undefined) style.height = height;
  } else {
    // HORIZONTAL 布局：主轴是 width，交叉轴是 height
    const width = mapSizingModeToCSS(config.primaryAxisSizingMode, config.width);
    const height = mapSizingModeToCSS(config.counterAxisSizingMode, config.height);
    if (width !== undefined) style.width = width;
    if (height !== undefined) style.height = height;
  }
  
  // 固定尺寸（直接指定的 width/height 优先）
  if (config.width !== undefined && config.primaryAxisSizingMode === undefined && config.counterAxisSizingMode === undefined) {
    style.width = config.width;
  }
  if (config.height !== undefined && config.primaryAxisSizingMode === undefined && config.counterAxisSizingMode === undefined) {
    style.height = config.height;
  }

  // 尺寸约束
  if (config.minWidth !== undefined) style.minWidth = config.minWidth;
  if (config.maxWidth !== undefined) style.maxWidth = config.maxWidth;
  if (config.minHeight !== undefined) style.minHeight = config.minHeight;
  if (config.maxHeight !== undefined) style.maxHeight = config.maxHeight;

  // flex grow
  // Figma 官方文档：layoutGrow: 1 时，对应轴的 sizingMode 应该是 FIXED
  // 在 CSS 中，需要使用 flex: 1 1 0 让元素从 0 开始计算，然后按比例分配空间
  // 只设置 flexGrow 不够，因为 flexBasis 默认是 auto（内容尺寸）
  if (config.layoutGrow !== undefined && config.layoutGrow > 0) {
    style.flexGrow = config.layoutGrow;
    style.flexShrink = 1;
    style.flexBasis = 0;
  }

  // 圆角
  const borderRadius = getBorderRadius(config);
  if (borderRadius !== undefined) {
    style.borderRadius = borderRadius;
  }

  // 填充（背景色）
  const fill = figmaFillsToCSS(config.fills);
  if (fill) {
    style.backgroundColor = fill;
  }

  // 描边
  const stroke = figmaFillsToCSS(config.strokes);
  if (stroke && config.strokeWeight) {
    style.borderColor = stroke;
    style.borderWidth = config.strokeWeight;
    style.borderStyle = 'solid';
  }

  // 字体
  if (config.fontSize !== undefined) style.fontSize = config.fontSize;
  if (config.fontWeight !== undefined) style.fontWeight = config.fontWeight;
  if (config.lineHeight !== undefined) style.lineHeight = config.lineHeight;

  // 文本溢出（需要配合 overflow: hidden）
  // minWidth: 0 防止 flex 子元素被内容撑开
  if (config.textOverflow === 'ellipsis') {
    style.overflow = 'hidden';
    style.textOverflow = 'ellipsis';
    style.whiteSpace = 'nowrap';
    style.minWidth = 0;
  }

  return style;
}
