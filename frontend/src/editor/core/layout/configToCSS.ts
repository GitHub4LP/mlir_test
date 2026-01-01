/**
 * 配置到 CSS 的映射工具
 * 将 Figma Auto Layout 配置转换为 CSS flexbox 属性
 */

import type { CSSProperties } from 'react';
import type { ContainerConfig } from './types';
import { getContainerConfig } from './LayoutConfig';

/**
 * 将 Figma counterAxisAlignItems 映射为 CSS alignItems
 */
function mapAlignItems(align?: 'MIN' | 'CENTER' | 'MAX' | 'BASELINE'): CSSProperties['alignItems'] {
  switch (align) {
    case 'MIN': return 'flex-start';
    case 'CENTER': return 'center';
    case 'MAX': return 'flex-end';
    case 'BASELINE': return 'baseline';
    default: return 'stretch';
  }
}

/**
 * 将 Figma primaryAxisAlignItems 映射为 CSS justifyContent
 */
function mapJustifyContent(align?: 'MIN' | 'CENTER' | 'MAX' | 'SPACE_BETWEEN'): CSSProperties['justifyContent'] {
  switch (align) {
    case 'MIN': return 'flex-start';
    case 'CENTER': return 'center';
    case 'MAX': return 'flex-end';
    case 'SPACE_BETWEEN': return 'space-between';
    default: return 'flex-start';
  }
}

/**
 * 将 Figma fills 转换为 CSS backgroundColor
 */
function mapFillToBackground(config: ContainerConfig): string | undefined {
  if (config.fill) return config.fill;
  
  if (config.fills && config.fills.length > 0) {
    const fill = config.fills[0];
    if (fill.type === 'SOLID' && fill.visible !== false && fill.color) {
      const { r, g, b } = fill.color;
      const opacity = fill.opacity ?? 1;
      if (opacity === 1) {
        return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
      }
      return `rgba(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)}, ${opacity})`;
    }
  }
  
  return undefined;
}

/**
 * 将 Figma cornerRadius 转换为 CSS borderRadius
 */
function mapCornerRadius(config: ContainerConfig): string | number | undefined {
  if (config.topLeftRadius !== undefined || config.topRightRadius !== undefined ||
      config.bottomLeftRadius !== undefined || config.bottomRightRadius !== undefined) {
    return `${config.topLeftRadius ?? 0}px ${config.topRightRadius ?? 0}px ${config.bottomRightRadius ?? 0}px ${config.bottomLeftRadius ?? 0}px`;
  }
  return config.cornerRadius;
}

/**
 * 将 ContainerConfig 转换为 CSS flexbox 样式
 * 
 * Figma layoutGrow 的语义：
 * - layoutGrow > 0 表示元素在主轴方向填充剩余空间
 * - 同时在交叉轴方向也填充（stretch）
 * 
 * CSS flexbox 的差异：
 * - flex-grow 只影响主轴
 * - 交叉轴填充需要 align-self: stretch
 * 
 * 关键问题：
 * - 当父容器是 VERTICAL 布局时，子元素的 layoutGrow 应该让宽度填充
 * - 但 CSS 中 flex-grow 只影响高度（主轴），宽度需要用 width: 100%
 * - 解决方案：layoutGrow > 0 时同时设置 align-self: stretch
 * 
 * Canvas 布局引擎与 CSS flexbox 的差异：
 * - Canvas：空容器的 measuredHeight = paddingTop + paddingBottom
 * - CSS：空 flex 容器高度为 0，padding 不会撑开高度
 * - 解决方案：自动计算 minHeight = paddingTop + paddingBottom
 * 
 * @param config - 容器配置
 * @param parentDirection - 父容器的布局方向（可选，用于正确处理 layoutGrow）
 */
export function configToFlexboxStyle(
  config: ContainerConfig,
  parentDirection?: 'HORIZONTAL' | 'VERTICAL'
): CSSProperties {
  const style: CSSProperties = {
    // 确保所有元素使用 border-box，与 Canvas 布局一致
    boxSizing: 'border-box',
  };
  
  // 布局方向
  if (config.layoutMode === 'HORIZONTAL') {
    style.display = 'flex';
    style.flexDirection = 'row';
  } else if (config.layoutMode === 'VERTICAL') {
    style.display = 'flex';
    style.flexDirection = 'column';
  }
  
  // 间距
  if (config.itemSpacing !== undefined) {
    style.gap = config.itemSpacing;
  }
  
  // Padding
  const paddingTop = config.paddingTop ?? 0;
  const paddingRight = config.paddingRight ?? 0;
  const paddingBottom = config.paddingBottom ?? 0;
  const paddingLeft = config.paddingLeft ?? 0;
  if (paddingTop || paddingRight || paddingBottom || paddingLeft) {
    style.padding = `${paddingTop}px ${paddingRight}px ${paddingBottom}px ${paddingLeft}px`;
  }
  
  // 对齐
  style.alignItems = mapAlignItems(config.counterAxisAlignItems);
  style.justifyContent = mapJustifyContent(config.primaryAxisAlignItems);
  
  // layoutGrow 处理
  // 
  // Figma/Canvas 布局引擎中 layoutGrow 的语义：
  // - 交叉轴方向：填充父容器（始终生效）
  // - 主轴方向：只有当父容器有剩余空间时才填充
  // 
  // CSS flexbox 的映射策略：
  // - 使用 align-self: stretch 实现交叉轴填充
  // - 只在 HORIZONTAL 父容器中设置 flex-grow（宽度填充）
  // - 在 VERTICAL 父容器中不设置 flex-grow（避免高度填充）
  // - 设置 minWidth: 0 允许收缩（用于文本省略等场景）
  if (config.layoutGrow !== undefined && config.layoutGrow > 0) {
    // 交叉轴方向：使用 align-self: stretch
    style.alignSelf = 'stretch';
    
    // 允许收缩（用于文本省略等场景）
    // 使用字符串格式确保 Vue h() 正确处理
    if (config.minWidth === undefined) {
      style.minWidth = '0px';
    }
    
    // 主轴方向：只在 HORIZONTAL 父容器中设置 flex-grow
    // 在 VERTICAL 父容器中，flex-grow 会导致高度填充，这不是我们想要的
    if (parentDirection === 'HORIZONTAL') {
      style.flexGrow = config.layoutGrow;
      style.flexShrink = 1;
      style.flexBasis = 0;
    }
  }
  
  // 尺寸
  if (typeof config.width === 'number') {
    style.width = config.width;
  } else if (config.width === 'fill-parent') {
    style.flex = 1;
    style.alignSelf = 'stretch';
  } else if (typeof config.width === 'string') {
    // 支持百分比等字符串值，如 '100%'
    style.width = config.width;
  }
  
  if (typeof config.height === 'number') {
    style.height = config.height;
  }
  
  // 最小/最大尺寸
  // 
  // Canvas 布局引擎中，空容器的 measuredHeight = paddingTop + paddingBottom
  // CSS flexbox 中，空容器高度为 0，padding 不会撑开高度
  // 解决方案：如果没有显式 minHeight，自动设置为 paddingTop + paddingBottom
  // 同理处理 minWidth
  // 
  // 注意：layoutGrow 处理中可能已经设置了 minWidth: 0（用于文本省略），
  // 这种情况下不应该被 implicitMinWidth 覆盖
  // 
  // 重要：使用字符串格式（如 '4px'）而非数字，确保 Vue h() 函数正确处理
  // React JSX 会自动为数字添加 px，但 Vue h() 不会
  const implicitMinHeight = paddingTop + paddingBottom;
  const implicitMinWidth = paddingLeft + paddingRight;
  
  if (config.minWidth !== undefined) {
    style.minWidth = typeof config.minWidth === 'number' ? `${config.minWidth}px` : config.minWidth;
  } else if (style.minWidth === undefined && implicitMinWidth > 0) {
    // 只有当 style.minWidth 未被设置时，才应用 implicitMinWidth
    style.minWidth = `${implicitMinWidth}px`;
  }
  
  if (config.maxWidth !== undefined) {
    style.maxWidth = typeof config.maxWidth === 'number' ? `${config.maxWidth}px` : config.maxWidth;
  }
  
  if (config.minHeight !== undefined) {
    style.minHeight = typeof config.minHeight === 'number' ? `${config.minHeight}px` : config.minHeight;
  } else if (style.minHeight === undefined && implicitMinHeight > 0) {
    // 只有当 style.minHeight 未被设置时，才应用 implicitMinHeight
    style.minHeight = `${implicitMinHeight}px`;
  }
  
  if (config.maxHeight !== undefined) {
    style.maxHeight = typeof config.maxHeight === 'number' ? `${config.maxHeight}px` : config.maxHeight;
  }
  
  // 背景
  const backgroundColor = mapFillToBackground(config);
  if (backgroundColor) style.backgroundColor = backgroundColor;
  
  // 圆角
  const borderRadius = mapCornerRadius(config);
  if (borderRadius !== undefined) style.borderRadius = borderRadius;
  
  // 边框
  if (config.stroke) {
    style.borderColor = config.stroke;
    style.borderStyle = 'solid';
    style.borderWidth = config.strokeWidth ?? 1;
  }
  
  // 溢出处理
  if (config.overflow === 'hidden') {
    style.overflow = 'hidden';
  }
  
  // 定位处理
  if (config.position === 'absolute') {
    style.position = 'absolute';
    style.left = 0;
    style.right = 0;
    style.bottom = 0;
  } else if (config.position === 'relative') {
    style.position = 'relative';
  }
  
  // Overlay 模式处理
  // overlay 元素使用 absolute 定位，不参与父容器宽度计算
  // 定位在父容器底部，宽度填充父容器
  if (config.overlay) {
    style.position = 'absolute';
    style.left = 0;
    style.right = 0;
    style.bottom = 0;
  }
  
  // 文本溢出处理
  if (config.textOverflow === 'ellipsis') {
    style.overflow = 'hidden';
    style.textOverflow = 'ellipsis';
    style.whiteSpace = 'nowrap';
    // 需要有宽度限制才能生效，使用 minWidth: 0 让 flex 子元素可以收缩
    style.minWidth = 0;
  }
  
  return style;
}

/**
 * 根据节点类型获取 flexbox 样式
 */
export function getFlexboxStyleForType(type: string): CSSProperties {
  const config = getContainerConfig(type);
  return configToFlexboxStyle(config);
}
