/**
 * Figma 样式获取函数
 * 
 * 从 layoutTokens.json（Figma 格式）生成 CSS 样式
 * 供 ReactFlow、VueFlow 等 DOM 渲染器使用
 */

import type { CSSProperties } from 'react';
import { layoutConfig } from '../../core/layout/LayoutConfig';
import type { ContainerConfig } from '../../core/layout/types';
import { figmaToCSS, figmaColorToCSS, figmaFillsToCSS } from '../../core/layout/figmaToCSS';
import type { FigmaLayoutConfig } from '../../core/layout/figmaTypes';

// ============================================================================
// 样式缓存
// ============================================================================

const styleCache = new Map<string, CSSProperties>();

/**
 * 清除样式缓存（用于热更新）
 */
export function clearStyleCache(): void {
  styleCache.clear();
}

// ============================================================================
// ContainerConfig → FigmaLayoutConfig 适配
// ============================================================================

/**
 * 将 ContainerConfig 转换为 FigmaLayoutConfig
 * 支持新旧两种格式
 */
function toFigmaConfig(config: ContainerConfig): FigmaLayoutConfig {
  const result: FigmaLayoutConfig = {};

  // 优先使用 Figma 原生属性
  if (config.layoutMode) result.layoutMode = config.layoutMode;
  if (config.itemSpacing !== undefined) result.itemSpacing = config.itemSpacing;
  if (config.paddingTop !== undefined) result.paddingTop = config.paddingTop;
  if (config.paddingRight !== undefined) result.paddingRight = config.paddingRight;
  if (config.paddingBottom !== undefined) result.paddingBottom = config.paddingBottom;
  if (config.paddingLeft !== undefined) result.paddingLeft = config.paddingLeft;
  if (config.primaryAxisSizingMode) result.primaryAxisSizingMode = config.primaryAxisSizingMode;
  if (config.counterAxisSizingMode) result.counterAxisSizingMode = config.counterAxisSizingMode;
  if (config.layoutGrow !== undefined) result.layoutGrow = config.layoutGrow;
  if (config.primaryAxisAlignItems) result.primaryAxisAlignItems = config.primaryAxisAlignItems;
  if (config.counterAxisAlignItems) result.counterAxisAlignItems = config.counterAxisAlignItems;

  // 圆角
  if (config.cornerRadius !== undefined) result.cornerRadius = config.cornerRadius;
  if (config.topLeftRadius !== undefined) result.topLeftRadius = config.topLeftRadius;
  if (config.topRightRadius !== undefined) result.topRightRadius = config.topRightRadius;
  if (config.bottomLeftRadius !== undefined) result.bottomLeftRadius = config.bottomLeftRadius;
  if (config.bottomRightRadius !== undefined) result.bottomRightRadius = config.bottomRightRadius;

  // 填充
  if (config.fills) result.fills = config.fills;

  // 尺寸
  if (typeof config.width === 'number') result.width = config.width;
  if (typeof config.height === 'number') result.height = config.height;
  if (config.minWidth !== undefined) result.minWidth = config.minWidth;
  if (config.maxWidth !== undefined) result.maxWidth = config.maxWidth;
  if (config.minHeight !== undefined) result.minHeight = config.minHeight;
  if (config.maxHeight !== undefined) result.maxHeight = config.maxHeight;

  // 描边
  if (config.strokeWeight !== undefined) result.strokeWeight = config.strokeWeight;

  return result;
}


// ============================================================================
// 带缓存的样式获取函数
// ============================================================================

/**
 * 获取容器样式（带缓存）
 */
function getCachedStyle(key: string, config: ContainerConfig): CSSProperties {
  if (styleCache.has(key)) {
    return styleCache.get(key)!;
  }
  const figmaConfig = toFigmaConfig(config);
  const style = figmaToCSS(figmaConfig);
  styleCache.set(key, style);
  return style;
}

// ============================================================================
// 节点容器样式
// ============================================================================

/**
 * 获取节点容器样式
 */
export function getNodeContainerStyle(selected = false): CSSProperties {
  const baseStyle = getCachedStyle('node', layoutConfig.node);
  
  // 仅选中时显示边框（与 Canvas 一致）
  if (selected) {
    return {
      ...baseStyle,
      borderWidth: 2,
      borderColor: '#60a5fa',
      borderStyle: 'solid',
    };
  }
  
  return baseStyle;
}

/**
 * 获取节点头部包装样式
 */
export function getHeaderWrapperStyle(): CSSProperties {
  return getCachedStyle('headerWrapper', layoutConfig.headerWrapper);
}

/**
 * 获取节点头部内容样式
 */
export function getHeaderContentStyle(headerColor?: string): CSSProperties {
  const baseStyle = getCachedStyle('headerContent', layoutConfig.headerContent);
  if (headerColor) {
    return { ...baseStyle, backgroundColor: headerColor };
  }
  return baseStyle;
}

/**
 * 获取标题组样式
 */
export function getTitleGroupStyle(): CSSProperties {
  return getCachedStyle('titleGroup', layoutConfig.titleGroup);
}

/**
 * 获取徽章组样式
 */
export function getBadgesGroupStyle(): CSSProperties {
  return getCachedStyle('badgesGroup', layoutConfig.badgesGroup);
}

// ============================================================================
// 引脚区域样式
// ============================================================================

/**
 * 获取引脚区域样式
 */
export function getPinAreaStyle(): CSSProperties {
  return getCachedStyle('pinArea', layoutConfig.pinArea);
}

/**
 * 获取引脚行样式
 */
export function getPinRowStyle(): CSSProperties {
  return getCachedStyle('pinRow', layoutConfig.pinRow);
}

/**
 * 获取引脚行内容样式
 */
export function getPinRowContentStyle(): CSSProperties {
  return getCachedStyle('pinRowContent', layoutConfig.pinRowContent);
}

/**
 * 获取左侧引脚组样式
 */
export function getLeftPinGroupStyle(): CSSProperties {
  return getCachedStyle('leftPinGroup', layoutConfig.leftPinGroup);
}

/**
 * 获取右侧引脚组样式
 */
export function getRightPinGroupStyle(): CSSProperties {
  return getCachedStyle('rightPinGroup', layoutConfig.rightPinGroup);
}

/**
 * 获取引脚内容样式（左对齐）
 */
export function getPinContentStyle(): CSSProperties {
  return getCachedStyle('pinContent', layoutConfig.pinContent);
}

/**
 * 获取引脚内容样式（右对齐）
 */
export function getPinContentRightStyle(): CSSProperties {
  return getCachedStyle('pinContentRight', layoutConfig.pinContentRight);
}


// ============================================================================
// 属性区域样式
// ============================================================================

/**
 * 获取属性区域样式
 */
export function getAttrAreaStyle(): CSSProperties {
  return getCachedStyle('attrArea', layoutConfig.attrArea);
}

/**
 * 获取属性内容样式
 */
export function getAttrContentStyle(): CSSProperties {
  return getCachedStyle('attrContent', layoutConfig.attrContent);
}

/**
 * 获取标签列样式
 */
export function getLabelColumnStyle(): CSSProperties {
  return getCachedStyle('labelColumn', layoutConfig.labelColumn);
}

/**
 * 获取值列样式
 */
export function getValueColumnStyle(): CSSProperties {
  return getCachedStyle('valueColumn', layoutConfig.valueColumn);
}

// ============================================================================
// 类型标签样式
// ============================================================================

/**
 * 获取类型标签样式
 */
export function getTypeLabelStyle(bgColor?: string): CSSProperties {
  const baseStyle = getCachedStyle('typeLabel', layoutConfig.typeLabel);
  if (bgColor) {
    return { ...baseStyle, backgroundColor: bgColor };
  }
  return baseStyle;
}

// ============================================================================
// 摘要区域样式
// ============================================================================

/**
 * 获取摘要区域样式
 */
export function getSummaryStyle(): CSSProperties {
  return getCachedStyle('summary', layoutConfig.summary);
}

/**
 * 获取摘要内容样式
 */
export function getSummaryContentStyle(): CSSProperties {
  return getCachedStyle('summaryContent', layoutConfig.summaryContent);
}

/**
 * 获取摘要文本样式
 */
export function getSummaryTextStyle(): CSSProperties {
  return getCachedStyle('summaryText', layoutConfig.summaryText);
}

// ============================================================================
// Handle 样式
// ============================================================================

/**
 * 获取 Handle 尺寸
 */
export function getHandleSize(): number {
  return typeof layoutConfig.handle.width === 'number' ? layoutConfig.handle.width : 12;
}

/**
 * 获取 Handle 描边宽度
 */
export function getHandleStrokeWeight(): number {
  return layoutConfig.handle.strokeWeight ?? 2;
}

// 默认值
const DEFAULTS = {
  nodeBg: '#1e1e2e',
  execColor: '#ffffff',
} as const;

/**
 * 获取执行引脚样式（左侧输入，三角形朝右 ▶）
 */
export function getExecHandleStyle(): CSSProperties {
  const execColor = layoutConfig.edge.exec.stroke ?? DEFAULTS.execColor;
  return {
    width: 0,
    height: 0,
    minWidth: 0,
    minHeight: 0,
    padding: 0,
    borderStyle: 'solid',
    borderWidth: '5px 0 5px 8px',
    borderColor: `transparent transparent transparent ${execColor}`,
    backgroundColor: 'transparent',
    borderRadius: 0,
  };
}

/**
 * 获取执行引脚样式（右侧输出，三角形朝右 ▶）
 */
export function getExecHandleStyleRight(): CSSProperties {
  const execColor = layoutConfig.edge.exec.stroke ?? DEFAULTS.execColor;
  return {
    width: 0,
    height: 0,
    minWidth: 0,
    minHeight: 0,
    padding: 0,
    borderStyle: 'solid',
    // 三角形朝右：左边框有颜色
    borderWidth: '5px 0 5px 8px',
    borderColor: `transparent transparent transparent ${execColor}`,
    backgroundColor: 'transparent',
    borderRadius: 0,
  };
}

/**
 * 获取数据引脚样式
 */
export function getDataHandleStyle(color: string): CSSProperties {
  const size = getHandleSize();
  const nodeBg = DEFAULTS.nodeBg;
  return {
    width: size,
    height: size,
    backgroundColor: color,
    border: `2px solid ${nodeBg}`,
    borderRadius: '50%',
  };
}

// ============================================================================
// Spacer 样式
// ============================================================================

/**
 * 获取头部左侧间隔样式
 */
export function getHeaderLeftSpacerStyle(): CSSProperties {
  return getCachedStyle('headerLeftSpacer', layoutConfig.headerLeftSpacer);
}

/**
 * 获取头部右侧间隔样式
 */
export function getHeaderRightSpacerStyle(): CSSProperties {
  return getCachedStyle('headerRightSpacer', layoutConfig.headerRightSpacer);
}

/**
 * 获取头部间隔样式
 */
export function getHeaderSpacerStyle(): CSSProperties {
  return getCachedStyle('headerSpacer', layoutConfig.headerSpacer);
}

/**
 * 获取引脚行间隔样式
 */
export function getPinRowSpacerStyle(): CSSProperties {
  return getCachedStyle('pinRowSpacer', layoutConfig.pinRowSpacer);
}

// ============================================================================
// 颜色获取函数（从 LayoutConfig 重导出）
// ============================================================================

// 从 LayoutConfig 导入统一实现
import {
  getDialectColor as _getDialectColor,
  getNodeTypeColor as _getNodeTypeColor,
  getTypeColor as _getTypeColor,
} from '../../core/layout/LayoutConfig';

/** 获取方言颜色 */
export const getDialectColor = _getDialectColor;

/** 获取节点类型颜色 */
export const getNodeTypeColor = _getNodeTypeColor;

/** 获取类型颜色 */
export const getTypeColor = _getTypeColor;

// ============================================================================
// 常量导出
// ============================================================================

/** 节点最小宽度 */
export const NODE_MIN_WIDTH = layoutConfig.node.minWidth ?? 0;

// ============================================================================
// 导出 layoutConfig 供直接访问
// ============================================================================

export { layoutConfig };

// 导出颜色转换函数
export { figmaColorToCSS, figmaFillsToCSS };
