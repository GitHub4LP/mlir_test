/**
 * 共享样式工具
 * 
 * 基于 Design Tokens 的样式工具函数
 * 供所有渲染器使用（ReactFlow、VueFlow、Canvas、GPU）
 * 
 * 这是样式的唯一权威来源，其他文件应从这里导入
 * 
 * 数据源：layoutTokens.json（唯一数据源）
 */

import { layoutConfig, normalizePadding } from '../../core/layout/LayoutConfig';

// ============================================================
// 直接导出 layoutConfig（激进策略）
// ============================================================

export { layoutConfig };

// 常用快捷访问
export const colors = layoutConfig.colors;
export const dialect = layoutConfig.dialect;
export const type = layoutConfig.type;
export const nodeType = layoutConfig.nodeType;

// ============================================================
// 派生的 tokens 对象（保持部分接口兼容）
// ============================================================

// ============================================================
// 默认值常量（用于处理可选属性）
// ============================================================

const DEFAULTS = {
  nodeBg: '#1e1e2e',
  nodeMinWidth: 200,
  borderRadius: 8,
  handleSize: 12,
  execColor: '#ffffff',
  dataDefaultColor: '#888888',
} as const;

// ============================================================
// 辅助函数：从 Figma 格式获取内边距
// ============================================================

function getFigmaPadding(config: { paddingTop?: number; paddingRight?: number; paddingBottom?: number; paddingLeft?: number; padding?: number | [number, number, number, number] }) {
  // 优先使用 Figma 格式
  if (config.paddingTop !== undefined || config.paddingRight !== undefined || config.paddingBottom !== undefined || config.paddingLeft !== undefined) {
    return {
      top: config.paddingTop ?? 0,
      right: config.paddingRight ?? 0,
      bottom: config.paddingBottom ?? 0,
      left: config.paddingLeft ?? 0,
    };
  }
  // 回退到旧格式
  return normalizePadding(config.padding);
}

// ============================================================
// 辅助函数：从 Figma fills 获取背景色
// ============================================================

function getFillColor(config: { fills?: readonly Paint[]; fill?: string }): string {
  // 优先使用 Figma fills
  if (config.fills && config.fills.length > 0) {
    const fill = config.fills[0];
    if (fill.type === 'SOLID' && fill.visible !== false) {
      const { color, opacity = 1 } = fill;
      const r = Math.round(color.r * 255);
      const g = Math.round(color.g * 255);
      const b = Math.round(color.b * 255);
      if (opacity === 1) {
        return `rgb(${r}, ${g}, ${b})`;
      }
      return `rgba(${r}, ${g}, ${b}, ${opacity})`;
    }
  }
  // 回退到旧格式
  return config.fill ?? DEFAULTS.nodeBg;
}

// ============================================================
// 辅助函数：获取圆角
// ============================================================

function getCornerRadius(config: { cornerRadius?: number; topLeftRadius?: number; topRightRadius?: number; bottomLeftRadius?: number; bottomRightRadius?: number }): number {
  // 优先使用统一圆角
  if (config.cornerRadius !== undefined) return config.cornerRadius;
  // 使用左上角作为默认值
  if (config.topLeftRadius !== undefined) return config.topLeftRadius;
  return DEFAULTS.borderRadius;
}

/**
 * tokens 对象 - 从 layoutConfig 派生
 * 注意：推荐直接使用 layoutConfig，此对象仅为过渡期兼容
 * 
 * 所有属性都有确定的类型（非 undefined），通过默认值保证
 */
export const tokens = {
  // 布局相关 - 从 layoutConfig 派生
  node: {
    bg: getFillColor(layoutConfig.pinRowContent),
    minWidth: layoutConfig.node.minWidth ?? DEFAULTS.nodeMinWidth,
    padding: getFigmaPadding(layoutConfig.node).top,
    border: {
      color: (layoutConfig.node as unknown as { border?: { color?: string } }).border?.color ?? '#3d3d4d',
      width: (layoutConfig.node as unknown as { border?: { width?: number } }).border?.width ?? 1,
      radius: getCornerRadius(layoutConfig.headerContent),
    },
    selected: {
      borderColor: (layoutConfig.node as unknown as { selected?: { borderColor?: string } }).selected?.borderColor ?? '#60a5fa',
      borderWidth: (layoutConfig.node as unknown as { selected?: { borderWidth?: number } }).selected?.borderWidth ?? 2,
    },
    header: {
      height: (() => {
        const padding = getFigmaPadding(layoutConfig.headerContent);
        return padding.top + padding.bottom + layoutConfig.text.title.fontSize;
      })(),
      paddingX: getFigmaPadding(layoutConfig.headerContent).right,
      titleGap: layoutConfig.titleGroup.itemSpacing ?? layoutConfig.titleGroup.spacing,
    },
    pin: {
      rowHeight: layoutConfig.pinRow.minHeight ?? 28,
      rowPadding: getFigmaPadding(layoutConfig.leftPinGroup).top,
      contentMargin: (() => {
        const spacing = layoutConfig.leftPinGroup.itemSpacing ?? layoutConfig.leftPinGroup.spacing;
        const handleWidth = layoutConfig.handle.width;
        return typeof spacing === 'number' ? spacing + (typeof handleWidth === 'number' ? handleWidth : DEFAULTS.handleSize) : 16;
      })(),
      labelTypeGap: layoutConfig.pinContent.itemSpacing ?? layoutConfig.pinContent.spacing,
      centerGap: 16,
    },
    handle: {
      size: typeof layoutConfig.handle.width === 'number' ? layoutConfig.handle.width : DEFAULTS.handleSize,
      radius: (typeof layoutConfig.handle.width === 'number' ? layoutConfig.handle.width : DEFAULTS.handleSize) / 2,
      offset: 0,
    },
    attr: {
      labelWidth: 60,
      rowHeight: 20,
      rowGap: 4,
      valueGap: 8,
    },
  },
  text: {
    fontFamily: layoutConfig.text.fontFamily,
    title: {
      size: layoutConfig.text.title.fontSize,
      color: layoutConfig.text.title.fill,
      weight: layoutConfig.text.title.fontWeight,
    },
    subtitle: {
      size: layoutConfig.text.subtitle.fontSize,
      color: layoutConfig.text.subtitle.fill,
      weight: layoutConfig.text.subtitle.fontWeight,
    },
    label: {
      size: layoutConfig.text.label.fontSize,
      color: layoutConfig.text.label.fill,
    },
    muted: {
      color: layoutConfig.text.muted.fill,
    },
  },
  edge: {
    width: layoutConfig.edge.data.strokeWidth,
    selectedWidth: 3,
    bezierOffset: layoutConfig.edge.bezierOffset,
    exec: {
      color: layoutConfig.edge.exec.stroke ?? DEFAULTS.execColor,
    },
    data: {
      defaultColor: layoutConfig.edge.data.defaultStroke ?? DEFAULTS.dataDefaultColor,
    },
  },
  typeLabel: {
    width: 60,
    height: 16,
    minWidth: layoutConfig.typeLabel.minWidth ?? 40,
    paddingX: getFigmaPadding(layoutConfig.typeLabel).right,
    borderRadius: getCornerRadius(layoutConfig.typeLabel),
    fontSize: layoutConfig.text.label.fontSize,
    bgAlpha: '0.3',
    textColor: '#ffffff',
    offsetFromHandle: 16,
    bg: getFillColor(layoutConfig.typeLabel),
  },
  nodeType: layoutConfig.nodeType,
  
  // 非布局属性 - 直接从 layoutConfig 继承
  color: layoutConfig.colors,
  colors: layoutConfig.colors,
  size: layoutConfig.size,
  radius: layoutConfig.radius,
  border: layoutConfig.border,
  font: layoutConfig.font,
  button: layoutConfig.buttonStyle,
  dialect: layoutConfig.dialect,
  overlay: layoutConfig.overlay,
  canvas: layoutConfig.canvas,
  minimap: layoutConfig.minimap,
  type: layoutConfig.type,
  ui: layoutConfig.ui,
} as const;


// ============================================================
// 节点样式
// ============================================================

/** 获取节点容器样式 */
export function getNodeContainerStyle(selected: boolean): React.CSSProperties {
  return {
    backgroundColor: tokens.node.bg,
    borderWidth: selected ? tokens.node.selected.borderWidth : tokens.node.border.width,
    borderColor: selected ? tokens.node.selected.borderColor : tokens.node.border.color,
    borderStyle: 'solid',
    borderRadius: tokens.node.border.radius,
    minWidth: tokens.node.minWidth,
    fontFamily: tokens.text.fontFamily,
  };
}

/** 获取节点头部样式 */
export function getNodeHeaderStyle(headerColor: string): React.CSSProperties {
  return {
    backgroundColor: headerColor,
    borderRadius: `${tokens.node.border.radius}px ${tokens.node.border.radius}px 0 0`,
    padding: `${tokens.node.padding}px 12px`,
    height: tokens.node.header.height,
    boxSizing: 'border-box',
  };
}

/** 获取节点 body 样式 */
export function getNodeBodyStyle(): React.CSSProperties {
  return {
    padding: tokens.node.padding,
  };
}

/** 获取引脚行样式 */
export function getPinRowStyle(): React.CSSProperties {
  return {
    minHeight: tokens.node.pin.rowHeight,
    display: 'flex',
    alignItems: 'center',
  };
}

// ============================================================
// Handle 样式（React 组件用）
// ============================================================

/** 获取执行引脚样式（左侧输入，三角形朝右 ▶） */
export function getExecHandleStyle(): React.CSSProperties {
  return {
    width: 0,
    height: 0,
    minWidth: 0,
    minHeight: 0,
    padding: 0,
    borderStyle: 'solid',
    borderWidth: '5px 0 5px 8px',
    borderColor: `transparent transparent transparent ${tokens.edge.exec.color}`,
    backgroundColor: 'transparent',
    borderRadius: 0,
  };
}

/** 获取执行引脚样式（右侧输出，三角形朝右 ▶） */
export function getExecHandleStyleRight(): React.CSSProperties {
  return {
    width: 0,
    height: 0,
    minWidth: 0,
    minHeight: 0,
    padding: 0,
    borderStyle: 'solid',
    // 三角形朝右：左边框有颜色
    borderWidth: '5px 0 5px 8px',
    borderColor: `transparent transparent transparent ${tokens.edge.exec.color}`,
    backgroundColor: 'transparent',
    borderRadius: 0,
  };
}

/** 获取数据引脚样式 */
export function getDataHandleStyle(color: string): React.CSSProperties {
  return {
    width: tokens.node.handle.size,
    height: tokens.node.handle.size,
    backgroundColor: color,
    border: `2px solid ${tokens.node.bg}`,
    borderRadius: '50%',
  };
}

// ============================================================
// Handle CSS 字符串（Vue 组件 scoped style 用）
// ============================================================

/** 获取执行引脚的 CSS 样式字符串（左侧输入） */
export function getExecHandleCSSLeft(): string {
  return `
    width: 0 !important;
    height: 0 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
    border: none !important;
    border-style: solid !important;
    border-width: 5px 0 5px 8px !important;
    border-color: transparent transparent transparent ${tokens.edge.exec.color} !important;
    border-radius: 0 !important;
  `;
}

/** 获取执行引脚的 CSS 样式字符串（右侧输出） */
export function getExecHandleCSSRight(): string {
  return `
    width: 0 !important;
    height: 0 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
    border: none !important;
    border-style: solid !important;
    border-width: 5px 8px 5px 0 !important;
    border-color: transparent ${tokens.edge.exec.color} transparent transparent !important;
    border-radius: 0 !important;
  `;
}

/** 获取数据引脚的 CSS 样式字符串 */
export function getDataHandleCSS(): string {
  const size = tokens.node.handle.size;
  return `
    width: ${size}px !important;
    height: ${size}px !important;
    border: 2px solid ${tokens.node.bg} !important;
    border-radius: 50% !important;
  `;
}

// ============================================================
// 颜色获取（从 LayoutConfig 重导出）
// ============================================================

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

/** 获取类型颜色（简化版，仅模式匹配，不支持约束展开） */
export const getTypeColor = _getTypeColor;


// ============================================================
// 常量导出
// ============================================================

/** 执行引脚颜色 */
export const EXEC_COLOR = tokens.edge.exec.color;

/** Handle 半径 */
export const HANDLE_RADIUS = tokens.node.handle.radius;

/** Handle 直径 */
export const HANDLE_SIZE = tokens.node.handle.size;

/** 节点最小宽度 */
export const NODE_MIN_WIDTH = tokens.node.minWidth;

/** 节点头部高度 */
export const NODE_HEADER_HEIGHT = tokens.node.header.height;

/** 引脚行高 */
export const PIN_ROW_HEIGHT = tokens.node.pin.rowHeight;

/** 节点内边距 */
export const NODE_PADDING = tokens.node.padding;

/** 节点圆角 */
export const NODE_BORDER_RADIUS = tokens.node.border.radius;

// ============================================================
// 布局常量（供 Canvas/GPU 渲染器使用）
// ============================================================

export const LAYOUT = {
  // 节点尺寸 - 从 layoutConfig 派生（带默认值保证类型确定）
  headerHeight: tokens.node.header.height,
  pinRowHeight: tokens.node.pin.rowHeight,
  pinRowPadding: tokens.node.pin.rowPadding,
  // 实际行高 = pinRowHeight + pinRowPadding * 2
  get actualPinRowHeight() { return this.pinRowHeight + this.pinRowPadding * 2; },
  padding: tokens.node.padding,
  handleRadius: tokens.node.handle.radius,
  minWidth: tokens.node.minWidth,
  borderRadius: tokens.node.border.radius,
  
  // 头部布局 - 使用 headerContent 配置
  headerPaddingX: tokens.node.header.paddingX,
  headerPaddingY: getFigmaPadding(layoutConfig.headerContent).top,
  
  // 引脚布局
  pinContentMargin: tokens.node.pin.contentMargin,
  
  // 引脚内容垂直布局
  pinLabelFontSize: tokens.text.label.size,
  pinTypeSelectorHeight: 20,
  pinContentGap: tokens.node.pin.labelTypeGap as number,
  
  // 文字间距
  titleSubtitleGap: typeof tokens.node.header.titleGap === 'number' ? tokens.node.header.titleGap : parseInt(String(tokens.node.header.titleGap)) || 4,
} as const;

/**
 * 计算引脚行内 Handle 的 Y 坐标（相对于节点顶部）
 */
export function getPinHandleY(pinIndex: number): number {
  const actualRowHeight = LAYOUT.pinRowHeight + LAYOUT.pinRowPadding * 2;
  return LAYOUT.headerHeight + LAYOUT.padding + pinIndex * actualRowHeight + actualRowHeight / 2;
}

/**
 * 计算引脚内容区域的 Y 范围（相对于节点顶部）
 */
export function getPinContentLayout(pinIndex: number): {
  rowTop: number;
  rowBottom: number;
  handleY: number;
  labelY: number;
  typeSelectorY: number;
} {
  const actualRowHeight = LAYOUT.pinRowHeight + LAYOUT.pinRowPadding * 2;
  const rowTop = LAYOUT.headerHeight + LAYOUT.padding + pinIndex * actualRowHeight;
  const rowBottom = rowTop + actualRowHeight;
  const handleY = rowTop + actualRowHeight / 2;
  
  const contentHeight = LAYOUT.pinLabelFontSize + LAYOUT.pinContentGap + LAYOUT.pinTypeSelectorHeight;
  const contentStartY = handleY - contentHeight / 2;
  
  const labelY = contentStartY + LAYOUT.pinLabelFontSize / 2;
  const typeSelectorY = contentStartY + LAYOUT.pinLabelFontSize + LAYOUT.pinContentGap;
  
  return { rowTop, rowBottom, handleY, labelY, typeSelectorY };
}

// ============================================================
// 文字样式常量
// ============================================================

export const TEXT = {
  fontFamily: tokens.text.fontFamily,
  titleSize: tokens.text.title.size,
  titleColor: tokens.text.title.color,
  titleWeight: tokens.text.title.weight,
  subtitleSize: tokens.text.subtitle.size,
  subtitleColor: tokens.text.subtitle.color,
  subtitleWeight: tokens.text.subtitle.weight,
  labelSize: tokens.text.label.size,
  labelColor: tokens.text.label.color,
  mutedColor: tokens.text.muted.color,
};

// ============================================================
// 文字宽度测量（供 Canvas 渲染器使用）
// ============================================================

let measureCtx: CanvasRenderingContext2D | null = null;

function getMeasureContext(): CanvasRenderingContext2D {
  if (!measureCtx) {
    const canvas = document.createElement('canvas');
    measureCtx = canvas.getContext('2d')!;
  }
  return measureCtx;
}

/**
 * 测量文字宽度
 */
export function measureTextWidth(
  text: string,
  fontSize: number,
  fontFamily: string = TEXT.fontFamily,
  fontWeight: number | string = 'normal'
): number {
  const ctx = getMeasureContext();
  ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;
  return ctx.measureText(text).width;
}

// ============================================================
// UI 样式常量（供 Canvas UI 组件使用）
// ============================================================

export const UI = {
  panelWidthNarrow: layoutConfig.ui.panelWidthNarrow,
  panelWidthMedium: layoutConfig.ui.panelWidthMedium,
  panelMaxHeight: layoutConfig.ui.panelMaxHeight,
  buttonBg: layoutConfig.ui.buttonBg,
  buttonHoverBg: layoutConfig.ui.buttonHoverBg,
  shadowBlur: layoutConfig.ui.shadowBlur,
  shadowColor: layoutConfig.ui.shadowColor,
  listItemHeight: layoutConfig.ui.listItemHeight,
  searchHeight: layoutConfig.ui.searchHeight,
  smallButtonHeight: layoutConfig.ui.smallButtonHeight,
  rowHeight: layoutConfig.ui.rowHeight,
  labelWidth: layoutConfig.ui.labelWidth,
  gap: layoutConfig.ui.gap,
  smallGap: layoutConfig.ui.smallGap,
  scrollbarWidth: layoutConfig.ui.scrollbarWidth,
  minScrollbarHeight: layoutConfig.ui.minScrollbarHeight,
  closeButtonOffset: layoutConfig.ui.closeButtonOffset,
  closeButtonSize: layoutConfig.ui.closeButtonSize,
  titleLeftPadding: layoutConfig.ui.titleLeftPadding,
  colorDotRadius: layoutConfig.ui.colorDotRadius,
  colorDotGap: layoutConfig.ui.colorDotGap,
  darkBg: layoutConfig.ui.darkBg,
  successColor: layoutConfig.ui.successColor,
  successHoverColor: layoutConfig.ui.successHoverColor,
  cursorBlinkInterval: layoutConfig.ui.cursorBlinkInterval,
};

// ============================================================
// Overlay 样式常量
// ============================================================

export const OVERLAY = {
  bg: layoutConfig.overlay.bg,
  borderColor: layoutConfig.overlay.borderColor,
  borderWidth: layoutConfig.overlay.borderWidth,
  borderRadius: layoutConfig.overlay.borderRadius,
  boxShadow: layoutConfig.overlay.boxShadow,
  padding: layoutConfig.overlay.padding,
};

// ============================================================
// Button 样式常量
// ============================================================

export const BUTTON = {
  size: layoutConfig.buttonStyle.size,
  borderRadius: layoutConfig.buttonStyle.borderRadius,
  bg: layoutConfig.buttonStyle.bg,
  hoverBg: layoutConfig.buttonStyle.hoverBg,
  borderColor: layoutConfig.buttonStyle.borderColor,
  borderWidth: layoutConfig.buttonStyle.borderWidth,
  textColor: layoutConfig.buttonStyle.textColor,
  fontSize: layoutConfig.buttonStyle.fontSize,
  dangerColor: layoutConfig.buttonStyle.danger.color,
  dangerHoverColor: layoutConfig.buttonStyle.danger.hoverColor,
};

// ============================================================
// TypeLabel 样式常量
// ============================================================

export const TYPE_LABEL = {
  width: tokens.typeLabel.width,
  height: tokens.typeLabel.height,
  borderRadius: tokens.typeLabel.borderRadius,
  bgAlpha: tokens.typeLabel.bgAlpha,
  textColor: tokens.typeLabel.textColor,
  fontSize: tokens.typeLabel.fontSize,
  offsetFromHandle: tokens.typeLabel.offsetFromHandle,
};
