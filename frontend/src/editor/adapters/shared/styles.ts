/**
 * 共享样式工具
 * 
 * 基于 Design Tokens 的样式工具函数
 * 供所有渲染器使用（ReactFlow、VueFlow、Canvas、GPU）
 */

import { tokens } from '../../../generated/tokens';

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
// Handle 样式
// ============================================================

/** 获取执行引脚样式（左侧输入，三角形朝右） */
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

/** 获取执行引脚样式（右侧输出，三角形朝右） */
export function getExecHandleStyleRight(): React.CSSProperties {
  return {
    width: 0,
    height: 0,
    minWidth: 0,
    minHeight: 0,
    padding: 0,
    borderStyle: 'solid',
    borderWidth: '5px 8px 5px 0',
    borderColor: `transparent ${tokens.edge.exec.color} transparent transparent`,
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
// 颜色获取
// ============================================================

/** 获取方言颜色 */
export function getDialectColor(dialect: string): string {
  const dialectColors = tokens.dialect as Record<string, string>;
  return dialectColors[dialect] ?? tokens.dialect.default;
}

/** 获取节点类型颜色 */
export function getNodeTypeColor(type: 'entry' | 'entryMain' | 'return' | 'returnMain' | 'call' | 'operation'): string {
  return tokens.nodeType[type];
}

/**
 * 获取类型颜色
 * 
 * 匹配规则（按优先级）：
 * 1. 精确匹配（如 I1、Index、BF16）
 * 2. 前缀模式匹配（如 UI* 匹配 UI8、UI16 等）
 * 3. 关键词匹配（如包含 Integer、Float）
 * 4. 默认颜色
 */
export function getTypeColor(typeConstraint: string): string {
  if (!typeConstraint) return tokens.type.default;
  
  const typeColors = tokens.type as Record<string, string>;
  
  // 1. 精确匹配
  if (typeColors[typeConstraint]) {
    return typeColors[typeConstraint];
  }
  
  // 2. 前缀模式匹配
  if (/^UI\d+$/.test(typeConstraint)) return tokens.type.unsignedInteger;
  if (/^I\d+$/.test(typeConstraint)) return tokens.type.signlessInteger;
  if (/^SI\d+$/.test(typeConstraint)) return tokens.type.signedInteger;
  if (/^F\d+/.test(typeConstraint)) return tokens.type.float;
  if (/^TF\d+/.test(typeConstraint)) return tokens.type.tensorFloat;
  
  // 3. 关键词匹配
  if (typeConstraint.includes('Integer') || typeConstraint.includes('Signless')) {
    return tokens.type.signlessInteger;
  }
  if (typeConstraint.includes('Signed') && !typeConstraint.includes('Signless')) {
    return tokens.type.signedInteger;
  }
  if (typeConstraint.includes('Unsigned')) {
    return tokens.type.unsignedInteger;
  }
  if (typeConstraint.includes('Float')) {
    return tokens.type.float;
  }
  if (typeConstraint.includes('Bool')) {
    return tokens.type.I1;
  }
  
  // 4. 默认颜色
  return tokens.type.default;
}

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
  headerHeight: tokens.node.header.height,
  pinRowHeight: tokens.node.pin.rowHeight,
  padding: tokens.node.padding,
  handleRadius: tokens.node.handle.radius,
  minWidth: tokens.node.minWidth,
  borderRadius: tokens.node.border.radius,
};

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
// UI 样式常量（供 Canvas UI 组件使用）
// ============================================================

export const UI = {
  // 面板
  panelWidthNarrow: tokens.ui.panelWidthNarrow,
  panelWidthMedium: tokens.ui.panelWidthMedium,
  panelMaxHeight: tokens.ui.panelMaxHeight,
  
  // 按钮
  buttonBg: tokens.ui.buttonBg,
  buttonHoverBg: tokens.ui.buttonHoverBg,
  
  // 阴影
  shadowBlur: tokens.ui.shadowBlur,
  shadowColor: tokens.ui.shadowColor,
  
  // 尺寸
  listItemHeight: tokens.ui.listItemHeight,
  searchHeight: tokens.ui.searchHeight,
  smallButtonHeight: tokens.ui.smallButtonHeight,
  rowHeight: tokens.ui.rowHeight,
  gap: tokens.ui.gap,
  smallGap: tokens.ui.smallGap,
  scrollbarWidth: tokens.ui.scrollbarWidth,
  minScrollbarHeight: tokens.ui.minScrollbarHeight,
  
  // 位置
  closeButtonOffset: tokens.ui.closeButtonOffset,
  closeButtonSize: tokens.ui.closeButtonSize,
  titleLeftPadding: tokens.ui.titleLeftPadding,
  colorDotRadius: tokens.ui.colorDotRadius,
  colorDotGap: tokens.ui.colorDotGap,
  
  // 颜色
  darkBg: tokens.ui.darkBg,
  successColor: tokens.ui.successColor,
  successHoverColor: tokens.ui.successHoverColor,
  
  // 动画
  cursorBlinkInterval: parseInt(tokens.ui.cursorBlinkInterval),
};

// ============================================================
// Overlay 样式常量
// ============================================================

export const OVERLAY = {
  bg: tokens.overlay.bg,
  borderColor: tokens.overlay.borderColor,
  borderWidth: tokens.overlay.borderWidth,
  borderRadius: tokens.overlay.borderRadius,
  boxShadow: tokens.overlay.boxShadow,
  padding: tokens.overlay.padding,
};

// ============================================================
// Button 样式常量
// ============================================================

export const BUTTON = {
  size: tokens.button.size,
  borderRadius: tokens.button.borderRadius,
  bg: tokens.button.bg,
  hoverBg: tokens.button.hoverBg,
  borderColor: tokens.button.borderColor,
  borderWidth: tokens.button.borderWidth,
  textColor: tokens.button.textColor,
  fontSize: tokens.button.fontSize,
  dangerColor: tokens.button.danger.color,
  dangerHoverColor: tokens.button.danger.hoverColor,
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

// ============================================================
// 重导出 tokens（供需要直接访问的场景）
// ============================================================

export { tokens };
