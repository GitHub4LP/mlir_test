/**
 * Vue Flow 节点共享样式
 * 
 * 从 Design Tokens 和 HandleStyles 获取样式常量，确保与 Canvas/GPU/React Flow 一致
 * 
 * 设计原则：
 * - 所有数值来自 Design Tokens，不硬编码
 * - Handle 样式从 HandleStyles 获取，与 React Flow 保持一致
 */

import { tokens, LAYOUT as SHARED_LAYOUT, TEXT as SHARED_TEXT, getDialectColor as sharedGetDialectColor } from '../../shared/styles';
import {
  getExecHandleStyle,
  getExecHandleStyleRight,
  getDataHandleStyle,
  getExecHandleCSSLeft,
  getExecHandleCSSRight,
  getDataHandleCSS,
  EXEC_COLOR,
  HANDLE_RADIUS,
  HANDLE_SIZE,
} from '../../shared/HandleStyles';

// ============================================================
// 布局常量 - 从 tokens 获取
// ============================================================

export const LAYOUT = {
  headerHeight: SHARED_LAYOUT.headerHeight,      // 32px
  pinRowHeight: SHARED_LAYOUT.pinRowHeight,      // 28px
  padding: SHARED_LAYOUT.padding,                // 4px
  handleRadius: SHARED_LAYOUT.handleRadius,      // 6px
  minWidth: SHARED_LAYOUT.minWidth,              // 200px
  borderRadius: SHARED_LAYOUT.borderRadius,      // 8px
};

// ============================================================
// 节点容器样式
// ============================================================

/** 节点容器样式 - 与 React Flow getNodeContainerStyle 一致 */
export function getContainerStyle(selected: boolean) {
  return {
    backgroundColor: tokens.node.bg,
    border: `${selected ? tokens.node.selected.borderWidth : tokens.node.border.width}px solid ${selected ? tokens.node.selected.borderColor : tokens.node.border.color}`,
    borderRadius: `${tokens.node.border.radius}px`,
    minWidth: `${tokens.node.minWidth}px`,
    fontFamily: tokens.text.fontFamily,
  };
}

/** 节点头部样式 - 与 React Flow getNodeHeaderStyle 一致 */
export function getHeaderStyle(headerColor: string) {
  return {
    backgroundColor: headerColor,
    borderRadius: `${tokens.node.border.radius}px ${tokens.node.border.radius}px 0 0`,
    padding: `${tokens.node.padding}px 12px`,
    height: `${tokens.node.header.height}px`,
    boxSizing: 'border-box' as const,
  };
}

/** 节点 body 样式 - 与 React Flow NodePins 的 px-1 py-1 一致 */
export function getBodyStyle() {
  return {
    padding: `${tokens.node.padding}px`,
  };
}

/** 引脚行样式 - 与 React Flow pinRowHeight 一致 */
export function getPinRowStyle() {
  return {
    height: `${tokens.node.pin.rowHeight}px`,
    display: 'flex',
    alignItems: 'center',
  };
}

// ============================================================
// Handle 位置计算
// ============================================================

/** 计算 Handle 的 top 位置 */
export function getHandleTop(idx: number): string {
  const y = tokens.node.header.height + tokens.node.padding + idx * tokens.node.pin.rowHeight + tokens.node.pin.rowHeight / 2;
  return `${y}px`;
}

// ============================================================
// 颜色获取
// ============================================================

/** 获取方言颜色 */
export function getDialectColor(dialect: string): string {
  return sharedGetDialectColor(dialect);
}

/** 获取执行引脚颜色 */
export function getExecColor(): string {
  return tokens.edge.exec.color;
}

// ============================================================
// Handle 样式 - 从 HandleStyles 重导出
// ============================================================

export {
  getExecHandleStyle,
  getExecHandleStyleRight,
  getDataHandleStyle,
  getExecHandleCSSLeft,
  getExecHandleCSSRight,
  getDataHandleCSS,
  EXEC_COLOR,
  HANDLE_RADIUS,
  HANDLE_SIZE,
};

// ============================================================
// 文字样式常量 - 从 tokens 获取
// ============================================================

export const TEXT = {
  titleFontSize: SHARED_TEXT.titleSize,           // 14px
  subtitleFontSize: SHARED_TEXT.subtitleSize,     // 12px
  labelFontSize: SHARED_TEXT.labelSize,           // 12px
  titleColor: SHARED_TEXT.titleColor,             // #ffffff
  subtitleColor: SHARED_TEXT.subtitleColor,       // rgba(255,255,255,0.7)
  labelColor: SHARED_TEXT.labelColor,             // #d1d5db
  titleFontWeight: SHARED_TEXT.titleWeight,       // 600
  subtitleFontWeight: SHARED_TEXT.subtitleWeight, // 500
};

// ============================================================
// CSS 变量（供 Vue 组件 scoped style 使用）
// ============================================================

/** 获取 CSS 变量对象 - 可用于 :style 绑定到根元素 */
export function getCSSVariables() {
  return {
    '--handle-radius': `${HANDLE_RADIUS}px`,
    '--handle-size': `${HANDLE_SIZE}px`,
    '--exec-color': EXEC_COLOR,
  };
}
