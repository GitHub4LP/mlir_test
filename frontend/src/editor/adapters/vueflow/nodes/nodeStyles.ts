/**
 * Vue Flow 节点共享样式
 * 
 * 从 StyleSystem 和 HandleStyles 获取样式常量，确保与 Canvas/GPU/React Flow 一致
 * 
 * 设计原则：
 * - 所有数值来自 StyleSystem，不硬编码
 * - Handle 样式从 HandleStyles 获取，与 React Flow 保持一致
 */

import { StyleSystem } from '../../../core/StyleSystem';
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

const nodeStyle = StyleSystem.getNodeStyle();
const textStyle = StyleSystem.getTextStyle();
const edgeStyle = StyleSystem.getEdgeStyle();

// ============================================================
// 布局常量 - 从 StyleSystem 获取
// ============================================================

export const LAYOUT = {
  headerHeight: nodeStyle.headerHeight,      // 32px
  pinRowHeight: nodeStyle.pinRowHeight,      // 24px
  padding: nodeStyle.padding,                // 8px
  handleRadius: nodeStyle.handleRadius,      // 6px
  minWidth: nodeStyle.minWidth,              // 200px
  borderRadius: nodeStyle.borderRadius,      // 8px
};

// ============================================================
// 节点容器样式
// ============================================================

/** 节点容器样式 - 与 React Flow getNodeContainerStyle 一致 */
export function getContainerStyle(selected: boolean) {
  return {
    backgroundColor: nodeStyle.backgroundColor,
    border: `${selected ? nodeStyle.selectedBorderWidth : nodeStyle.borderWidth}px solid ${selected ? nodeStyle.selectedBorderColor : nodeStyle.borderColor}`,
    borderRadius: `${nodeStyle.borderRadius}px`,
    minWidth: `${nodeStyle.minWidth}px`,
    fontFamily: textStyle.fontFamily,
  };
}

/** 节点头部样式 - 与 React Flow getNodeHeaderStyle 一致 */
export function getHeaderStyle(headerColor: string) {
  return {
    backgroundColor: headerColor,
    borderRadius: `${nodeStyle.borderRadius}px ${nodeStyle.borderRadius}px 0 0`,
    padding: `${nodeStyle.padding}px 12px`,
    height: `${nodeStyle.headerHeight}px`,
    boxSizing: 'border-box' as const,
  };
}

/** 节点 body 样式 - 与 React Flow NodePins 的 px-1 py-1 一致 */
export function getBodyStyle() {
  return {
    padding: '4px',  // px-1 py-1 = 4px
  };
}

/** 引脚行样式 - 与 React Flow pinRowHeight 一致 */
export function getPinRowStyle() {
  return {
    height: `${nodeStyle.pinRowHeight}px`,
    display: 'flex',
    alignItems: 'center',
  };
}

// ============================================================
// Handle 位置计算
// ============================================================

// React Flow 使用 py-1.5 min-h-7，实际行高约 28px
const ACTUAL_PIN_ROW_HEIGHT = 28;

/** 计算 Handle 的 top 位置 - 与 React Flow py-1.5 min-h-7 一致 */
export function getHandleTop(idx: number): string {
  // header (32px) + body padding (4px) + row * 28px + row center (14px)
  const y = nodeStyle.headerHeight + 4 + idx * ACTUAL_PIN_ROW_HEIGHT + ACTUAL_PIN_ROW_HEIGHT / 2;
  return `${y}px`;
}

// ============================================================
// 颜色获取
// ============================================================

/** 获取方言颜色 */
export function getDialectColor(dialect: string): string {
  return StyleSystem.getDialectColor(dialect);
}

/** 获取执行引脚颜色 */
export function getExecColor(): string {
  return edgeStyle.execColor;
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
// 文字样式常量 - 从 StyleSystem 获取
// ============================================================

export const TEXT = {
  titleFontSize: textStyle.titleFontSize,           // 14px (text-sm)
  subtitleFontSize: textStyle.subtitleFontSize,     // 12px (text-xs)
  labelFontSize: textStyle.labelFontSize,           // 12px
  titleColor: textStyle.titleColor,                 // #ffffff
  subtitleColor: textStyle.subtitleColor,           // rgba(255,255,255,0.7)
  labelColor: textStyle.labelColor,                 // #cccccc
  titleFontWeight: textStyle.titleFontWeight,       // 600 (font-semibold)
  subtitleFontWeight: textStyle.subtitleFontWeight, // 500 (font-medium)
};

// ============================================================
// CSS 变量（供 Vue 组件 scoped style 使用）
// ============================================================

/** 获取 CSS 变量对象 - 可用于 :style 绑定到根元素 */
export function getCSSVariables() {
  return {
    '--node-header-height': `${nodeStyle.headerHeight}px`,
    '--node-pin-row-height': `${nodeStyle.pinRowHeight}px`,
    '--node-padding': `${nodeStyle.padding}px`,
    '--node-handle-radius': `${nodeStyle.handleRadius}px`,
    '--node-min-width': `${nodeStyle.minWidth}px`,
    '--node-border-radius': `${nodeStyle.borderRadius}px`,
    '--node-bg-color': nodeStyle.backgroundColor,
    // 文字样式
    '--text-title-size': `${textStyle.titleFontSize}px`,
    '--text-subtitle-size': `${textStyle.subtitleFontSize}px`,
    '--text-label-size': `${textStyle.labelFontSize}px`,
    '--text-title-color': textStyle.titleColor,
    '--text-subtitle-color': textStyle.subtitleColor,
    '--text-label-color': textStyle.labelColor,
    '--text-title-weight': textStyle.titleFontWeight,
    '--text-subtitle-weight': textStyle.subtitleFontWeight,
    // Handle 相关
    '--handle-radius': `${HANDLE_RADIUS}px`,
    '--handle-size': `${HANDLE_SIZE}px`,
    '--exec-color': EXEC_COLOR,
  };
}
