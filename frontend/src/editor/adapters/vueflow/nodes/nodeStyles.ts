/**
 * Vue Flow 节点共享样式
 * 
 * 从 shared/figmaStyles.ts 获取样式，确保与 Canvas/GPU/React Flow 一致
 * 
 * 设计原则：
 * - 所有数值来自 layoutTokens.json（Figma 格式）
 * - 样式函数从 figmaStyles.ts 获取，保持统一
 */

import type { CSSProperties } from 'vue';
import {
  layoutConfig,
  getNodeContainerStyle as figmaGetNodeContainerStyle,
  getHeaderContentStyle as figmaGetHeaderContentStyle,
  getPinAreaStyle,
  getPinRowStyle as figmaGetPinRowStyle,
  getPinRowContentStyle,
  getLeftPinGroupStyle,
  getRightPinGroupStyle,
  getPinContentStyle,
  getPinContentRightStyle,
  getPinRowSpacerStyle,
  getAttrContentStyle,
  getSummaryContentStyle,
  getTypeLabelStyle as figmaGetTypeLabelStyle,
  getHandleSize,
  getExecHandleStyle as figmaGetExecHandleStyle,
  getExecHandleStyleRight as figmaGetExecHandleStyleRight,
  getDataHandleStyle as figmaGetDataHandleStyle,
  getDialectColor as figmaGetDialectColor,
  getNodeTypeColor as figmaGetNodeTypeColor,
  getTypeColor as figmaGetTypeColor,
} from '../../shared/figmaStyles';

// ============================================================
// 常量导出
// ============================================================

/** 执行引脚颜色 */
export const EXEC_COLOR = layoutConfig.edge.exec.stroke ?? '#ffffff';

/** Handle 半径 */
export const HANDLE_RADIUS = getHandleSize() / 2;

/** Handle 直径 */
export const HANDLE_SIZE = getHandleSize();

// ============================================================
// 布局常量 - 从 layoutConfig 派生
// ============================================================

export const LAYOUT = {
  headerHeight: (() => {
    const padding = layoutConfig.headerContent.paddingTop ?? 0;
    const paddingBottom = layoutConfig.headerContent.paddingBottom ?? 0;
    return padding + paddingBottom + layoutConfig.text.title.fontSize;
  })(),
  pinRowHeight: layoutConfig.pinRow.minHeight ?? 28,
  padding: layoutConfig.node.paddingTop ?? 0,
  handleRadius: HANDLE_RADIUS,
  minWidth: layoutConfig.node.minWidth ?? 200,
  borderRadius: layoutConfig.headerContent.cornerRadius ?? layoutConfig.headerContent.topLeftRadius ?? 8,
};

// ============================================================
// 节点容器样式
// ============================================================

/** 节点容器样式 - 使用 figmaStyles */
export function getContainerStyle(selected: boolean): CSSProperties {
  return figmaGetNodeContainerStyle(selected) as CSSProperties;
}

/** 节点头部样式 - 使用 figmaStyles */
export function getHeaderStyle(headerColor: string): CSSProperties {
  return figmaGetHeaderContentStyle(headerColor) as CSSProperties;
}

/** 节点 body 样式 */
export function getBodyStyle(): CSSProperties {
  return getPinAreaStyle() as CSSProperties;
}

/** 引脚行样式 */
export function getPinRowStyle(): CSSProperties {
  return figmaGetPinRowStyle() as CSSProperties;
}

/** 引脚行内容样式 */
export function getPinRowContent(): CSSProperties {
  return getPinRowContentStyle() as CSSProperties;
}

/** 左侧引脚组样式 */
export function getLeftPinGroup(): CSSProperties {
  return getLeftPinGroupStyle() as CSSProperties;
}

/** 右侧引脚组样式 */
export function getRightPinGroup(): CSSProperties {
  return getRightPinGroupStyle() as CSSProperties;
}

/** 引脚内容样式（左对齐） */
export function getPinContent(): CSSProperties {
  return getPinContentStyle() as CSSProperties;
}

/** 引脚内容样式（右对齐） */
export function getPinContentRight(): CSSProperties {
  return getPinContentRightStyle() as CSSProperties;
}

/** 引脚行间隔样式 */
export function getPinRowSpacer(): CSSProperties {
  return getPinRowSpacerStyle() as CSSProperties;
}

/** 属性内容样式 */
export function getAttrContent(): CSSProperties {
  return getAttrContentStyle() as CSSProperties;
}

/** 摘要内容样式 */
export function getSummaryContent(): CSSProperties {
  return getSummaryContentStyle() as CSSProperties;
}

/** 类型标签样式 */
export function getTypeLabelStyle(bgColor?: string): CSSProperties {
  return figmaGetTypeLabelStyle(bgColor) as CSSProperties;
}

// ============================================================
// Handle 位置计算
// ============================================================

/** 计算 Handle 的 top 位置 */
export function getHandleTop(idx: number): string {
  const pinRowHeight = layoutConfig.pinRow.minHeight ?? 28;
  const headerHeight = LAYOUT.headerHeight;
  const padding = layoutConfig.node.paddingTop ?? 0;
  
  // Handle 在每行的中心
  const rowTop = headerHeight + padding + idx * pinRowHeight;
  const handleY = rowTop + pinRowHeight / 2;
  
  return `${handleY}px`;
}

// ============================================================
// 颜色获取
// ============================================================

/** 获取方言颜色 */
export function getDialectColor(dialect: string): string {
  return figmaGetDialectColor(dialect);
}

/** 获取执行引脚颜色 */
export function getExecColor(): string {
  return EXEC_COLOR;
}

/** 获取节点类型颜色 */
export function getNodeTypeColor(type: 'entry' | 'entryMain' | 'return' | 'returnMain' | 'call' | 'operation'): string {
  return figmaGetNodeTypeColor(type);
}

/** 获取类型颜色 */
export function getTypeColor(typeConstraint: string): string {
  return figmaGetTypeColor(typeConstraint);
}

// ============================================================
// Handle 样式
// ============================================================

/** 获取执行引脚样式（左侧输入） */
export function getExecHandleStyle(): CSSProperties {
  return figmaGetExecHandleStyle() as CSSProperties;
}

/** 获取执行引脚样式（右侧输出） */
export function getExecHandleStyleRight(): CSSProperties {
  return figmaGetExecHandleStyleRight() as CSSProperties;
}

/** 获取数据引脚样式 */
export function getDataHandleStyle(color: string): CSSProperties {
  return figmaGetDataHandleStyle(color) as CSSProperties;
}

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
    border-color: transparent transparent transparent ${EXEC_COLOR} !important;
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
    border-color: transparent ${EXEC_COLOR} transparent transparent !important;
    border-radius: 0 !important;
  `;
}

/** 获取数据引脚的 CSS 样式字符串 */
export function getDataHandleCSS(): string {
  const size = HANDLE_SIZE;
  const nodeBg = '#1e1e2e';
  return `
    width: ${size}px !important;
    height: ${size}px !important;
    border: 2px solid ${nodeBg} !important;
    border-radius: 50% !important;
  `;
}

// ============================================================
// 文字样式常量 - 从 layoutConfig 获取
// ============================================================

export const TEXT = {
  titleFontSize: layoutConfig.text.title.fontSize,
  subtitleFontSize: layoutConfig.text.subtitle.fontSize,
  labelFontSize: layoutConfig.text.label.fontSize,
  titleColor: layoutConfig.text.title.fill,
  subtitleColor: layoutConfig.text.subtitle.fill,
  labelColor: layoutConfig.text.label.fill,
  mutedColor: layoutConfig.text.muted.fill,
  titleFontWeight: layoutConfig.text.title.fontWeight,
  subtitleFontWeight: layoutConfig.text.subtitle.fontWeight,
};

// ============================================================
// 节点 Body 样式
// ============================================================

/** 节点 body padding 样式 */
export function getBodyPadding(): string {
  return `${layoutConfig.pinArea.paddingTop ?? 4}px`;
}

// ============================================================
// 引脚行样式常量
// ============================================================

export const PIN_ROW = {
  minHeight: layoutConfig.pinRow.minHeight ?? 28,
  paddingY: layoutConfig.leftPinGroup.paddingTop ?? 6,
  contentMarginLeft: 16,  // Handle 外侧到内容的距离
  contentMarginRight: 16,
  contentSpacing: layoutConfig.pinContent.itemSpacing ?? 2,
};

// ============================================================
// CSS 变量（供 Vue 组件 scoped style 使用）
// ============================================================

/** 获取 CSS 变量对象 - 可用于 :style 绑定到根元素 */
export function getCSSVariables(): Record<string, string> {
  return {
    // Handle
    '--handle-radius': `${HANDLE_RADIUS}px`,
    '--handle-size': `${HANDLE_SIZE}px`,
    '--exec-color': EXEC_COLOR,
    // Node
    '--node-bg-color': '#2d2d3d',
    '--node-border-color': '#3d3d4d',
    '--node-border-radius': `${layoutConfig.headerContent.topLeftRadius ?? 8}px`,
    // Text
    '--text-title-size': `${layoutConfig.text.title.fontSize}px`,
    '--text-title-color': layoutConfig.text.title.fill,
    '--text-title-weight': String(layoutConfig.text.title.fontWeight),
    '--text-subtitle-size': `${layoutConfig.text.subtitle.fontSize}px`,
    '--text-subtitle-color': layoutConfig.text.subtitle.fill,
    '--text-subtitle-weight': String(layoutConfig.text.subtitle.fontWeight),
    '--text-label-size': `${layoutConfig.text.label.fontSize}px`,
    '--text-label-color': layoutConfig.text.label.fill,
    '--text-muted-color': layoutConfig.text.muted.fill,
    // Pin Row
    '--pin-row-min-height': `${PIN_ROW.minHeight}px`,
    '--pin-row-padding-y': `${PIN_ROW.paddingY}px`,
    '--pin-content-margin-left': `${PIN_ROW.contentMarginLeft}px`,
    '--pin-content-margin-right': `${PIN_ROW.contentMarginRight}px`,
    '--pin-content-spacing': `${PIN_ROW.contentSpacing}px`,
    // Body
    '--body-padding': getBodyPadding(),
    // Button
    '--btn-danger-hover-color': layoutConfig.button?.danger?.hoverColor ?? '#f87171',
  };
}
