/**
 * Handle（端口连接点）样式
 * 
 * 统一的执行引脚和数据引脚样式定义
 * React Flow 和 Vue Flow 共用，确保视觉一致
 * 
 * 设计原则：
 * - 所有数值来自 StyleSystem，不硬编码
 * - 执行引脚：白色三角形（CSS border 技巧）
 * - 数据引脚：彩色圆形，颜色由类型决定
 */

import { StyleSystem } from '../../core/StyleSystem';

// ============================================================
// 执行引脚样式 - 白色三角形
// ============================================================

/**
 * 获取执行引脚样式（左侧输入，三角形朝右）
 * 使用 CSS border 技巧创建三角形
 */
export function getExecHandleStyle(): Record<string, string | number> {
  const execColor = StyleSystem.getEdgeStyle().execColor;
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
    background: 'transparent',
    borderRadius: 0,
  };
}

/**
 * 获取执行引脚样式（右侧输出，三角形朝右）
 * 右侧三角形方向与左侧相同
 */
export function getExecHandleStyleRight(): Record<string, string | number> {
  const execColor = StyleSystem.getEdgeStyle().execColor;
  return {
    width: 0,
    height: 0,
    minWidth: 0,
    minHeight: 0,
    padding: 0,
    borderStyle: 'solid',
    borderWidth: '5px 8px 5px 0',
    borderColor: `transparent ${execColor} transparent transparent`,
    backgroundColor: 'transparent',
    background: 'transparent',
    borderRadius: 0,
  };
}

// ============================================================
// 数据引脚样式 - 彩色圆形
// ============================================================

/**
 * 获取数据引脚样式
 * @param color - 引脚颜色（通常从 StyleSystem.getTypeColor() 获取）
 */
export function getDataHandleStyle(color: string): Record<string, string | number> {
  const nodeStyle = StyleSystem.getNodeStyle();
  const size = nodeStyle.handleRadius * 2;
  return {
    width: size,
    height: size,
    backgroundColor: color,
    border: `2px solid ${nodeStyle.backgroundColor}`,
    borderRadius: '50%',
  };
}

// ============================================================
// Vue Flow CSS 类名生成（用于 scoped style）
// ============================================================

/**
 * 获取执行引脚的 CSS 样式字符串（用于 Vue scoped style）
 */
export function getExecHandleCSSLeft(): string {
  const execColor = StyleSystem.getEdgeStyle().execColor;
  return `
    width: 0 !important;
    height: 0 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
    border: none !important;
    border-style: solid !important;
    border-width: 5px 0 5px 8px !important;
    border-color: transparent transparent transparent ${execColor} !important;
    border-radius: 0 !important;
  `;
}

/**
 * 获取执行引脚的 CSS 样式字符串（右侧，用于 Vue scoped style）
 */
export function getExecHandleCSSRight(): string {
  const execColor = StyleSystem.getEdgeStyle().execColor;
  return `
    width: 0 !important;
    height: 0 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
    border: none !important;
    border-style: solid !important;
    border-width: 5px 8px 5px 0 !important;
    border-color: transparent ${execColor} transparent transparent !important;
    border-radius: 0 !important;
  `;
}

/**
 * 获取数据引脚的 CSS 样式字符串（用于 Vue scoped style）
 */
export function getDataHandleCSS(): string {
  const nodeStyle = StyleSystem.getNodeStyle();
  const size = nodeStyle.handleRadius * 2;
  return `
    width: ${size}px !important;
    height: ${size}px !important;
    border: 2px solid ${nodeStyle.backgroundColor} !important;
    border-radius: 50% !important;
  `;
}

// ============================================================
// 常量导出（供直接使用）
// ============================================================

/** 执行引脚颜色 */
export const EXEC_COLOR = StyleSystem.getEdgeStyle().execColor;

/** Handle 半径 */
export const HANDLE_RADIUS = StyleSystem.getNodeStyle().handleRadius;

/** Handle 直径 */
export const HANDLE_SIZE = HANDLE_RADIUS * 2;
