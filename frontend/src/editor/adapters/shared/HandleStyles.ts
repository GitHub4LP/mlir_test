/**
 * Handle（端口连接点）样式
 * 
 * 统一的执行引脚和数据引脚样式定义
 * React Flow 和 Vue Flow 共用，确保视觉一致
 * 
 * 注意：此文件现在从 styles.ts 重新导出，保持向后兼容
 * 新代码应直接使用 styles.ts
 */

import { tokens } from '../../../generated/tokens';
import {
  getExecHandleStyle,
  getDataHandleStyle,
  EXEC_COLOR,
  HANDLE_RADIUS,
  HANDLE_SIZE,
} from './styles';

// 重新导出 styles.ts 中的函数
export { getExecHandleStyle, getDataHandleStyle, EXEC_COLOR, HANDLE_RADIUS, HANDLE_SIZE };

// ============================================================
// 执行引脚样式（右侧）- styles.ts 中没有，保留在这里
// ============================================================

/**
 * 获取执行引脚样式（右侧输出，三角形朝右）
 */
export function getExecHandleStyleRight(): Record<string, string | number> {
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
    background: 'transparent',
    borderRadius: 0,
  };
}

// ============================================================
// Vue Flow CSS 类名生成（用于 scoped style）
// ============================================================

/**
 * 获取执行引脚的 CSS 样式字符串（用于 Vue scoped style）
 */
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

/**
 * 获取执行引脚的 CSS 样式字符串（右侧，用于 Vue scoped style）
 */
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

/**
 * 获取数据引脚的 CSS 样式字符串（用于 Vue scoped style）
 */
export function getDataHandleCSS(): string {
  const size = tokens.node.handle.size;
  return `
    width: ${size}px !important;
    height: ${size}px !important;
    border: 2px solid ${tokens.node.bg} !important;
    border-radius: 50% !important;
  `;
}
