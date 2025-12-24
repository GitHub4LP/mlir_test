/**
 * 引脚样式定义
 * 
 * 统一的执行引脚和数据引脚样式
 * 使用 shared/styles 获取样式，确保与所有渲染器一致
 */

import type React from 'react';
import {
  getExecHandleStyle,
  getDataHandleStyle,
  getNodeContainerStyle,
  getNodeHeaderStyle,
} from '../../editor/adapters/shared/styles';

/**
 * 执行引脚样式 - 白色实心向右三角形
 */
export const execPinStyle: React.CSSProperties = getExecHandleStyle();

/**
 * 数据引脚样式 - 彩色圆形
 * @param color - 引脚颜色
 */
export function dataPinStyle(color: string): React.CSSProperties {
  return getDataHandleStyle(color);
}

// 重导出节点样式函数
export { getNodeContainerStyle, getNodeHeaderStyle };
