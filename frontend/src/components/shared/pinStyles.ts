/**
 * 引脚样式定义
 * 
 * 统一的执行引脚和数据引脚样式
 * 使用 HandleStyles 获取样式，确保与 Vue Flow 和 Canvas/GPU 渲染器一致
 */

import type React from 'react';
import { StyleSystem } from '../../editor/core/StyleSystem';
import { getExecHandleStyle, getDataHandleStyle } from '../../editor/adapters/shared/HandleStyles';

/**
 * 执行引脚样式 - 白色实心向右三角形
 * 从 HandleStyles 获取，确保与其他渲染器一致
 */
export const execPinStyle: React.CSSProperties = getExecHandleStyle() as React.CSSProperties;

/**
 * 数据引脚样式 - 彩色圆形
 * @param color - 引脚颜色（通常从 StyleSystem.getTypeColor() 获取）
 */
export function dataPinStyle(color: string): React.CSSProperties {
  return getDataHandleStyle(color) as React.CSSProperties;
}

/**
 * 获取节点容器样式
 * @param selected - 是否选中
 */
export function getNodeContainerStyle(selected: boolean) {
  const nodeStyle = StyleSystem.getNodeStyle();
  return {
    backgroundColor: nodeStyle.backgroundColor,
    border: `${selected ? nodeStyle.selectedBorderWidth : nodeStyle.borderWidth}px solid ${selected ? nodeStyle.selectedBorderColor : nodeStyle.borderColor}`,
    borderRadius: `${nodeStyle.borderRadius}px`,
  };
}

/**
 * 获取节点头部样式
 * @param headerColor - 头部背景色（通常从 StyleSystem.getDialectColor() 获取）
 * 
 * 与 Vue Flow nodeStyles.ts 的 getHeaderStyle 保持一致
 */
export function getNodeHeaderStyle(headerColor: string) {
  const nodeStyle = StyleSystem.getNodeStyle();
  return {
    backgroundColor: headerColor,
    borderRadius: `${nodeStyle.borderRadius}px ${nodeStyle.borderRadius}px 0 0`,
    padding: `${nodeStyle.padding}px 12px`,
    height: `${nodeStyle.headerHeight}px`,
    boxSizing: 'border-box' as const,
  };
}
