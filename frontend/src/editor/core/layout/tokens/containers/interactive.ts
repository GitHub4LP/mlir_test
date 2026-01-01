/**
 * 交互组件配置
 * editableName, typeLabel, button, handle
 * 
 * 每个配置包含 className 属性，用于 DOM 渲染器应用 CSS 样式
 */

import type { ContainerConfig } from '../types';

/** 可编辑名称配置 */
export const editableName = {
  paddingTop: 2,
  paddingRight: 4,
  paddingBottom: 2,
  paddingLeft: 4,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  cornerRadius: 3,
  minWidth: 24,
  minHeight: 16,
  className: 'layout-editable-name',
} as const satisfies ContainerConfig;

/** 类型标签配置 */
export const typeLabel = {
  paddingTop: 4,
  paddingRight: 8,
  paddingBottom: 4,
  paddingLeft: 8,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  cornerRadius: 3,
  minWidth: 40,
  minHeight: 22,
  fills: [{ type: 'SOLID', color: { r: 0.216, g: 0.255, b: 0.318 }, opacity: 1 }],
  className: 'layout-type-label',
} as const satisfies ContainerConfig;

/** 按钮配置 */
export const button = {
  width: 16,
  height: 16,
  minWidth: 16,
  minHeight: 16,
  className: 'layout-button',
} as const satisfies ContainerConfig;

/** Handle 配置 */
export const handle = {
  width: 12,
  height: 12,
  strokeWeight: 2,
  className: 'layout-handle',
} as const satisfies ContainerConfig;
