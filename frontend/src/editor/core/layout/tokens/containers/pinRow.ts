/**
 * 引脚行相关容器配置
 * pinArea, pinRow, pinRowContent, pinRowLeftSpacer, pinRowRightSpacer, pinRowSpacer
 */

import type { ContainerConfig } from '../types';

/** 引脚区域 */
export const pinArea = {
  layoutMode: 'VERTICAL',
  itemSpacing: 0,
  paddingTop: 0,
  paddingRight: 0,
  paddingBottom: 0,
  paddingLeft: 0,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
} as const satisfies ContainerConfig;

/** 引脚行 */
export const pinRow = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 0,
  paddingTop: 0,
  paddingRight: 0,
  paddingBottom: 0,
  paddingLeft: 0,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  minHeight: 24,
  counterAxisAlignItems: 'CENTER',
} as const satisfies ContainerConfig;

/** 引脚行左侧间隔 */
export const pinRowLeftSpacer = {
  layoutMode: 'HORIZONTAL',
  width: 6,
  minHeight: 1,
} as const satisfies ContainerConfig;

/** 引脚行右侧间隔 */
export const pinRowRightSpacer = {
  layoutMode: 'HORIZONTAL',
  width: 6,
  minHeight: 1,
} as const satisfies ContainerConfig;

/** 引脚行内容区 */
export const pinRowContent = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 0,
  paddingTop: 0,
  paddingRight: 0,
  paddingBottom: 0,
  paddingLeft: 0,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  primaryAxisAlignItems: 'SPACE_BETWEEN',
  minHeight: 24,
  counterAxisAlignItems: 'CENTER',
  fills: [{ type: 'SOLID', color: { r: 0.176, g: 0.176, b: 0.239 }, opacity: 1 }],
} as const satisfies ContainerConfig;

/** 引脚行间隔 */
export const pinRowSpacer = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 0,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  minWidth: 4,
} as const satisfies ContainerConfig;
