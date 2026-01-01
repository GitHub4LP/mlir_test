/**
 * 属性区域相关容器配置
 * attrArea, attrWrapper, attrContent, labelColumn, valueColumn, attrLabel, attrValue
 */

import type { ContainerConfig } from '../types';

/** 属性区域 */
export const attrArea = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 8,
  paddingTop: 4,
  paddingRight: 12,
  paddingBottom: 4,
  paddingLeft: 12,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
} as const satisfies ContainerConfig;

/** 属性包装器 */
export const attrWrapper = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 0,
  paddingTop: 0,
  paddingRight: 0,
  paddingBottom: 0,
  paddingLeft: 0,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
} as const satisfies ContainerConfig;

/** 属性左侧间隔 */
export const attrLeftSpacer = {
  layoutMode: 'HORIZONTAL',
  width: 6,
  minHeight: 1,
} as const satisfies ContainerConfig;

/** 属性右侧间隔 */
export const attrRightSpacer = {
  layoutMode: 'HORIZONTAL',
  width: 6,
  minHeight: 1,
} as const satisfies ContainerConfig;

/** 属性内容区 */
export const attrContent = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 8,
  paddingTop: 4,
  paddingRight: 6,
  paddingBottom: 4,
  paddingLeft: 6,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  fills: [{ type: 'SOLID', color: { r: 0.176, g: 0.176, b: 0.239 }, opacity: 1 }],
} as const satisfies ContainerConfig;

/** 标签列 */
export const labelColumn = {
  layoutMode: 'VERTICAL',
  itemSpacing: 4,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  counterAxisAlignItems: 'MAX',
} as const satisfies ContainerConfig;

/** 值列 */
export const valueColumn = {
  layoutMode: 'VERTICAL',
  itemSpacing: 4,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
} as const satisfies ContainerConfig;

/** 属性标签 */
export const attrLabel = {
  primaryAxisSizingMode: 'AUTO',
  height: 20,
  counterAxisAlignItems: 'CENTER',
} as const satisfies ContainerConfig;

/** 属性值 */
export const attrValue = {
  primaryAxisSizingMode: 'AUTO',
  height: 20,
  counterAxisAlignItems: 'CENTER',
} as const satisfies ContainerConfig;
