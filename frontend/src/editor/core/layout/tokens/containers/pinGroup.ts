/**
 * 引脚组相关容器配置
 * leftPinGroup, rightPinGroup, pinContent, pinContentRight
 */

import type { ContainerConfig } from '../types';

/** 左侧引脚组 */
export const leftPinGroup = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 4,
  paddingTop: 4,
  paddingRight: 0,
  paddingBottom: 4,
  paddingLeft: 0,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  counterAxisAlignItems: 'CENTER',
} as const satisfies ContainerConfig;

/** 右侧引脚组 */
export const rightPinGroup = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 4,
  paddingTop: 4,
  paddingRight: 0,
  paddingBottom: 4,
  paddingLeft: 0,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  counterAxisAlignItems: 'CENTER',
} as const satisfies ContainerConfig;

/** 引脚内容（左对齐） */
export const pinContent = {
  layoutMode: 'VERTICAL',
  itemSpacing: 2,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  counterAxisAlignItems: 'MIN',
} as const satisfies ContainerConfig;

/** 引脚内容（右对齐） */
export const pinContentRight = {
  layoutMode: 'VERTICAL',
  itemSpacing: 2,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  counterAxisAlignItems: 'MAX',
} as const satisfies ContainerConfig;
