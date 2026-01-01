/**
 * 头部相关容器配置
 * headerWrapper, headerContent, titleGroup, badgesGroup, headerSpacer
 */

import type { ContainerConfig } from '../types';

/** 头部包装器 */
export const headerWrapper = {
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

/** 头部左侧间隔 */
export const headerLeftSpacer = {
  layoutMode: 'HORIZONTAL',
  width: 6,
  minHeight: 1,
} as const satisfies ContainerConfig;

/** 头部右侧间隔 */
export const headerRightSpacer = {
  layoutMode: 'HORIZONTAL',
  width: 6,
  minHeight: 1,
} as const satisfies ContainerConfig;

/** 头部内容区 */
export const headerContent = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 0,
  paddingTop: 8,
  paddingRight: 6,
  paddingBottom: 8,
  paddingLeft: 6,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  primaryAxisAlignItems: 'SPACE_BETWEEN',
  counterAxisAlignItems: 'CENTER',
  topLeftRadius: 8,
  topRightRadius: 8,
  bottomLeftRadius: 0,
  bottomRightRadius: 0,
} as const satisfies ContainerConfig;

/** 标题组 */
export const titleGroup = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 4,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  counterAxisAlignItems: 'BASELINE',
} as const satisfies ContainerConfig;

/** 徽章组 */
export const badgesGroup = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 4,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  counterAxisAlignItems: 'CENTER',
} as const satisfies ContainerConfig;

/** 头部间隔 */
export const headerSpacer = {
  layoutMode: 'HORIZONTAL',
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  minWidth: 8,
} as const satisfies ContainerConfig;
