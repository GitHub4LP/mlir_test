/**
 * 摘要区域相关容器配置
 * summary, summaryWrapper, summaryContent, summaryText
 */

import type { ContainerConfig } from '../types';

/** 摘要区域 */
export const summary = {
  layoutMode: 'HORIZONTAL',
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  paddingTop: 4,
  paddingRight: 12,
  paddingBottom: 4,
  paddingLeft: 12,
} as const satisfies ContainerConfig;

/** 摘要包装器 */
export const summaryWrapper = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 0,
  paddingTop: 0,
  paddingRight: 0,
  paddingBottom: 0,
  paddingLeft: 0,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  overflow: 'hidden',
  // Overlay 模式：不参与父容器宽度计算，但宽度填充父容器
  // layoutGrow: 1 让 layoutNormal 使用 contentWidth 而不是 measuredWidth
  overlay: true,
  overlayHeight: 28,
  layoutGrow: 1,
} as const satisfies ContainerConfig;

/** 摘要左侧间隔 */
export const summaryLeftSpacer = {
  layoutMode: 'HORIZONTAL',
  width: 6,
  minHeight: 1,
} as const satisfies ContainerConfig;

/** 摘要右侧间隔 */
export const summaryRightSpacer = {
  layoutMode: 'HORIZONTAL',
  width: 6,
  minHeight: 1,
} as const satisfies ContainerConfig;

/** 摘要内容区 */
export const summaryContent = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 0,
  paddingTop: 6,
  paddingRight: 6,
  paddingBottom: 6,
  paddingLeft: 6,
  layoutGrow: 1,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  fills: [{ type: 'SOLID', color: { r: 0.176, g: 0.176, b: 0.239 }, opacity: 1 }],
  topLeftRadius: 0,
  topRightRadius: 0,
  bottomLeftRadius: 8,
  bottomRightRadius: 8,
  // 允许收缩，让子元素的 textOverflow 生效
  overflow: 'hidden',
  minWidth: 0,
} as const satisfies ContainerConfig;

/** 摘要文本 */
export const summaryText = {
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  textOverflow: 'ellipsis',
  // layoutGrow: 1 让文本填充 summaryContent 的剩余空间
  // 这样 box.width 才是正确的可用宽度，用于 ellipsis 截断
  layoutGrow: 1,
  minWidth: 0,
} as const satisfies ContainerConfig;
