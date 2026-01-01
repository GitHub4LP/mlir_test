/**
 * 节点根容器配置
 */

import type { ContainerConfig } from '../types';

/** 节点根容器 */
export const node = {
  layoutMode: 'VERTICAL',
  itemSpacing: 0,
  paddingTop: 0,
  paddingRight: 0,
  paddingBottom: 0,
  paddingLeft: 0,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
  // 不设置 minWidth，完全由内容决定宽度
  // 设置 relative 定位，让 overlay 子元素的 absolute 定位相对于节点
  position: 'relative',
  // 设置 overflow: hidden，防止 overlay 子元素溢出
  overflow: 'hidden',
  selected: {
    stroke: '#60a5fa',
    strokeWidth: 2,
  },
} as const satisfies ContainerConfig;
