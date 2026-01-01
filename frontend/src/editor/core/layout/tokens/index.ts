/**
 * 布局配置聚合入口
 * 自动发现并导出所有配置模块
 */

// 重导出类型
export type * from './types';

// 重导出容器配置
export * from './containers';
export { containers } from './containers';

// 重导出非容器配置
export { text } from './text';
export { colors, dialect, type, nodeType } from './colors';
export { edge } from './edge';
export { overlay, ui, canvas, minimap, buttonStyle } from './ui';
export { size, radius, border } from './sizing';
export { font } from './font';

// 导入所有配置用于聚合
import {
  node,
  headerWrapper,
  headerLeftSpacer,
  headerRightSpacer,
  headerContent,
  titleGroup,
  badgesGroup,
  headerSpacer,
  pinArea,
  pinRow,
  pinRowLeftSpacer,
  pinRowRightSpacer,
  pinRowContent,
  pinRowSpacer,
  leftPinGroup,
  rightPinGroup,
  pinContent,
  pinContentRight,
  attrArea,
  attrWrapper,
  attrLeftSpacer,
  attrRightSpacer,
  attrContent,
  labelColumn,
  valueColumn,
  attrLabel,
  attrValue,
  editableName,
  typeLabel,
  button,
  handle,
  summary,
  summaryWrapper,
  summaryLeftSpacer,
  summaryRightSpacer,
  summaryContent,
  summaryText,
  containers,
} from './containers';
import { text } from './text';
import { colors, dialect, type, nodeType } from './colors';
import { edge } from './edge';
import { overlay, ui, canvas, minimap, buttonStyle } from './ui';
import { size, radius, border } from './sizing';
import { font } from './font';

import type { LayoutConfig, ContainerConfig } from './types';

/**
 * 全局布局配置
 * 保持与原 layoutConfig 相同的访问方式
 */
export const layoutConfig: LayoutConfig = {
  // 容器配置（显式列出以保持类型）
  node,
  headerWrapper,
  headerLeftSpacer,
  headerRightSpacer,
  headerContent,
  titleGroup,
  badgesGroup,
  headerSpacer,
  pinArea,
  pinRow,
  pinRowLeftSpacer,
  pinRowRightSpacer,
  pinRowContent,
  pinRowSpacer,
  leftPinGroup,
  rightPinGroup,
  pinContent,
  pinContentRight,
  attrArea,
  attrWrapper,
  attrLeftSpacer,
  attrRightSpacer,
  attrContent,
  labelColumn,
  valueColumn,
  attrLabel,
  attrValue,
  editableName,
  typeLabel,
  summary,
  summaryWrapper,
  summaryLeftSpacer,
  summaryRightSpacer,
  summaryContent,
  summaryText,
  handle,
  // 非容器配置
  text,
  edge,
  nodeType,
  colors,
  dialect,
  type,
  button,
  buttonStyle,
  overlay,
  ui,
  canvas,
  minimap,
  size,
  radius,
  border,
  font,
};

/** 默认容器配置（Figma 格式） */
const defaultContainerConfig: ContainerConfig = {
  layoutMode: 'HORIZONTAL',
  itemSpacing: 0,
  primaryAxisSizingMode: 'AUTO',
  counterAxisSizingMode: 'AUTO',
};

/**
 * 获取指定容器的配置
 * @param type - 容器类型
 * @returns 容器配置
 */
export function getContainerConfig(type: string): ContainerConfig {
  // 优先从 containers 映射表查找
  if (type in containers) {
    return containers[type];
  }
  
  // 兼容：从 layoutConfig 查找（排除非容器配置）
  const key = type as keyof LayoutConfig;
  const config = layoutConfig[key];
  
  if (config && typeof config === 'object' && !('fontFamily' in config) && !('bezierOffset' in config) && !('entry' in config)) {
    return config as ContainerConfig;
  }
  
  // 返回默认配置
  return defaultContainerConfig;
}
