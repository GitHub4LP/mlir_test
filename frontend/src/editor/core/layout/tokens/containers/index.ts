/**
 * 容器配置聚合入口
 * 自动发现并导出所有容器配置
 */

// Node
export { node } from './node';

// Header
export {
  headerWrapper,
  headerLeftSpacer,
  headerRightSpacer,
  headerContent,
  titleGroup,
  badgesGroup,
  headerSpacer,
} from './header';

// Pin Row
export {
  pinArea,
  pinRow,
  pinRowLeftSpacer,
  pinRowRightSpacer,
  pinRowContent,
  pinRowSpacer,
} from './pinRow';

// Pin Group
export {
  leftPinGroup,
  rightPinGroup,
  pinContent,
  pinContentRight,
} from './pinGroup';

// Attr Area
export {
  attrArea,
  attrWrapper,
  attrLeftSpacer,
  attrRightSpacer,
  attrContent,
  labelColumn,
  valueColumn,
  attrLabel,
  attrValue,
} from './attrArea';

// Summary
export {
  summary,
  summaryWrapper,
  summaryLeftSpacer,
  summaryRightSpacer,
  summaryContent,
  summaryText,
} from './summary';

// Interactive
export {
  editableName,
  typeLabel,
  button,
  handle,
} from './interactive';

// 聚合所有容器配置为对象（用于动态查找）
import { node } from './node';
import {
  headerWrapper,
  headerLeftSpacer,
  headerRightSpacer,
  headerContent,
  titleGroup,
  badgesGroup,
  headerSpacer,
} from './header';
import {
  pinArea,
  pinRow,
  pinRowLeftSpacer,
  pinRowRightSpacer,
  pinRowContent,
  pinRowSpacer,
} from './pinRow';
import {
  leftPinGroup,
  rightPinGroup,
  pinContent,
  pinContentRight,
} from './pinGroup';
import {
  attrArea,
  attrWrapper,
  attrLeftSpacer,
  attrRightSpacer,
  attrContent,
  labelColumn,
  valueColumn,
  attrLabel,
  attrValue,
} from './attrArea';
import {
  summary,
  summaryWrapper,
  summaryLeftSpacer,
  summaryRightSpacer,
  summaryContent,
  summaryText,
} from './summary';
import {
  editableName,
  typeLabel,
  button,
  handle,
} from './interactive';

import type { ContainerConfig } from '../types';

/** 所有容器配置的映射表 */
export const containers: Record<string, ContainerConfig> = {
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
};
