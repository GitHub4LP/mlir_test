/**
 * Canvas 渲染后端 - 布局计算
 * 
 * 统一使用 LayoutBox 系统，此文件仅保留边计算和几何工具函数。
 */

// 几何工具函数
export {
  computeEdgePath,
  distanceToEdge,
  isPointInRect,
  isPointInCircle,
} from '../../core/geometry';

// 从 Design Tokens 派生的常量
import { tokens, LAYOUT, getPinContentLayout } from '../shared/styles';

export { LAYOUT, getPinContentLayout };

/** 节点布局常量 */
export const NODE_LAYOUT = {
  MIN_WIDTH: LAYOUT.minWidth,
  HEADER_HEIGHT: LAYOUT.headerHeight,
  PIN_ROW_HEIGHT: LAYOUT.pinRowHeight,
  HANDLE_RADIUS: LAYOUT.handleRadius,
  HANDLE_OFFSET: typeof tokens.node.handle.offset === 'string' ? parseInt(tokens.node.handle.offset) : tokens.node.handle.offset,
  PADDING: LAYOUT.padding,
  BORDER_RADIUS: LAYOUT.borderRadius,
  DEFAULT_BG_COLOR: tokens.node.bg,
  DEFAULT_BORDER_COLOR: tokens.node.border.color,
  SELECTED_BORDER_COLOR: tokens.node.selected.borderColor,
  HEADER_PADDING_X: LAYOUT.headerPaddingX,
  HEADER_PADDING_Y: LAYOUT.headerPaddingY,
} as const;

/** 边布局常量 */
export const EDGE_LAYOUT = {
  WIDTH: tokens.edge.width,
  SELECTED_WIDTH: tokens.edge.selectedWidth,
  BEZIER_OFFSET: typeof tokens.edge.bezierOffset === 'string' ? parseInt(tokens.edge.bezierOffset) : tokens.edge.bezierOffset,
  EXEC_COLOR: tokens.edge.exec.color,
  DEFAULT_DATA_COLOR: tokens.edge.data.defaultColor,
} as const;
