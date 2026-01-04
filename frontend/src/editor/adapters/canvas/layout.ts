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
import { layoutConfig, LAYOUT, getPinContentLayout } from '../shared/styles';

export { LAYOUT, getPinContentLayout };

/** 节点布局常量 */
export const NODE_LAYOUT = {
  MIN_WIDTH: LAYOUT.minWidth,
  HEADER_HEIGHT: LAYOUT.headerHeight,
  PIN_ROW_HEIGHT: LAYOUT.pinRowHeight,
  HANDLE_RADIUS: LAYOUT.handleRadius,
  HANDLE_OFFSET: 0,  // 默认值
  PADDING: LAYOUT.padding,
  BORDER_RADIUS: LAYOUT.borderRadius,
  DEFAULT_BG_COLOR: LAYOUT.nodeBg,
  DEFAULT_BORDER_COLOR: '#3d3d4d',  // 默认边框色
  SELECTED_BORDER_COLOR: LAYOUT.selectedBorderColor,
  HEADER_PADDING_X: LAYOUT.headerPaddingX,
  HEADER_PADDING_Y: LAYOUT.headerPaddingY,
} as const;

/** 边布局常量 */
export const EDGE_LAYOUT = {
  WIDTH: layoutConfig.edge.data.strokeWidth,
  SELECTED_WIDTH: 3,
  BEZIER_OFFSET: typeof layoutConfig.edge.bezierOffset === 'string' ? parseInt(layoutConfig.edge.bezierOffset) : layoutConfig.edge.bezierOffset,
  EXEC_COLOR: LAYOUT.execColor,
  DEFAULT_DATA_COLOR: layoutConfig.edge.data.defaultStroke ?? '#888888',
} as const;
