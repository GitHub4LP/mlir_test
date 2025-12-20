/**
 * Canvas 渲染后端 - 布局计算（兼容层）
 * 
 * 此文件保留作为兼容层，实际实现已移动到 core/LayoutEngine.ts。
 * 所有导出都从 core 模块重导出。
 */

// 重导出所有类型和函数
export {
  computeNodeLayout,
  computeEdgePath,
  computeEdgeLayout,
  distanceToEdge,
  isPointInRect,
  isPointInCircle,
} from '../../core/LayoutEngine';

export type {
  HandleLayout,
  NodeLayout,
  EdgeLayout,
  ComputeNodeLayoutFn,
} from '../../core/LayoutEngine';

// 为了向后兼容，保留常量导出（从 StyleSystem 派生）
import { StyleSystem } from '../../core/StyleSystem';

const nodeStyle = StyleSystem.getNodeStyle();
const edgeStyle = StyleSystem.getEdgeStyle();

/** @deprecated 使用 StyleSystem.getNodeStyle() 代替 */
export const NODE_LAYOUT = {
  MIN_WIDTH: nodeStyle.minWidth,
  HEADER_HEIGHT: nodeStyle.headerHeight,
  PIN_ROW_HEIGHT: nodeStyle.pinRowHeight,
  HANDLE_RADIUS: nodeStyle.handleRadius,
  HANDLE_OFFSET: nodeStyle.handleOffset,
  PADDING: nodeStyle.padding,
  BORDER_RADIUS: nodeStyle.borderRadius,
  DEFAULT_BG_COLOR: nodeStyle.backgroundColor,
  DEFAULT_BORDER_COLOR: nodeStyle.borderColor,
  SELECTED_BORDER_COLOR: nodeStyle.selectedBorderColor,
} as const;

/** @deprecated 使用 StyleSystem.getEdgeStyle() 代替 */
export const EDGE_LAYOUT = {
  WIDTH: edgeStyle.width,
  SELECTED_WIDTH: edgeStyle.selectedWidth,
  BEZIER_OFFSET: edgeStyle.bezierOffset,
  EXEC_COLOR: edgeStyle.execColor,
  DEFAULT_DATA_COLOR: edgeStyle.defaultDataColor,
} as const;
