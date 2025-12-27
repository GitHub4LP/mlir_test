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

// 为了向后兼容，保留常量导出（从 Design Tokens 派生）
import { tokens, LAYOUT, getPinContentLayout } from '../shared/styles';

// 重导出 LAYOUT 和布局函数供其他模块使用
export { LAYOUT, getPinContentLayout };

/** @deprecated 使用 tokens 或 LAYOUT 代替 */
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
  // 新增：与 ReactFlow 一致的头部布局
  HEADER_PADDING_X: LAYOUT.headerPaddingX,
  HEADER_PADDING_Y: LAYOUT.headerPaddingY,
} as const;

/** @deprecated 使用 tokens 代替 */
export const EDGE_LAYOUT = {
  WIDTH: tokens.edge.width,
  SELECTED_WIDTH: tokens.edge.selectedWidth,
  BEZIER_OFFSET: typeof tokens.edge.bezierOffset === 'string' ? parseInt(tokens.edge.bezierOffset) : tokens.edge.bezierOffset,
  EXEC_COLOR: tokens.edge.exec.color,
  DEFAULT_DATA_COLOR: tokens.edge.data.defaultColor,
} as const;
