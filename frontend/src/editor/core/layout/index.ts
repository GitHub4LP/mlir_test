/**
 * Layout 模块导出
 * 基于 Figma Auto Layout 的布局系统
 */

// 类型导出
export type {
  // 基础类型
  SizingMode,
  Alignment,
  Direction,
  Spacing,
  Padding,
  CornerRadius,
  // 配置类型
  ContainerConfig,
  TextStyleConfig,
  TextConfig,
  EdgeStyleConfig,
  EdgeConfig,
  NodeTypeConfig,
  LayoutConfig,
  // LayoutBox 相关
  HitTestBehavior,
  Interactive,
  LayoutBoxStyle,
  LayoutBoxText,
  LayoutBox,
  HitResult,
  // 输入事件
  Modifiers,
  InputEvent,
  PointerInputEvent,
  WheelInputEvent,
  KeyboardInputEvent,
  FocusInputEvent,
  AnyInputEvent,
  // 布局计算
  LayoutNode,
  Size,
  NormalizedPadding,
} from './types';

// LayoutConfig 导出
export {
  layoutConfig,
  getContainerConfig,
  normalizePadding,
  formatPadding,
  // 颜色工具函数
  getDialectColor,
  getNodeTypeColor,
  getTypeColor,
} from './LayoutConfig';

// LayoutEngine 导出
export {
  measure,
  layout,
  computeLayout,
} from './LayoutEngine';

// buildNodeLayoutTree 导出
export {
  buildNodeLayoutTree,
  isEntryNode,
  isReturnNode,
  supportsParamEdit,
  supportsReturnEdit,
} from './buildNodeLayoutTree';

// DOMRenderer 导出
export {
  DOMRenderer,
} from './DOMRenderer';

export type {
  DOMRendererProps,
  HandleRenderConfig,
  TypeSelectorRenderConfig,
  EditableNameRenderConfig,
  ButtonRenderConfig,
  InteractiveRenderers,
  CallbackMap,
} from './DOMRenderer';

// figmaToCSS 导出
export { figmaToCSS, figmaColorToCSS, figmaFillsToCSS, figmaPaintToCSS } from './figmaToCSS';

// configToCSS 导出
export { configToFlexboxStyle, getFlexboxStyleForType } from './configToCSS';

// 便捷函数：GraphNode → LayoutBox
import { buildNodeLayoutTree as _buildNodeLayoutTree } from './buildNodeLayoutTree';
import { computeLayout as _computeLayout } from './LayoutEngine';
import type { LayoutBox } from './types';
import type { GraphNode } from '../../../types';

/**
 * 计算节点的 LayoutBox
 * 完整流程：GraphNode → LayoutNode → LayoutBox
 * @param node - GraphNode
 * @param nodeX - 节点在画布上的 X 坐标
 * @param nodeY - 节点在画布上的 Y 坐标
 * @returns LayoutBox（坐标已偏移到画布位置，样式已设置）
 */
export function computeNodeLayoutBox(
  node: GraphNode, 
  nodeX: number, 
  nodeY: number
): LayoutBox {
  // 1. 构建 LayoutNode 树（包含动态样式如 headerColor）
  const layoutNode = _buildNodeLayoutTree(node);
  
  // 2. 计算布局
  const layoutBox = _computeLayout(layoutNode);
  
  // 3. 偏移到画布位置
  layoutBox.x = nodeX;
  layoutBox.y = nodeY;
  
  return layoutBox;
}

// hitTest 导出
export {
  hitTestLayoutBox,
  findBoxInPath,
  findBoxByIdPrefix,
  getHitId,
  parseInteractiveId,
  extractHandlePositions,
  isOutputHandle,
} from './hitTest';

export type { HandlePosition } from './hitTest';

// cache 导出
export { LayoutCache, layoutCache, computeNodeDataHash } from './cache';

// TextMeasureCache 导出
export { TextMeasureCache, textMeasureCache } from './TextMeasureCache';

// sizeProvider 导出
export type { SizeProvider, DOMSizeProvider, InteractiveComponentType } from './sizeProvider';
export {
  ConfigSizeProvider,
  configSizeProvider,
  DOMSizeProviderImpl,
  domSizeProvider,
  setSizeProvider,
  getSizeProvider,
} from './sizeProvider';

// tokens 导出
export { containers } from './tokens';
export * from './tokens';
