/**
 * 核心共享层
 * 
 * 包含所有渲染器共享的类型定义、接口和工具。
 */

// 样式系统
export { StyleSystem, DEFAULT_THEME } from './StyleSystem';
export type {
  NodeStyleConfig,
  EdgeStyleConfig,
  TextStyleConfig,
  ThemeConfig,
} from './StyleSystem';

// 渲染数据模型
export type {
  BorderRadius,
  RenderRect,
  RenderText,
  RenderPath,
  RenderCircle,
  RenderTriangle,
  Viewport,
  InteractionHint,
  OverlayInfo,
  RenderData,
} from './RenderData';
export {
  createDefaultViewport,
  createDefaultHint,
  createEmptyRenderData,
} from './RenderData';

// 渲染器接口
export type { IRenderer } from './IRenderer';
export { BaseRenderer } from './IRenderer';

// 输入事件
export type {
  Modifiers,
  PointerEventType,
  MouseButton,
  PointerInput,
  WheelInput,
  KeyEventType,
  KeyInput,
  RawInput,
  RawInputCallback,
} from './input';
export {
  createDefaultModifiers,
  extractModifiers,
  createPointerInput,
  createWheelInput,
  createKeyInput,
} from './input';

// 布局引擎
export {
  computeNodeLayout,
  computeEdgePath,
  computeEdgeLayout,
  distanceToEdge,
  isPointInRect,
  isPointInCircle,
} from './LayoutEngine';
export type {
  HandleLayout,
  NodeLayout,
  EdgeLayout,
  ComputeNodeLayoutFn,
} from './LayoutEngine';
