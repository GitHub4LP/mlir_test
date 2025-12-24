/**
 * Canvas 渲染后端模块
 * 
 * 内部实现，供 CanvasNodeEditor 使用。
 */

// 图元类型
export type {
  RenderRect,
  RenderText,
  RenderPath,
  RenderCircle,
  InteractionHint,
  OverlayInfo,
  RenderData,
  Viewport,
} from './types';

export {
  createDefaultViewport,
  createDefaultHint,
  createEmptyRenderData,
} from './types';

// 原始输入类型
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

// 渲染后端接口
export type { IRenderer } from './IRenderer';
export { BaseRenderer } from './IRenderer';

// 图控制器
export { GraphController } from './GraphController';
export type { ControllerState, HitResult } from './GraphController';

// 布局工具
export type {
  HandleLayout,
  NodeLayout,
  EdgeLayout,
  ComputeNodeLayoutFn,
} from './layout';

export {
  NODE_LAYOUT,
  EDGE_LAYOUT,
  computeNodeLayout,
  computeEdgePath,
  computeEdgeLayout,
  distanceToEdge,
  isPointInRect,
  isPointInCircle,
} from './layout';

// Canvas 渲染器
export { CanvasRenderer } from './CanvasRenderer';

// 多层 Canvas 渲染器
export { MultiLayerCanvasRenderer } from './MultiLayerCanvasRenderer';
export type { MultiLayerCanvasConfig } from './MultiLayerCanvasRenderer';

// 多层架构组件
export * from './layers';
export * from './ui';

// 性能监控
export { PerformanceMonitor, performanceMonitor } from './PerformanceMonitor';
export type { PerformanceMetrics, PerformanceCallback } from './PerformanceMonitor';

// 覆盖层管理
export { OverlayManager, getOverlayManager, resetOverlayManager } from './OverlayManager';
export type { OverlayType, OverlayConfig, ActiveOverlay, OverlayRenderCallback } from './OverlayManager';

// 扩展命中测试
export {
  hitTestTypeLabel,
  hitTestHandle,
  hitTestNode,
  hitTestNodeComplete,
  getTypeLabelPosition,
} from './HitTest';
export type {
  HitNone,
  HitNode,
  HitHandle,
  HitTypeLabel,
  HitAttribute,
  HitEdge,
} from './HitTest';
