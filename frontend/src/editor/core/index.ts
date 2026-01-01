/**
 * 核心共享层
 * 
 * 包含所有渲染器共享的类型定义、接口和工具。
 * 
 * 布局系统已统一使用 LayoutBox，详见 core/layout/
 */

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

// 几何工具函数
export {
  computeEdgePath,
  distanceToEdge,
  isPointInRect,
  isPointInCircle,
} from './geometry';

// 新布局系统
export * from './layout';
