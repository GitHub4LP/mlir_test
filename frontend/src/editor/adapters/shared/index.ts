/**
 * 共享模块导出
 * 
 * 所有渲染器共享的工具和接口
 */

// 快捷键配置
export {
  type KeyBindings,
  type Modifiers,
  defaultKeyBindings,
  matchesShortcut,
  matchesAction,
  extractModifiersFromEvent,
  createKeyHandler,
} from './KeyBindings';

// 坐标系统
export {
  type Point,
  screenToCanvas,
  canvasToScreen,
  getScreenCoordinates,
  getCanvasCoordinates,
  zoomAtPoint,
  clampZoom,
} from './CoordinateSystem';

// 视口
export {
  type Viewport,
  defaultViewport,
  cloneViewport,
  viewportsEqual,
  fromReactFlowViewport,
  toReactFlowViewport,
} from './Viewport';
