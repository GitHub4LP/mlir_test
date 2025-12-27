/**
 * 共享模块导出
 * 
 * 所有渲染器共享的工具和接口
 */

// 快捷键配置
export {
  type KeyAction,
  type KeyBindings,
  type Modifiers,
  type KeyBindingInfo,
  type ReactFlowKeyConfig,
  type VueFlowKeyConfig,
  defaultKeyBindings,
  matchesShortcut,
  matchesAction,
  extractModifiersFromEvent,
  createKeyHandler,
  // React Flow / Vue Flow 配置生成
  getReactFlowKeyConfig,
  getReactFlowCustomActions,
  getVueFlowKeyConfig,
  getVueFlowCustomActions,
  // 用户配置管理
  getAllKeyBindingInfos,
  loadUserKeyBindings,
  saveUserKeyBindings,
  resetKeyBindings,
  exportKeyBindings,
  importKeyBindings,
  updateKeyBinding,
  formatShortcutForDisplay,
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

// Handle 样式（从 styles.ts 导入）
export {
  getExecHandleStyle,
  getExecHandleStyleRight,
  getDataHandleStyle,
  getExecHandleCSSLeft,
  getExecHandleCSSRight,
  getDataHandleCSS,
  EXEC_COLOR,
  HANDLE_RADIUS,
  HANDLE_SIZE,
} from './styles';

// 连线验证适配器
export {
  type GetPortTypeFn,
  type ConnectionValidationResult,
  // 核心验证
  validatePorts,
  // 框架适配器
  createReactFlowValidator,
  createVueFlowValidator,
  createValidatorWithStore,
  // 端口类型检测（便捷重导出）
  isExecPort,
  isDataPort,
} from './ConnectionValidator';

// 编辑器覆盖层
export {
  EditorOverlay,
} from './EditorOverlay';

// 覆盖层类型和 reducer
export {
  type TypeSelectorState,
  type NameEditorState,
  type TraitsEditorState,
  type OverlayState,
  type OverlayAction,
  overlayReducer,
} from './overlayTypes';

// 端口类型信息
export {
  type PortTypeInfo,
  getPortTypeInfo,
} from './PortTypeInfo';
