/**
 * Canvas 覆盖层组件
 * 
 * 这些组件用于在 Canvas/GPU 渲染器上显示 HTML 交互 UI。
 * 覆盖层位置跟随画布变换，支持点击外部关闭。
 */

export { TypeSelectorOverlay } from './TypeSelectorOverlay';
export { EditableNameOverlay } from './EditableNameOverlay';
export { TraitsEditorOverlay } from './TraitsEditorOverlay';
export { OverlayContainer } from './OverlayContainer';
export { useOverlayState } from './useOverlayState';
export type { 
  OverlayState, 
  OverlayAction,
  TypeSelectorOverlayState,
  EditableNameOverlayState,
  TraitsEditorOverlayState,
} from './useOverlayState';
