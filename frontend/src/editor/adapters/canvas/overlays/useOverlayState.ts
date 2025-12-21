/**
 * useOverlayState - 覆盖层状态管理 Hook
 */

import { useReducer } from 'react';

// ============================================================
// 覆盖层状态类型
// ============================================================

/** 类型选择器覆盖层 */
export interface TypeSelectorOverlayState {
  kind: 'type-selector';
  nodeId: string;
  handleId: string;
  /** 画布坐标 */
  canvasX: number;
  canvasY: number;
  /** 当前类型 */
  currentType: string;
  /** 约束 */
  constraint?: string;
  /** 允许的类型列表 */
  allowedTypes?: string[];
}

/** 可编辑名称覆盖层 */
export interface EditableNameOverlayState {
  kind: 'editable-name';
  nodeId: string;
  /** 参数或返回值索引 */
  index: number;
  /** 是参数还是返回值 */
  target: 'param' | 'return';
  /** 画布坐标 */
  canvasX: number;
  canvasY: number;
  /** 当前名称 */
  currentName: string;
}

/** Traits 编辑器覆盖层 */
export interface TraitsEditorOverlayState {
  kind: 'traits-editor';
  nodeId: string;
  /** 画布坐标 */
  canvasX: number;
  canvasY: number;
}

/** 覆盖层状态联合类型 */
export type OverlayState = 
  | TypeSelectorOverlayState 
  | EditableNameOverlayState 
  | TraitsEditorOverlayState
  | null;

// ============================================================
// 覆盖层动作
// ============================================================

export type OverlayAction =
  | { type: 'show-type-selector'; payload: Omit<TypeSelectorOverlayState, 'kind'> }
  | { type: 'show-editable-name'; payload: Omit<EditableNameOverlayState, 'kind'> }
  | { type: 'show-traits-editor'; payload: Omit<TraitsEditorOverlayState, 'kind'> }
  | { type: 'close' };

function overlayReducer(_state: OverlayState, action: OverlayAction): OverlayState {
  switch (action.type) {
    case 'show-type-selector':
      return { kind: 'type-selector', ...action.payload };
    case 'show-editable-name':
      return { kind: 'editable-name', ...action.payload };
    case 'show-traits-editor':
      return { kind: 'traits-editor', ...action.payload };
    case 'close':
      return null;
  }
}

// ============================================================
// Hook
// ============================================================

export function useOverlayState() {
  return useReducer(overlayReducer, null);
}
