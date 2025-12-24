/**
 * Overlay 类型定义
 * 
 * 分离到单独文件以避免 React Fast Refresh 警告
 */

/** 类型选择器状态 */
export interface TypeSelectorState {
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

/** 名称编辑器状态 */
export interface NameEditorState {
  kind: 'name-editor';
  nodeId: string;
  target: 'param' | 'return';
  index: number;
  canvasX: number;
  canvasY: number;
  currentName: string;
}

/** Traits 编辑器状态 */
export interface TraitsEditorState {
  kind: 'traits-editor';
  nodeId: string;
  canvasX: number;
  canvasY: number;
}

/** 覆盖层状态联合类型 */
export type OverlayState = TypeSelectorState | NameEditorState | TraitsEditorState | null;

/** 覆盖层动作 */
export type OverlayAction =
  | { type: 'show-type-selector'; payload: Omit<TypeSelectorState, 'kind'> }
  | { type: 'show-name-editor'; payload: Omit<NameEditorState, 'kind'> }
  | { type: 'show-traits-editor'; payload: Omit<TraitsEditorState, 'kind'> }
  | { type: 'close' };

/** Overlay reducer */
export function overlayReducer(_state: OverlayState, action: OverlayAction): OverlayState {
  switch (action.type) {
    case 'show-type-selector':
      return { kind: 'type-selector', ...action.payload };
    case 'show-name-editor':
      return { kind: 'name-editor', ...action.payload };
    case 'show-traits-editor':
      return { kind: 'traits-editor', ...action.payload };
    case 'close':
      return null;
  }
}
