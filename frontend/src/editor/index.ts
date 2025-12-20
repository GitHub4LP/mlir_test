/**
 * 节点编辑器模块导出
 */

// 类型
export type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
  NodeChange,
  EdgeChange,
  NodePositionChange,
  NodeSelectChange,
  NodeRemoveChange,
  EdgeSelectChange,
  EdgeRemoveChange,
} from './types';

// 辅助函数
export {
  applyNodeChanges,
  applyEdgeChanges,
  extractSelectionFromChanges,
} from './types';

// 接口
export type { INodeEditor } from './INodeEditor';
export { createEmptyCallbacks } from './INodeEditor';

// 注册表
export {
  registerNodeEditor,
  getAvailableEditors,
  createNodeEditor,
  getDefaultEditorName,
  hasNodeEditor,
  isNodeEditorAvailable,
} from './NodeEditorRegistry';

// ReactFlow 适配器
export { ReactFlowEditorWrapper } from './adapters/ReactFlowEditorWrapper';
export type { ReactFlowEditorHandle, ReactFlowEditorWrapperProps } from './adapters/ReactFlowEditorWrapper';

// Canvas 适配器
export { CanvasNodeEditor, createCanvasNodeEditor } from './adapters/CanvasNodeEditor';
export { CanvasEditorWrapper } from './adapters/CanvasEditorWrapper';
export type { CanvasEditorHandle, CanvasEditorWrapperProps } from './adapters/CanvasEditorWrapper';
