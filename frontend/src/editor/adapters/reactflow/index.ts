/**
 * React Flow Adapter
 * 
 * 将 React Flow 封装为 INodeEditor 实现。
 * 所有 React Flow 相关代码都在此目录内。
 */

// INodeEditor 实现
export { ReactFlowNodeEditor, createReactFlowNodeEditor } from './ReactFlowNodeEditor';

// 内部画布组件（供直接使用 React Flow 的场景）
export { ReactFlowCanvas } from './ReactFlowCanvas';
export type { ReactFlowCanvasHandle, ReactFlowCanvasProps } from './ReactFlowCanvas';

// Node types
export { nodeTypes } from './nodes';
export { BlueprintNode, FunctionEntryNode, FunctionReturnNode, FunctionCallNode } from './nodes';
export type {
  BlueprintNodeType,
  BlueprintNodeProps,
  FunctionEntryNodeType,
  FunctionEntryNodeProps,
  FunctionReturnNodeType,
  FunctionReturnNodeProps,
  FunctionCallNodeType,
  FunctionCallNodeProps,
} from './nodes';

// Edge types
export { edgeTypes, ExecutionEdge, DataEdge } from './edges';
export type { DataEdgeData, DataEdgeType } from './edges';

// Adapter utilities
export {
  toReactFlowNode,
  fromReactFlowNode,
  toReactFlowEdge,
  fromReactFlowEdge,
  toReactFlowViewport,
  fromReactFlowViewport,
  convertNodeChanges,
  convertEdgeChanges,
  extractSelection,
  applySelectionToNodes,
  applySelectionToEdges,
  toConnectionRequest,
} from '../ReactFlowAdapter';

// Connection utilities
export {
  type ConnectionValidationResult,
  getPortEffectiveSet,
  getPortOriginalConstraint,
  validateConnectionCount,
  validateConnection,
  createConnectionValidator,
  getNodeErrors,
} from './connectionUtils';
