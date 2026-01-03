/**
 * Components Index
 * 
 * Re-exports all components for easy importing.
 */

// React Flow 节点/边组件从 adapter 导出
export {
  BlueprintNode,
  FunctionEntryNode,
  FunctionReturnNode,
  FunctionCallNode,
  ExecutionEdge,
  DataEdge,
  nodeTypes,
  edgeTypes,
  type BlueprintNodeProps,
  type BlueprintNodeType,
  type FunctionEntryNodeProps,
  type FunctionEntryNodeType,
  type FunctionReturnNodeProps,
  type FunctionReturnNodeType,
  type FunctionCallNodeProps,
  type FunctionCallNodeType,
} from '../editor/adapters/reactflow';

// UI 组件
export { AttributeEditor, type AttributeEditorProps } from './AttributeEditor';
export { UnifiedTypeSelector } from './UnifiedTypeSelector';
export { type TypeNode, type ScalarNode, type CompositeNode, parseType, serializeType } from '../services/typeNodeUtils';
export { NodePalette, type NodePaletteProps } from './NodePalette';
export { FunctionManager, type FunctionManagerProps } from './FunctionManager';
export { ExecutionPanel } from './ExecutionPanel';
export { ExecutionPin } from './ExecutionPin';
export { CreateProjectDialog, OpenProjectDialog, SaveProjectDialog } from './ProjectDialog';

// 左侧面板组件
export { LeftPanelTabs, type LeftPanelTabsProps } from './LeftPanelTabs';
export { TypeConstraintPanel } from './TypeConstraintPanel';
export { ConstraintNode, type ConstraintNodeProps } from './ConstraintNode';

// MainLayout 已移动到 app/ 目录
// 使用: import { MainLayout } from './app/MainLayout'
