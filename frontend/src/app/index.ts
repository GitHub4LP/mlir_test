/**
 * App 模块导出
 * 
 * 此模块包含不直接依赖 React Flow 的 UI 组件。
 * 通过 INodeEditor 接口与编辑器交互。
 */

// 主布局
export { MainLayout } from './MainLayout';
export type { MainLayoutProps } from './MainLayout';

// 编辑器容器
export { EditorContainer } from './components/EditorContainer';
export type { EditorContainerProps, RendererType } from './components/EditorContainer';

// Hooks
export { useEditor } from './hooks/useEditor';
export { useGraphEditor } from './hooks/useGraphEditor';
export { useEditorFactory } from './hooks/useEditorFactory';
