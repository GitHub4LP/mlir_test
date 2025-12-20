/**
 * MainLayout 组件
 * 
 * 主应用布局：
 * - 左侧：节点面板（MLIR 操作浏览/搜索）
 * - 中央：节点编辑画布
 * - 右侧：属性面板
 * - 底部：执行面板
 * 
 * 重构后：
 * - 不直接导入 @xyflow/react
 * - 通过 EditorContainer + INodeEditor 接口与编辑器交互
 * - 使用 editorStore 管理节点/边状态
 */

import { useCallback, useState, useEffect, useMemo, type ReactNode } from 'react';
import type { Node } from '@xyflow/react';

import { NodePalette } from '../components/NodePalette';
import { FunctionManager } from '../components/FunctionManager';
import { ExecutionPanel } from '../components/ExecutionPanel';
import { CreateProjectDialog, OpenProjectDialog, SaveProjectDialog } from '../components/ProjectDialog';
import { EditorContainer, type RendererType } from './components/EditorContainer';
import { useEditorFactory } from './hooks/useEditorFactory';
import { useGraphEditor } from './hooks/useGraphEditor';
import type { EditorSelection, ConnectionRequest } from '../editor/types';
import type { OperationDef, FunctionDef } from '../types';
import { useProjectStore } from '../stores/projectStore';
import { useDialectStore } from '../stores/dialectStore';
import { useEditorStore } from '../core/stores/editorStore';

// Layout components
import { ConnectionErrorToast, PropertiesPanel, ProjectToolbar } from '../components/layout';

/**
 * Props for MainLayout component
 */
export interface MainLayoutProps {
  /** Optional header content */
  header?: ReactNode;
  /** Optional footer content */
  footer?: ReactNode;
}

/**
 * MainLayout Component
 * 
 * Provides the main application layout with:
 * - Left: Node palette (operations grouped by dialect)
 * - Center: Node editor (via EditorContainer)
 * - Right: Properties panel
 * - Bottom: Execution panel
 */
export function MainLayout({ header, footer }: MainLayoutProps) {
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [executionPanelExpanded, setExecutionPanelExpanded] = useState(true);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Project dialog states
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isOpenDialogOpen, setIsOpenDialogOpen] = useState(false);
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);

  // Renderer state
  const [renderer, setRenderer] = useState<RendererType>('reactflow');
  
  // Editor factory
  const { createEditor } = useEditorFactory();
  
  // 使用 useGraphEditor hook
  const graphEditor = useGraphEditor();
  const {
    nodes,
    handleConnect: graphHandleConnect,
    handleDrop: graphHandleDrop,
    handleEdgeDoubleClick: graphHandleEdgeDoubleClick,
    copySelectedNodes,
    pasteNodes,
    deleteSelected,
    handleFunctionSelect,
    saveCurrentGraph,
    loadFunctionGraph,
    setViewport,
  } = graphEditor;
  
  // 渲染器切换处理
  const handleRendererChange = useCallback((newRenderer: RendererType) => {
    setRenderer(newRenderer);
  }, []);
  
  // GPU availability detection
  const webglAvailable = useMemo(() => {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2');
      return !!gl;
    } catch {
      return false;
    }
  }, []);
  
  const webgpuAvailable = useMemo(() => {
    return 'gpu' in navigator;
  }, []);

  // Get project state
  const project = useProjectStore(state => state.project);
  const currentFunctionId = useProjectStore(state => state.currentFunctionId);
  const createProject = useProjectStore(state => state.createProject);

  // Initialize dialect store on mount
  const initializeDialects = useDialectStore(state => state.initialize);
  useEffect(() => {
    initializeDialects();
  }, [initializeDialects]);

  // Derive selected node validity
  const effectiveSelectedNode = selectedNode && nodes.find(n => n.id === selectedNode.id)
    ? selectedNode
    : null;

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return;
      }

      if ((event.ctrlKey || event.metaKey) && event.key === 'c') {
        event.preventDefault();
        copySelectedNodes();
      }

      if ((event.ctrlKey || event.metaKey) && event.key === 'v') {
        event.preventDefault();
        pasteNodes();
      }

      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        deleteSelected();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [copySelectedNodes, pasteNodes, deleteSelected]);

  const handleFunctionDeleted = useCallback(() => {
    const currentId = useProjectStore.getState().currentFunctionId;
    if (currentId) {
      loadFunctionGraph(currentId);
    }
  }, [loadFunctionGraph]);

  const handleProjectCreated = useCallback(() => {
    const mainId = useProjectStore.getState().currentFunctionId;
    if (mainId) {
      loadFunctionGraph(mainId);
    }
  }, [loadFunctionGraph]);

  const handleProjectOpened = useCallback(() => {
    const mainId = useProjectStore.getState().currentFunctionId;
    if (mainId) {
      loadFunctionGraph(mainId);
    }
  }, [loadFunctionGraph]);

  // Initialize a default project if none exists
  useEffect(() => {
    if (!project) {
      createProject('Untitled Project', './untitled_project', ['arith', 'func']);
      setTimeout(() => {
        handleProjectCreated();
      }, 0);
    }
  }, [project, createProject, handleProjectCreated]);

  // Handle drag start from palette
  const handleDragStart = useCallback((_event: React.DragEvent, operation: OperationDef) => {
    console.log('Dragging operation:', operation.fullName);
  }, []);

  const handleFunctionDragStart = useCallback((_event: React.DragEvent, func: FunctionDef) => {
    console.log('Dragging function:', func.name);
  }, []);

  // EditorContainer 事件处理
  const handleConnect = useCallback((request: ConnectionRequest) => {
    const result = graphHandleConnect({
      source: request.source,
      sourceHandle: request.sourceHandle,
      target: request.target,
      targetHandle: request.targetHandle,
    });
    if (!result.success) {
      setConnectionError(result.error || 'Invalid connection');
      setTimeout(() => setConnectionError(null), 3000);
    }
  }, [graphHandleConnect]);

  const handleDrop = useCallback((x: number, y: number, dataTransfer: DataTransfer) => {
    graphHandleDrop(x, y, dataTransfer);
  }, [graphHandleDrop]);

  const handleEdgeDoubleClick = useCallback((edgeId: string) => {
    graphHandleEdgeDoubleClick(edgeId);
  }, [graphHandleEdgeDoubleClick]);

  const handleDeleteRequest = useCallback((nodeIds: string[], edgeIds: string[]) => {
    if (nodeIds.length > 0 || edgeIds.length > 0) {
      deleteSelected();
    }
  }, [deleteSelected]);

  const handleSelectionChange = useCallback((selection: EditorSelection) => {
    useEditorStore.getState().setSelection(selection);
    if (selection.nodeIds.length === 1) {
      const node = nodes.find(n => n.id === selection.nodeIds[0]);
      if (node) {
        setSelectedNode(node as unknown as Node);
      }
    } else {
      setSelectedNode(null);
    }
  }, [nodes]);

  const dismissError = useCallback(() => {
    setConnectionError(null);
  }, []);

  const toggleExecutionPanel = useCallback(() => {
    setExecutionPanelExpanded(prev => !prev);
  }, []);

  const executionPanelHeight = executionPanelExpanded ? 256 : 32;

  return (
    <div className="w-full h-screen flex flex-col bg-gray-900">
      {/* Project Toolbar */}
      <ProjectToolbar
        project={project}
        renderer={renderer}
        onRendererChange={handleRendererChange}
        webglAvailable={webglAvailable}
        webgpuAvailable={webgpuAvailable}
        onCreateClick={() => setIsCreateDialogOpen(true)}
        onOpenClick={() => setIsOpenDialogOpen(true)}
        onSaveClick={() => setIsSaveDialogOpen(true)}
      />

      {/* Optional Header */}
      {header && (
        <div className="flex-shrink-0 border-b border-gray-700">
          {header}
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Function Manager + Node Palette */}
        <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col flex-shrink-0">
          <div className="border-b border-gray-700 max-h-64 flex-shrink-0">
            <FunctionManager
              onFunctionSelect={handleFunctionSelect}
              onFunctionDeleted={handleFunctionDeleted}
            />
          </div>
          <div className="flex-1 flex flex-col overflow-hidden">
            <h2 className="text-lg font-semibold text-white p-4 pb-0 flex-shrink-0">Node Palette</h2>
            <NodePalette
              onDragStart={handleDragStart}
              onFunctionDragStart={handleFunctionDragStart}
            />
          </div>
        </div>

        {/* Center - Node Editor */}
        <div
          className="flex-1 flex flex-col overflow-hidden"
          style={{ paddingBottom: executionPanelHeight }}
        >
          <div className="flex-1 relative">
            <EditorContainer
              rendererType={renderer}
              createEditor={createEditor}
              onConnect={handleConnect}
              onDrop={handleDrop}
              onEdgeDoubleClick={handleEdgeDoubleClick}
              onDeleteRequest={handleDeleteRequest}
              onSelectionChange={handleSelectionChange}
              onViewportChange={setViewport}
            />

            {connectionError && (
              <ConnectionErrorToast
                message={connectionError}
                onClose={dismissError}
              />
            )}
          </div>
        </div>

        {/* Right Panel - Properties */}
        {effectiveSelectedNode && (
          <div className="w-72 bg-gray-800 border-l border-gray-700 flex-shrink-0 overflow-hidden">
            <PropertiesPanel selectedNode={effectiveSelectedNode} />
          </div>
        )}
      </div>

      {/* Bottom Panel - Execution */}
      <div
        className="flex-shrink-0 absolute bottom-0 left-64"
        style={{
          height: executionPanelHeight,
          right: effectiveSelectedNode ? 288 : 0,
        }}
      >
        <ExecutionPanel
          projectPath={project?.path}
          isExpanded={executionPanelExpanded}
          onToggleExpand={toggleExecutionPanel}
          onSaveCurrentGraph={() => {
            if (currentFunctionId) {
              saveCurrentGraph(currentFunctionId);
            }
          }}
          onSaveProject={async () => {
            if (!project?.path) return false;
            const saveProjectToPath = useProjectStore.getState().saveProjectToPath;
            if (currentFunctionId) {
              saveCurrentGraph(currentFunctionId);
            }
            return await saveProjectToPath(project.path);
          }}
        />
      </div>

      {/* Optional Footer */}
      {footer && (
        <div className="flex-shrink-0 border-t border-gray-700">
          {footer}
        </div>
      )}

      {/* Project Dialogs */}
      <CreateProjectDialog
        isOpen={isCreateDialogOpen}
        onClose={() => setIsCreateDialogOpen(false)}
        onCreated={handleProjectCreated}
      />
      <OpenProjectDialog
        isOpen={isOpenDialogOpen}
        onClose={() => setIsOpenDialogOpen(false)}
        onOpened={handleProjectOpened}
      />
      <SaveProjectDialog
        isOpen={isSaveDialogOpen}
        onClose={() => setIsSaveDialogOpen(false)}
        onSaveCurrentGraph={() => {
          if (currentFunctionId) {
            saveCurrentGraph(currentFunctionId);
          }
        }}
      />
    </div>
  );
}

export default MainLayout;
