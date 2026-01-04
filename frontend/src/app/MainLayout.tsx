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

import { useCallback, useState, useEffect, useRef, type ReactNode } from 'react';
import type { Node } from '@xyflow/react';

import { LeftPanelTabs } from '../components/LeftPanelTabs';
import { CreateProjectDialog, OpenProjectDialog, SaveProjectDialog } from '../components/ProjectDialog';
import { CodeView } from '../components/CodeView';
import { EditorContainer } from './components/EditorContainer';
import { useEditorFactory } from './hooks/useEditorFactory';
import { useGraphEditor } from './hooks/useGraphEditor';
import type { EditorSelection, ConnectionRequest } from '../editor/types';
import type { INodeEditor } from '../editor/INodeEditor';
import type { OperationDef, FunctionDef } from '../types';
import { useProjectStore } from '../stores/projectStore';
import { useDialectStore } from '../stores/dialectStore';
import { useEditorStore } from '../core/stores/editorStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { useRendererStore } from '../stores/rendererStore';
import { handlePinnedTypeChange, type TypeChangeHandlerDeps } from '../services/typeChangeHandler';

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
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  
  // 左侧面板宽度（可拖拽调整）
  const [leftPanelWidth, setLeftPanelWidth] = useState(256);
  const isResizing = useRef(false);

  // Project dialog states
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isOpenDialogOpen, setIsOpenDialogOpen] = useState(false);
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);

  // Renderer state from store
  const renderer = useRendererStore(state => state.currentRenderer);
  const canvasBackend = useRendererStore(state => state.canvasBackend);
  const textRenderMode = useRendererStore(state => state.textRenderMode);
  const edgeRenderMode = useRendererStore(state => state.edgeRenderMode);
  const viewMode = useRendererStore(state => state.viewMode);
  const setMlirCode = useRendererStore(state => state.setMlirCode);
  const addLog = useRendererStore(state => state.addLog);
  const setProcessing = useRendererStore(state => state.setProcessing);
  const clearLogs = useRendererStore(state => state.clearLogs);
  
  // 编辑器实例引用
  const editorRef = useRef<INodeEditor | null>(null);
  
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
  
  // 从 editorStore 获取选择状态
  const selection = useEditorStore(state => state.selection);
  
  // 编辑器就绪回调
  const handleEditorReady = useCallback((editor: INodeEditor) => {
    editorRef.current = editor;
    // 如果是 GPU 后端（WebGL 或 WebGPU），同步当前渲染模式
    if (renderer === 'canvas' && (canvasBackend === 'webgl' || canvasBackend === 'webgpu')) {
      if (typeof (editor as unknown as { setTextRenderMode?: (m: string) => void }).setTextRenderMode === 'function') {
        (editor as unknown as { setTextRenderMode: (m: string) => void }).setTextRenderMode(textRenderMode);
      }
      if (typeof (editor as unknown as { setEdgeRenderMode?: (m: string) => void }).setEdgeRenderMode === 'function') {
        (editor as unknown as { setEdgeRenderMode: (m: string) => void }).setEdgeRenderMode(edgeRenderMode);
      }
    }
  }, [renderer, canvasBackend, textRenderMode, edgeRenderMode]);
  
  // 监听渲染模式变化，同步到编辑器实例
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    
    // 仅 GPU 后端需要同步
    if (renderer !== 'canvas' || (canvasBackend !== 'webgl' && canvasBackend !== 'webgpu')) {
      return;
    }
    
    // 同步文字渲染模式
    if (typeof (editor as unknown as { setTextRenderMode?: (m: string) => void }).setTextRenderMode === 'function') {
      (editor as unknown as { setTextRenderMode: (m: string) => void }).setTextRenderMode(textRenderMode);
    }
    
    // 同步边渲染模式
    if (typeof (editor as unknown as { setEdgeRenderMode?: (m: string) => void }).setEdgeRenderMode === 'function') {
      (editor as unknown as { setEdgeRenderMode: (m: string) => void }).setEdgeRenderMode(edgeRenderMode);
    }
  }, [renderer, canvasBackend, textRenderMode, edgeRenderMode]);
  
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

  // 左侧面板拖拽调整宽度
  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing.current) return;
      const newWidth = Math.max(200, Math.min(500, e.clientX));
      setLeftPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      if (isResizing.current) {
        isResizing.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

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

  // 类型选择处理
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);
  const getConstraintElements = useTypeConstraintStore(state => state.getConstraintElements);
  const pickConstraintName = useTypeConstraintStore(state => state.pickConstraintName);
  
  const handleTypeSelect = useCallback((nodeId: string, handleId: string, type: string) => {
    const state = useEditorStore.getState();
    const node = state.nodes.find(n => n.id === nodeId);
    if (!node) return;
    
    // 获取原始约束
    let originalConstraint: string | undefined;
    const data = node.data as Record<string, unknown>;
    
    if (node.type === 'operation') {
      const operation = data.operation as { arguments: Array<{ kind: string; name: string; typeConstraint?: string }>; results: Array<{ name: string; typeConstraint?: string }> };
      const isOutput = handleId.startsWith('data-out-');
      const portName = handleId.replace(/^data-(in|out)-/, '');
      
      if (isOutput) {
        const result = operation.results.find(r => r.name === portName);
        originalConstraint = result?.typeConstraint;
      } else {
        const operand = operation.arguments.find(a => a.kind === 'operand' && a.name === portName);
        originalConstraint = operand?.typeConstraint;
      }
    } else if (node.type === 'function-entry') {
      const outputs = data.outputs as Array<{ name: string; typeConstraint?: string }> | undefined;
      const portName = handleId.replace('data-out-', '');
      const param = outputs?.find(o => o.name === portName);
      originalConstraint = param?.typeConstraint;
    } else if (node.type === 'function-return') {
      const inputs = data.inputs as Array<{ name: string; typeConstraint?: string }> | undefined;
      const portName = handleId.replace('data-in-', '');
      const ret = inputs?.find(i => i.name === portName);
      originalConstraint = ret?.typeConstraint;
    } else if (node.type === 'function-call') {
      const isOutput = handleId.startsWith('data-out-');
      const portName = handleId.replace(/^data-(in|out)-/, '');
      
      if (isOutput) {
        const outputs = data.outputs as Array<{ name: string; typeConstraint?: string }> | undefined;
        const output = outputs?.find(o => o.name === portName);
        originalConstraint = output?.typeConstraint;
      } else {
        const inputs = data.inputs as Array<{ name: string; typeConstraint?: string }> | undefined;
        const input = inputs?.find(i => i.name === portName);
        originalConstraint = input?.typeConstraint;
      }
    }
    
    // 构建依赖项
    const deps: TypeChangeHandlerDeps = {
      edges: state.edges,
      getCurrentFunction,
      getConstraintElements,
      pickConstraintName,
      findSubsetConstraints: useTypeConstraintStore.getState().findSubsetConstraints,
    };
    
    // 处理类型变更
    const updatedNodes = handlePinnedTypeChange(
      nodeId,
      handleId,
      type,
      originalConstraint,
      state.nodes,
      deps
    );
    
    // 更新 store
    useEditorStore.getState().setNodes(updatedNodes);
  }, [getCurrentFunction, getConstraintElements, pickConstraintName]);

  // 参数重命名处理（Canvas 渲染器）
  const updateParameter = useProjectStore(state => state.updateParameter);
  const handleParameterRename = useCallback((functionId: string, oldName: string, newName: string) => {
    if (oldName === newName) return;
    
    // 获取当前参数信息
    const project = useProjectStore.getState().project;
    if (!project) return;
    
    const func = project.mainFunction.id === functionId
      ? project.mainFunction
      : project.customFunctions.find(f => f.id === functionId);
    
    if (!func) return;
    
    const param = func.parameters.find(p => p.name === oldName);
    if (!param) return;
    
    // 更新参数名称
    updateParameter(functionId, oldName, { ...param, name: newName });
  }, [updateParameter]);

  // 返回值重命名处理（Canvas 渲染器）
  const updateReturnType = useProjectStore(state => state.updateReturnType);
  const handleReturnTypeRename = useCallback((functionId: string, oldName: string, newName: string) => {
    if (oldName === newName) return;
    
    // 获取当前返回值信息
    const project = useProjectStore.getState().project;
    if (!project) return;
    
    const func = project.mainFunction.id === functionId
      ? project.mainFunction
      : project.customFunctions.find(f => f.id === functionId);
    
    if (!func) return;
    
    const returnType = func.returnTypes.find(r => r.name === oldName);
    if (!returnType) return;
    
    // 更新返回值名称
    updateReturnType(functionId, oldName, { ...returnType, name: newName });
  }, [updateReturnType]);

  const dismissError = useCallback(() => {
    setConnectionError(null);
  }, []);

  // API base URL
  const API_BASE_URL = '/api';

  // 切换到 Code 视图时：保存 + 预览
  const handleSwitchToCode = useCallback(async () => {
    if (!project?.path) return;
    
    setProcessing(true);
    clearLogs();
    addLog('info', 'Saving current graph...');
    
    try {
      // 1. 保存当前图到 projectStore
      if (currentFunctionId) {
        saveCurrentGraph(currentFunctionId);
      }
      
      // 2. 保存项目到磁盘
      addLog('info', 'Saving project to disk...');
      const saveProjectToPath = useProjectStore.getState().saveProjectToPath;
      const saved = await saveProjectToPath(project.path);
      if (!saved) {
        addLog('error', 'Failed to save project to disk');
        setMlirCode('', false);
        setProcessing(false);
        return;
      }
      
      // 3. 调用 Preview API 生成 MLIR
      addLog('info', 'Generating MLIR code...');
      const response = await fetch(`${API_BASE_URL}/build/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectPath: project.path }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setMlirCode(data.mlirCode || '', data.verified || false);
        addLog('success', `MLIR generated (${data.verified ? 'verified' : 'unverified'})`);
      } else {
        setMlirCode('', false);
        addLog('error', data.error || 'Preview failed');
      }
    } catch (error) {
      setMlirCode('', false);
      addLog('error', `Preview failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setProcessing(false);
    }
  }, [project?.path, currentFunctionId, saveCurrentGraph, setProcessing, clearLogs, addLog, setMlirCode]);

  // Run - JIT 执行
  const handleRun = useCallback(async () => {
    if (!project?.path) return;
    
    setProcessing(true);
    addLog('info', 'Saving and executing with JIT...');
    
    try {
      // 保存当前图
      if (currentFunctionId) {
        saveCurrentGraph(currentFunctionId);
      }
      
      // 保存项目到磁盘
      const saveProjectToPath = useProjectStore.getState().saveProjectToPath;
      await saveProjectToPath(project.path);
      
      // 执行
      const response = await fetch(`${API_BASE_URL}/build/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectPath: project.path }),
      });
      
      const data = await response.json();
      
      if (data.mlirCode) {
        setMlirCode(data.mlirCode, data.verified || false);
      }
      
      if (data.success) {
        addLog('success', 'Execution successful');
        if (data.output) {
          data.output.split('\n').forEach((line: string) => {
            if (line.trim()) addLog('output', line);
          });
        }
      } else {
        addLog('error', data.error || 'Execution failed');
        if (data.output) {
          data.output.split('\n').forEach((line: string) => {
            if (line.trim()) addLog('output', line);
          });
        }
      }
    } catch (error) {
      addLog('error', `Execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setProcessing(false);
    }
  }, [project?.path, currentFunctionId, saveCurrentGraph, setProcessing, addLog, setMlirCode]);

  // Build - 构建项目
  const handleBuild = useCallback(async () => {
    if (!project?.path) return;
    
    setProcessing(true);
    addLog('info', 'Building project...');
    
    try {
      // 保存当前图
      if (currentFunctionId) {
        saveCurrentGraph(currentFunctionId);
      }
      
      // 保存项目到磁盘
      const saveProjectToPath = useProjectStore.getState().saveProjectToPath;
      await saveProjectToPath(project.path);
      
      // 构建
      const response = await fetch(`${API_BASE_URL}/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectPath: project.path,
          generateLlvm: true,
          generateExecutable: true,
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        addLog('success', `MLIR: ${data.mlirPath}`);
        if (data.llvmPath) {
          addLog('success', `LLVM IR: ${data.llvmPath}`);
        }
        if (data.binPath) {
          addLog('success', `Executable: ${data.binPath}`);
        }
        if (!data.llvmPath && !data.binPath) {
          addLog('info', 'LLVM tools not found, skipped IR/executable generation');
        }
        addLog('success', 'Build completed');
      } else {
        addLog('error', data.error || 'Build failed');
      }
    } catch (error) {
      addLog('error', `Build failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setProcessing(false);
    }
  }, [project?.path, currentFunctionId, saveCurrentGraph, setProcessing, addLog]);

  return (
    <div className="w-full h-screen flex flex-col bg-gray-900">
      {/* Project Toolbar */}
      <ProjectToolbar
        project={project}
        onCreateClick={() => setIsCreateDialogOpen(true)}
        onOpenClick={() => setIsOpenDialogOpen(true)}
        onSaveClick={() => setIsSaveDialogOpen(true)}
        onSwitchToCode={handleSwitchToCode}
      />

      {/* Optional Header */}
      {header && (
        <div className="flex-shrink-0 border-b border-gray-700">
          {header}
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Node Palette */}
        <div 
          className="bg-gray-800 border-r border-gray-700 flex flex-col flex-shrink-0 relative"
          style={{ width: leftPanelWidth }}
        >
          <LeftPanelTabs
            onDragStart={handleDragStart}
            onFunctionDragStart={handleFunctionDragStart}
            onFunctionSelect={handleFunctionSelect}
            onFunctionDeleted={handleFunctionDeleted}
          />
          {/* 拖拽调整宽度手柄 */}
          <div
            className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-blue-500/50 transition-colors"
            onMouseDown={handleResizeMouseDown}
          />
        </div>

        {/* Center - Editor Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 relative">
            {/* Graph View */}
            {viewMode === 'graph' && (
              <EditorContainer
                rendererType={renderer}
                canvasBackend={canvasBackend}
                createEditor={createEditor}
                onConnect={handleConnect}
                onDrop={handleDrop}
                onEdgeDoubleClick={handleEdgeDoubleClick}
                onDeleteRequest={handleDeleteRequest}
                onSelectionChange={handleSelectionChange}
                onViewportChange={setViewport}
                onTypeSelect={handleTypeSelect}
                onEditorReady={handleEditorReady}
                onParameterRename={handleParameterRename}
                onReturnTypeRename={handleReturnTypeRename}
              />
            )}
            
            {/* Code View */}
            {viewMode === 'code' && (
              <CodeView
                onRunClick={handleRun}
                onBuildClick={handleBuild}
              />
            )}

            {connectionError && viewMode === 'graph' && (
              <ConnectionErrorToast
                message={connectionError}
                onClose={dismissError}
              />
            )}
          </div>
        </div>

        {/* Right Panel - Properties (only in graph mode) */}
        {viewMode === 'graph' && (effectiveSelectedNode || selection.nodeIds.length > 1) && (
          <div className="w-72 bg-gray-800 border-l border-gray-700 flex-shrink-0 overflow-hidden">
            <PropertiesPanel 
              selectedNode={effectiveSelectedNode} 
              selectedCount={selection.nodeIds.length}
            />
          </div>
        )}
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
