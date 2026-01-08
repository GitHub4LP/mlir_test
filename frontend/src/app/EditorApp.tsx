/**
 * VS Code 编辑器入口
 * 
 * 包含节点画布和属性面板，用于 VS Code Webview。
 * 使用 VSCodeBridge 与扩展通讯。
 */

import { StrictMode, useEffect, useState, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import type { Node } from '@xyflow/react';
import { getPlatformBridge } from '../platform';
import type { VSCodeBridge } from '../platform/vscode';
import { EditorContainer } from './components/EditorContainer';
import { useEditorFactory } from './hooks/useEditorFactory';
import { useGraphEditor } from './hooks/useGraphEditor';
import { useEditorStore } from '../core/stores/editorStore';
import { useDialectStore } from '../stores/dialectStore';
import { useProjectStore } from '../stores/projectStore';
import { useRendererStore } from '../stores/rendererStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { PropertiesPanel } from '../components/layout';
import type { EditorSelection, ConnectionRequest } from '../editor/types';

import '@xyflow/react/dist/style.css';
import '../index.css';

/** 编辑器应用 */
function EditorApp() {
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const bridge = getPlatformBridge();
  
  // Renderer state
  const renderer = useRendererStore(state => state.currentRenderer);
  const canvasBackend = useRendererStore(state => state.canvasBackend);
  
  // Editor factory
  const { createEditor } = useEditorFactory();
  
  // Graph editor hook
  const graphEditor = useGraphEditor();
  const {
    nodes,
    handleConnect: graphHandleConnect,
    handleDrop: graphHandleDrop,
    handleEdgeDoubleClick: graphHandleEdgeDoubleClick,
    deleteSelected,
    loadFunctionGraph,
    setViewport,
  } = graphEditor;
  
  // Selection state from store
  const selection = useEditorStore(state => state.selection);
  
  // Initialize dialects
  const initializeDialects = useDialectStore(state => state.initialize);
  useEffect(() => {
    initializeDialects();
  }, [initializeDialects]);
  
  // Initialize type constraints
  const loadTypeConstraints = useTypeConstraintStore(state => state.loadTypeConstraints);
  useEffect(() => {
    loadTypeConstraints();
  }, [loadTypeConstraints]);
  
  // Initialize default project
  const project = useProjectStore(state => state.project);
  const createProjectAction = useProjectStore(state => state.createProject);
  const loadProjectFromPath = useProjectStore(state => state.loadProjectFromPath);
  const saveProjectToPath = useProjectStore(state => state.saveProjectToPath);
  const currentFunctionName = useProjectStore(state => state.currentFunctionName);
  const addFunction = useProjectStore(state => state.addFunction);
  const renameFunction = useProjectStore(state => state.renameFunction);
  const removeFunction = useProjectStore(state => state.removeFunction);
  
  useEffect(() => {
    if (!project) {
      createProjectAction('Untitled Project', './untitled_project');
    }
  }, [project, createProjectAction]);
  
  // Load function graph when project is ready
  useEffect(() => {
    if (currentFunctionName) {
      loadFunctionGraph(currentFunctionName);
    }
  }, [currentFunctionName, loadFunctionGraph]);
  
  // Handle MLIR preview request
  const handleMlirPreview = useCallback(async () => {
    if (!project) return;
    try {
      const response = await bridge.callApi<{ mlir: string; verified: boolean }>('/build/preview', {
        method: 'POST',
        body: { project },
      });
      await bridge.showMlirCode(response.mlir, response.verified);
    } catch (error) {
      bridge.showNotification(`MLIR 预览失败: ${error}`, 'error');
    }
  }, [project, bridge]);
  
  // Handle save project request
  const handleSaveProject = useCallback(async () => {
    if (!project) return;
    try {
      await saveProjectToPath();
      bridge.showNotification('项目已保存', 'info');
    } catch (error) {
      bridge.showNotification(`保存失败: ${error}`, 'error');
    }
  }, [project, saveProjectToPath, bridge]);
  
  // Handle create project from extension
  const handleCreateProject = useCallback(async (path: string) => {
    const name = path.split(/[/\\]/).pop() || 'New Project';
    createProjectAction(name, path);
    bridge.showNotification(`项目已创建: ${name}`, 'info');
  }, [createProjectAction, bridge]);
  
  // Handle open project from extension
  const handleOpenProject = useCallback(async (path: string) => {
    try {
      bridge.appendOutput(`Loading project from: ${path}`, 'info');
      const success = await loadProjectFromPath(path);
      if (success) {
        bridge.showNotification('项目已加载', 'info');
        // 手动触发图加载，因为 useEffect 可能不会立即响应
        const newFunctionName = useProjectStore.getState().currentFunctionName;
        if (newFunctionName) {
          loadFunctionGraph(newFunctionName);
        }
      } else {
        const error = useProjectStore.getState().error;
        bridge.showNotification(`加载失败: ${error || '未知错误'}`, 'error');
        bridge.appendOutput(`Load failed: ${error}`, 'error');
      }
    } catch (error) {
      bridge.showNotification(`加载失败: ${error}`, 'error');
      bridge.appendOutput(`Load error: ${error}`, 'error');
    }
  }, [loadProjectFromPath, loadFunctionGraph, bridge]);
  
  // Handle open project and switch to specific function
  const handleOpenProjectAndFunction = useCallback(async (path: string, functionName: string) => {
    try {
      bridge.appendOutput(`Loading project from: ${path}, function: ${functionName}`, 'info');
      const success = await loadProjectFromPath(path);
      if (success) {
        bridge.showNotification('项目已加载', 'info');
        // 切换到指定函数
        graphEditor.handleFunctionSelect(functionName);
      } else {
        const error = useProjectStore.getState().error;
        bridge.showNotification(`加载失败: ${error || '未知错误'}`, 'error');
        bridge.appendOutput(`Load failed: ${error}`, 'error');
      }
    } catch (error) {
      bridge.showNotification(`加载失败: ${error}`, 'error');
      bridge.appendOutput(`Load error: ${error}`, 'error');
    }
  }, [loadProjectFromPath, graphEditor, bridge]);
  
  // Handle file change notification from extension
  const handleFileChange = useCallback(async (changeType: 'created' | 'deleted', functionName: string) => {
    bridge.appendOutput(`File ${changeType}: ${functionName}`, 'info');
    
    // 重新加载项目以刷新函数列表
    const projectPath = useProjectStore.getState().project?.path;
    if (projectPath) {
      await loadProjectFromPath(projectPath);
    }
  }, [loadProjectFromPath, bridge]);
  
  // Handle file rename notification from extension
  const handleFileRename = useCallback(async (oldName: string, newName: string) => {
    bridge.appendOutput(`File renamed: ${oldName} -> ${newName}`, 'info');
    
    // 调用 renameFunction 更新内存中的状态和 Call 节点引用
    renameFunction(oldName, newName);
    
    // 重新加载项目以确保同步
    const projectPath = useProjectStore.getState().project?.path;
    if (projectPath) {
      await loadProjectFromPath(projectPath);
    }
  }, [renameFunction, loadProjectFromPath, bridge]);
  
  // Handle add function from extension
  const handleAddFunction = useCallback((name: string) => {
    const newFunc = addFunction(name);
    if (newFunc) {
      // 切换到新函数
      graphEditor.handleFunctionSelect(newFunc.name);
      bridge.showNotification(`函数 "${name}" 已创建`, 'info');
    }
  }, [addFunction, graphEditor, bridge]);
  
  // Handle rename function from extension
  const handleRenameFunction = useCallback((functionName: string, newName: string) => {
    renameFunction(functionName, newName);
    bridge.showNotification(`函数已重命名为 "${newName}"`, 'info');
  }, [renameFunction, bridge]);
  
  // Handle delete function from extension
  const handleDeleteFunction = useCallback((functionName: string) => {
    if (removeFunction(functionName)) {
      bridge.showNotification('函数已删除', 'info');
    }
  }, [removeFunction, bridge]);
  
  // Handle select function from extension
  const handleSelectFunction = useCallback((functionName: string) => {
    graphEditor.handleFunctionSelect(functionName);
  }, [graphEditor]);
  
  // Get function names from store
  const functionNames = useProjectStore(state => state.functionNames);
  const loadedFunctions = useProjectStore(state => state.loadedFunctions);
  
  // Sync functions to VS Code TreeView
  const syncFunctionsToTreeView = useCallback(() => {
    if (!project) return;
    const vscodeBridge = bridge as VSCodeBridge;
    if (vscodeBridge.updateFunctions) {
      // 使用 functionNames 而不是 customFunctions，因为懒加载模式下 customFunctions 可能为空
      const functions = functionNames.map(name => {
        const func = loadedFunctions.get(name);
        if (func) {
          return {
            name: func.name,
            parameters: func.parameters,
            returnTypes: func.returnTypes,
          };
        }
        // 函数还没加载，只返回名称
        return {
          name,
          parameters: [],
          returnTypes: [],
        };
      });
      vscodeBridge.updateFunctions(functions, currentFunctionName);
    }
  }, [project, functionNames, loadedFunctions, currentFunctionName, bridge]);
  
  // Sync functions when project or currentFunctionName changes
  useEffect(() => {
    syncFunctionsToTreeView();
  }, [syncFunctionsToTreeView]);
  
  // Notify extension when ready and subscribe to events
  useEffect(() => {
    bridge.appendOutput('Editor ready', 'info');
    
    const vscodeBridge = bridge as VSCodeBridge;
    const unsubscribes: Array<() => void> = [];
    
    // Listen for backend ready message
    if (vscodeBridge.onBackendReady) {
      unsubscribes.push(vscodeBridge.onBackendReady((_port, url) => {
        bridge.appendOutput(`Backend ready at ${url}`, 'info');
      }));
    }
    
    // Listen for extension commands
    if (vscodeBridge.onCreateProject) {
      unsubscribes.push(vscodeBridge.onCreateProject(handleCreateProject));
    }
    if (vscodeBridge.onOpenProject) {
      unsubscribes.push(vscodeBridge.onOpenProject(handleOpenProject));
    }
    if (vscodeBridge.onOpenProjectAndFunction) {
      unsubscribes.push(vscodeBridge.onOpenProjectAndFunction(handleOpenProjectAndFunction));
    }
    if (vscodeBridge.onFileChange) {
      unsubscribes.push(vscodeBridge.onFileChange(handleFileChange));
    }
    if (vscodeBridge.onFileRename) {
      unsubscribes.push(vscodeBridge.onFileRename(handleFileRename));
    }
    if (vscodeBridge.onMlirPreviewRequest) {
      unsubscribes.push(vscodeBridge.onMlirPreviewRequest(handleMlirPreview));
    }
    if (vscodeBridge.onSaveProjectRequest) {
      unsubscribes.push(vscodeBridge.onSaveProjectRequest(handleSaveProject));
    }
    if (vscodeBridge.onAddFunction) {
      unsubscribes.push(vscodeBridge.onAddFunction(handleAddFunction));
    }
    if (vscodeBridge.onRenameFunction) {
      unsubscribes.push(vscodeBridge.onRenameFunction(handleRenameFunction));
    }
    if (vscodeBridge.onDeleteFunction) {
      unsubscribes.push(vscodeBridge.onDeleteFunction(handleDeleteFunction));
    }
    if (vscodeBridge.onSelectFunction) {
      unsubscribes.push(vscodeBridge.onSelectFunction(handleSelectFunction));
    }
    
    return () => {
      unsubscribes.forEach(unsub => unsub());
    };
  }, [bridge, handleCreateProject, handleOpenProject, handleOpenProjectAndFunction, handleFileChange, handleFileRename, handleMlirPreview, handleSaveProject, handleAddFunction, handleRenameFunction, handleDeleteFunction, handleSelectFunction]);
  
  // Handle connection
  const handleConnect = (request: ConnectionRequest) => {
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
  };
  
  // Handle drop
  const handleDrop = (x: number, y: number, dataTransfer: DataTransfer) => {
    graphHandleDrop(x, y, dataTransfer);
  };
  
  // Handle edge double click
  const handleEdgeDoubleClick = (edgeId: string) => {
    graphHandleEdgeDoubleClick(edgeId);
  };
  
  // Handle delete request
  const handleDeleteRequest = (nodeIds: string[], edgeIds: string[]) => {
    if (nodeIds.length > 0 || edgeIds.length > 0) {
      deleteSelected();
    }
  };
  
  // Handle selection change
  const handleSelectionChange = (selection: EditorSelection) => {
    useEditorStore.getState().setSelection(selection);
    
    // Update selected node for properties panel
    if (selection.nodeIds.length === 1) {
      const node = nodes.find(n => n.id === selection.nodeIds[0]);
      if (node) {
        setSelectedNode(node as unknown as Node);
      }
    } else {
      setSelectedNode(null);
    }
    
    // Notify extension about selection change
    const allNodes = useEditorStore.getState().nodes;
    const selectedNodes = allNodes.filter(n => selection.nodeIds.includes(n.id));
    bridge.notifySelectionChanged(selection.nodeIds, selectedNodes);
  };
  
  // Derive selected node validity (ensure node still exists)
  const effectiveSelectedNode = selectedNode && nodes.find(n => n.id === selectedNode.id)
    ? selectedNode
    : null;
  
  // Note: selectedNode is cleared in handleSelectionChange when selection.nodeIds.length !== 1
  // No need for a separate useEffect to sync this state
  
  return (
    <div className="w-full h-screen bg-gray-900 flex">
      {/* Main Editor Area */}
      <div className="flex-1 relative">
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
        />
        
        {connectionError && (
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-red-600 text-white px-4 py-2 rounded shadow-lg">
            {connectionError}
          </div>
        )}
      </div>
      
      {/* Right Panel - Properties (when node selected) */}
      {(effectiveSelectedNode || selection.nodeIds.length > 1) && (
        <div className="w-72 bg-gray-800 border-l border-gray-700 flex-shrink-0 overflow-hidden">
          <PropertiesPanel 
            selectedNode={effectiveSelectedNode} 
            selectedCount={selection.nodeIds.length}
          />
        </div>
      )}
    </div>
  );
}

// Mount app
const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(
    <StrictMode>
      <EditorApp />
    </StrictMode>
  );
}

export default EditorApp;
