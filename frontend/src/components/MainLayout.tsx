/**
 * MainLayout 组件
 * 
 * 主应用布局：
 * - 左侧：节点面板（MLIR 操作浏览/搜索）
 * - 中央：节点编辑画布（React Flow）
 * - 右侧：属性面板
 * - 底部：执行面板
 */

import { useCallback, useRef, useState, useEffect, type ReactNode } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  SelectionMode,
  type Node,
  type Edge,
  type Connection,
  type OnConnect,
  type NodeChange as RFNodeChange,
  type EdgeChange as RFEdgeChange,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { NodePalette } from './NodePalette';
import { FunctionManager } from './FunctionManager';
import { ExecutionPanel } from './ExecutionPanel';
import { CreateProjectDialog, OpenProjectDialog, SaveProjectDialog } from './ProjectDialog';
import { CanvasEditorWrapper } from '../editor';
import type { EditorNode, NodeChange as EditorNodeChange, EditorSelection, ConnectionRequest } from '../editor';
import { nodeTypes } from './nodeTypes';
import { edgeTypes } from './edgeTypes';
import type { OperationDef, BlueprintNodeData, FunctionDef, GraphState, FunctionEntryData, FunctionReturnData, DataPin } from '../types';
import { validateConnection, type ConnectionValidationResult } from '../services/connectionValidator';
import { useProjectStore } from '../stores/projectStore';
import { useDialectStore } from '../stores/dialectStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { triggerTypePropagationWithSignature } from '../services/typePropagation';
import { dataInHandle, dataOutHandle } from '../services/port';
import { getDisplayType } from '../services/typeSelectorRenderer';

// Layout components
import { ConnectionErrorToast, PropertiesPanel, ProjectToolbar } from './layout';

// Utility functions
import {
  generateNodeId,
  generateEdgeId,
  edgesEqual,
  convertGraphEdgeToReactFlowEdge,
  convertGraphNodeToReactFlowNode,
  createBlueprintNodeData,
  getEdgeColor,
  updateEdgeColors,
} from '../utils';

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
 * Clipboard data structure for copy/paste operations
 */
interface ClipboardData {
  nodes: Node[];
  edges: Edge[];
}

/**
 * MainLayout Inner Component
 * 
 * Contains the actual layout implementation.
 * Must be wrapped in ReactFlowProvider to use useReactFlow hook.
 */
function MainLayoutInner({ header, footer }: MainLayoutProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [executionPanelExpanded, setExecutionPanelExpanded] = useState(true);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Clipboard for copy/paste operations
  const clipboardRef = useRef<ClipboardData | null>(null);

  // Project dialog states
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isOpenDialogOpen, setIsOpenDialogOpen] = useState(false);
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);

  // Canvas preview state
  const [showCanvasPreview, setShowCanvasPreview] = useState(false);

  // Get project state and actions
  const project = useProjectStore(state => state.project);
  const currentFunctionId = useProjectStore(state => state.currentFunctionId);
  const createProject = useProjectStore(state => state.createProject);
  const updateFunctionGraph = useProjectStore(state => state.updateFunctionGraph);
  const updateSignatureConstraints = useProjectStore(state => state.updateSignatureConstraints);

  // Get the current function directly from the store to react to changes
  const currentFunction = useProjectStore(state => {
    if (!state.project || !state.currentFunctionId) return null;
    if (state.project.mainFunction.id === state.currentFunctionId) {
      return state.project.mainFunction;
    }
    return state.project.customFunctions.find(f => f.id === state.currentFunctionId) || null;
  });

  // Initialize dialect store on mount
  const initializeDialects = useDialectStore(state => state.initialize);
  useEffect(() => {
    initializeDialects();
  }, [initializeDialects]);

  // Get type constraint store methods for narrowing calculation
  const getConstraintElements = useTypeConstraintStore(state => state.getConstraintElements);
  const pickConstraintName = useTypeConstraintStore(state => state.pickConstraintName);

  // Track if we're loading from store to avoid triggering auto-save
  const isLoadingFromStoreRef = useRef(false);

  // 异步同步签名：监听 Entry/Return 节点的类型变化，同步到 FunctionDef
  useEffect(() => {
    if (!currentFunctionId || !currentFunction || isLoadingFromStoreRef.current) return;
    if (currentFunction.isMain) return;

    const entryNode = nodes.find(n => n.type === 'function-entry' && n.id === 'entry');
    const returnNode = nodes.find(n => n.type === 'function-return');

    if (!entryNode && !returnNode) return;

    const parameterConstraints: Record<string, string> = {};
    const returnTypeConstraints: Record<string, string> = {};

    if (entryNode) {
      const entryData = entryNode.data as FunctionEntryData;
      const outputs = entryData.outputs || [];
      for (const port of outputs) {
        const dataPin: DataPin = {
          id: dataOutHandle(port.name),
          label: port.name,
          typeConstraint: port.typeConstraint,
          displayName: port.typeConstraint,
        };
        parameterConstraints[port.name] = getDisplayType(dataPin, entryData);
      }
    }

    if (returnNode) {
      const returnData = returnNode.data as FunctionReturnData;
      const inputs = returnData.inputs || [];
      for (const port of inputs) {
        const dataPin: DataPin = {
          id: dataInHandle(port.name),
          label: port.name,
          typeConstraint: port.typeConstraint,
          displayName: port.typeConstraint,
        };
        returnTypeConstraints[port.name] = getDisplayType(dataPin, returnData);
      }
    }

    updateSignatureConstraints(currentFunctionId, parameterConstraints, returnTypeConstraints);
  }, [nodes, currentFunctionId, currentFunction, updateSignatureConstraints]);

  // Convert nodes and edges to GraphState format for saving
  const convertToGraphState = useCallback((): GraphState => {
    return {
      nodes: nodes.map(n => ({
        id: n.id,
        type: (n.type || 'operation') as 'operation' | 'function-entry' | 'function-return' | 'function-call',
        position: { x: n.position.x, y: n.position.y },
        data: n.data as BlueprintNodeData,
      })),
      edges: edges.map(e => ({
        source: e.source,
        sourceHandle: e.sourceHandle || '',
        target: e.target,
        targetHandle: e.targetHandle || '',
      })),
    };
  }, [nodes, edges]);

  const convertToGraphStateRef = useRef(convertToGraphState);
  useEffect(() => {
    convertToGraphStateRef.current = convertToGraphState;
  }, [convertToGraphState]);

  const getFunctionById = useProjectStore(state => state.getFunctionById);

  const saveCurrentGraph = useCallback((functionId: string) => {
    const graphState = convertToGraphStateRef.current();
    if (graphState.nodes.length > 0) {
      updateFunctionGraph(functionId, graphState);
    }
  }, [updateFunctionGraph]);

  const loadFunctionGraph = useCallback((functionId: string) => {
    const func = getFunctionById(functionId);
    if (!func) return;

    isLoadingFromStoreRef.current = true;
    
    const graphNodes = func.graph.nodes.map(convertGraphNodeToReactFlowNode);
    const graphEdges = func.graph.edges.map(convertGraphEdgeToReactFlowEdge);

    const result = triggerTypePropagationWithSignature(
      graphNodes, graphEdges, func, getConstraintElements, pickConstraintName
    );

    setNodes(result.nodes);
    const edgesWithColors = updateEdgeColors(result.nodes, graphEdges);
    setEdges(edgesWithColors);

    queueMicrotask(() => {
      isLoadingFromStoreRef.current = false;
    });
  }, [getFunctionById, getConstraintElements, pickConstraintName, setNodes, setEdges]);

  // Derive selected node validity
  const effectiveSelectedNode = selectedNode && nodes.find(n => n.id === selectedNode.id)
    ? selectedNode
    : null;

  // Handle node selection
  const handleNodesChange = useCallback((changes: RFNodeChange[]) => {
    onNodesChange(changes);

    for (const change of changes) {
      if (change.type === 'select' && change.selected) {
        const node = nodes.find(n => n.id === change.id);
        if (node) {
          setSelectedNode(node);
        }
      } else if (change.type === 'select' && !change.selected) {
        setSelectedNode(prev => prev?.id === change.id ? null : prev);
      }
    }
  }, [nodes, onNodesChange]);

  const handleEdgesChange = useCallback((changes: RFEdgeChange[]) => {
    onEdgesChange(changes);
  }, [onEdgesChange]);


  /**
   * Copy selected nodes and their internal edges to clipboard
   */
  const copySelectedNodes = useCallback(() => {
    const selectedNodes = nodes.filter(n => n.selected);
    if (selectedNodes.length === 0) return;

    const selectedNodeIds = new Set(selectedNodes.map(n => n.id));
    const internalEdges = edges.filter(
      e => selectedNodeIds.has(e.source) && selectedNodeIds.has(e.target)
    );

    clipboardRef.current = {
      nodes: selectedNodes.map(n => ({ ...n })),
      edges: internalEdges.map(e => ({ ...e })),
    };

    console.log(`Copied ${selectedNodes.length} nodes and ${internalEdges.length} edges`);
  }, [nodes, edges]);

  /**
   * Paste nodes from clipboard with new IDs and offset position
   */
  const pasteNodes = useCallback(() => {
    if (!clipboardRef.current || clipboardRef.current.nodes.length === 0) return;

    const { nodes: copiedNodes, edges: copiedEdges } = clipboardRef.current;
    const idMap = new Map<string, string>();
    const offset = { x: 50, y: 50 };

    const existingReturnNodes = nodes.filter(n => n.type === 'function-return');
    const getNextReturnIndex = (): number => {
      const indices = existingReturnNodes
        .map(n => {
          const match = n.id.match(/^return-(\d+)$/);
          return match ? parseInt(match[1], 10) : -1;
        })
        .filter(idx => idx >= 0);
      if (indices.length === 0) return 0;
      return Math.max(...indices) + 1;
    };

    let nextReturnIndex = getNextReturnIndex();

    const newNodes: Node[] = [];
    for (const node of copiedNodes) {
      if (node.type === 'function-entry') continue;

      let newId: string;
      if (node.type === 'function-return') {
        newId = `return-${nextReturnIndex}`;
        nextReturnIndex++;
      } else {
        newId = generateNodeId();
      }
      idMap.set(node.id, newId);

      newNodes.push({
        ...node,
        id: newId,
        position: {
          x: node.position.x + offset.x,
          y: node.position.y + offset.y,
        },
        selected: true,
        data: { ...node.data },
      });
    }

    if (newNodes.length === 0) return;

    const newEdges: Edge[] = copiedEdges
      .filter(edge => idMap.has(edge.source) && idMap.has(edge.target))
      .map(edge => ({
        ...edge,
        id: generateEdgeId({
          source: idMap.get(edge.source)!,
          sourceHandle: edge.sourceHandle!,
          target: idMap.get(edge.target)!,
          targetHandle: edge.targetHandle!,
        }),
        source: idMap.get(edge.source)!,
        target: idMap.get(edge.target)!,
        selected: false,
      }));

    setNodes(nds => [
      ...nds.map(n => ({ ...n, selected: false })),
      ...newNodes,
    ]);

    setEdges(eds => [...eds, ...newEdges]);

    console.log(`Pasted ${newNodes.length} nodes and ${newEdges.length} edges`);
  }, [nodes, setNodes, setEdges]);

  /**
   * Delete selected nodes and edges
   */
  const deleteSelected = useCallback(() => {
    const allReturnNodes = nodes.filter(n => n.type === 'function-return');
    const selectedReturnNodes = allReturnNodes.filter(n => n.selected);

    const maxReturnNodesToDelete = Math.max(0, allReturnNodes.length - 1);
    const returnNodesToDelete = selectedReturnNodes.slice(0, maxReturnNodesToDelete);
    const returnNodeIdsToDelete = new Set(returnNodesToDelete.map(n => n.id));

    const selectedNodeIds = new Set(
      nodes
        .filter(n => {
          if (!n.selected) return false;
          if (n.type === 'function-entry') return false;
          if (n.type === 'function-return') return returnNodeIdsToDelete.has(n.id);
          return true;
        })
        .map(n => n.id)
    );

    const selectedEdges = edges.filter(e => e.selected);

    if (selectedNodeIds.size === 0 && selectedEdges.length === 0) return;

    if (selectedNodeIds.size > 0) {
      setNodes(nds => nds.filter(n => !selectedNodeIds.has(n.id)));
      setEdges(eds => eds.filter(e =>
        !selectedNodeIds.has(e.source) && !selectedNodeIds.has(e.target)
      ));
    }

    if (selectedEdges.length > 0) {
      setEdges(eds => eds.filter(e => 
        !selectedEdges.some(se => edgesEqual(e, se))
      ));
    }

    const skippedReturnNodes = selectedReturnNodes.length - returnNodesToDelete.length;
    if (skippedReturnNodes > 0) {
      console.log(`Kept ${skippedReturnNodes} Return node(s) to ensure at least one remains`);
    }
    console.log(`Deleted ${selectedNodeIds.size} nodes and ${selectedEdges.length} edges`);
  }, [nodes, edges, setNodes, setEdges]);

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

  /**
   * Handle double-click on edge to delete it
   */
  const handleEdgeDoubleClick = useCallback((_event: React.MouseEvent, edge: Edge) => {
    const remainingEdges = edges.filter(e => e.id !== edge.id);
    setEdges(remainingEdges);

    if (!edge.sourceHandle?.startsWith('exec-')) {
      setNodes(nds => {
        const result = triggerTypePropagationWithSignature(
          nds, remainingEdges, currentFunction ?? undefined, getConstraintElements, pickConstraintName
        );
        const updatedEdges = updateEdgeColors(result.nodes, remainingEdges);
        setEdges(updatedEdges);
        return result.nodes;
      });
    }
  }, [edges, setEdges, setNodes, currentFunction, getConstraintElements, pickConstraintName]);

  // Handle function selection from FunctionManager
  const handleFunctionSelect = useCallback((functionId: string) => {
    if (currentFunctionId && currentFunctionId !== functionId) {
      saveCurrentGraph(currentFunctionId);
    }
    loadFunctionGraph(functionId);
  }, [currentFunctionId, saveCurrentGraph, loadFunctionGraph]);

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

  const handleFunctionDeleted = useCallback(() => {
    const currentId = useProjectStore.getState().currentFunctionId;
    if (currentId) {
      loadFunctionGraph(currentId);
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

  const reactFlowInstance = useReactFlow();


  // Handle drop on canvas
  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const functionData = event.dataTransfer.getData('application/reactflow-function');
      if (functionData) {
        try {
          const functionCallData = JSON.parse(functionData);
          const newNode: Node = {
            id: generateNodeId(),
            type: 'function-call',
            position,
            data: functionCallData,
          };
          setNodes((nds) => [...nds, newNode]);
          return;
        } catch (err) {
          console.error('Failed to parse dropped function:', err);
        }
      }

      const data = event.dataTransfer.getData('application/json');
      if (!data) return;

      try {
        const operation: OperationDef = JSON.parse(data);
        const newNode: Node = {
          id: generateNodeId(),
          type: 'operation',
          position,
          data: createBlueprintNodeData(operation),
        };
        setNodes((nds) => [...nds, newNode]);
      } catch (err) {
        console.error('Failed to parse dropped operation:', err);
      }
    },
    [setNodes, reactFlowInstance]
  );

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  }, []);

  /**
   * Validates a connection before it's created
   */
  const isValidConnection = useCallback(
    (connection: Edge | Connection): boolean => {
      const normalizedConnection: Connection = {
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle ?? null,
        targetHandle: connection.targetHandle ?? null,
      };
      const existingEdges = edges.map(e => ({
        source: e.source,
        sourceHandle: e.sourceHandle ?? null,
        target: e.target,
        targetHandle: e.targetHandle ?? null,
      }));
      const result = validateConnection(normalizedConnection, nodes, undefined, existingEdges);
      return result.isValid;
    },
    [nodes, edges]
  );

  const isExecHandle = useCallback((handleId: string | null | undefined): boolean => {
    if (!handleId) return false;
    return handleId.startsWith('exec-');
  }, []);

  /**
   * Handles connection creation with validation
   */
  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      const existingEdges = edges.map(e => ({
        source: e.source,
        sourceHandle: e.sourceHandle ?? null,
        target: e.target,
        targetHandle: e.targetHandle ?? null,
      }));

      const validationResult: ConnectionValidationResult = validateConnection(
        connection,
        nodes,
        undefined,
        existingEdges
      );

      if (!validationResult.isValid) {
        setConnectionError(validationResult.errorMessage || 'Invalid connection');
        setTimeout(() => setConnectionError(null), 3000);
        return;
      }

      const isExec = isExecHandle(connection.sourceHandle) || isExecHandle(connection.targetHandle);

      const edgeWithStyle: Edge = {
        ...connection,
        id: generateEdgeId({
          source: connection.source!,
          sourceHandle: connection.sourceHandle!,
          target: connection.target!,
          targetHandle: connection.targetHandle!,
        }),
        type: isExec ? 'execution' : 'data',
        data: isExec ? undefined : { color: getEdgeColor(nodes, connection.source!, connection.sourceHandle) },
      };

      const newEdges = [...edges, edgeWithStyle];
      setEdges(newEdges);

      if (!isExec) {
        setNodes(nds => {
          const result = triggerTypePropagationWithSignature(
            nds, newEdges, currentFunction ?? undefined, getConstraintElements, pickConstraintName
          );
          const updatedEdges = updateEdgeColors(result.nodes, newEdges);
          setEdges(updatedEdges);
          return result.nodes;
        });
      }
    },
    [nodes, edges, setEdges, setNodes, isExecHandle, currentFunction, getConstraintElements, pickConstraintName]
  );

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
        showCanvasPreview={showCanvasPreview}
        onShowCanvasPreviewChange={setShowCanvasPreview}
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
          <div className="flex-1 relative" ref={reactFlowWrapper}>
            {showCanvasPreview ? (
              <CanvasEditorWrapper 
                nodes={nodes.map(n => ({
                  id: n.id,
                  type: n.type as EditorNode['type'],
                  position: n.position,
                  data: n.data,
                  selected: n.selected,
                }))}
                edges={edges.map(e => ({
                  id: e.id,
                  source: e.source,
                  sourceHandle: e.sourceHandle ?? '',
                  target: e.target,
                  targetHandle: e.targetHandle ?? '',
                  selected: e.selected,
                  type: e.type as 'execution' | 'data' | undefined,
                  data: e.data,
                }))}
                onNodesChange={(changes: EditorNodeChange[]) => {
                  for (const change of changes) {
                    if (change.type === 'position') {
                      setNodes(nds => nds.map(n => 
                        n.id === change.id ? { ...n, position: change.position } : n
                      ));
                    }
                  }
                }}
                onSelectionChange={(selection: EditorSelection) => {
                  setNodes(nds => nds.map(n => ({
                    ...n,
                    selected: selection.nodeIds.includes(n.id),
                  })));
                  if (selection.nodeIds.length === 1) {
                    const node = nodes.find(n => n.id === selection.nodeIds[0]);
                    if (node) setSelectedNode(node);
                  } else {
                    setSelectedNode(null);
                  }
                }}
                onConnect={(request: ConnectionRequest) => {
                  const connection: Connection = {
                    source: request.source,
                    sourceHandle: request.sourceHandle,
                    target: request.target,
                    targetHandle: request.targetHandle,
                  };
                  
                  const existingEdges = edges.map(e => ({
                    source: e.source,
                    sourceHandle: e.sourceHandle ?? null,
                    target: e.target,
                    targetHandle: e.targetHandle ?? null,
                  }));

                  const validationResult = validateConnection(
                    connection,
                    nodes,
                    undefined,
                    existingEdges
                  );

                  if (!validationResult.isValid) {
                    setConnectionError(validationResult.errorMessage || 'Invalid connection');
                    setTimeout(() => setConnectionError(null), 3000);
                    return;
                  }

                  const isExec = request.sourceHandle.startsWith('exec-') || request.targetHandle.startsWith('exec-');
                  const edgeWithStyle: Edge = {
                    ...connection,
                    id: generateEdgeId({
                      source: request.source,
                      sourceHandle: request.sourceHandle,
                      target: request.target,
                      targetHandle: request.targetHandle,
                    }),
                    type: isExec ? 'execution' : 'data',
                    data: isExec ? undefined : { color: getEdgeColor(nodes, request.source, request.sourceHandle) },
                  };

                  const newEdges = [...edges, edgeWithStyle];
                  setEdges(newEdges);

                  if (!isExec) {
                    setNodes(nds => {
                      const result = triggerTypePropagationWithSignature(
                        nds, newEdges, currentFunction ?? undefined, getConstraintElements, pickConstraintName
                      );
                      return result.nodes;
                    });
                  }
                }}
                onDeleteRequest={(nodeIds, edgeIds) => {
                  if (nodeIds.length > 0 || edgeIds.length > 0) {
                    deleteSelected();
                  }
                }}
                onEdgeDoubleClick={(edgeId) => {
                  const edge = edges.find(e => {
                    const computedId = `${e.source}-${e.sourceHandle}-${e.target}-${e.targetHandle}`;
                    return computedId === edgeId || e.id === edgeId;
                  });
                  if (edge) {
                    const remainingEdges = edges.filter(e => e.id !== edge.id);
                    setEdges(remainingEdges);

                    if (!edge.sourceHandle?.startsWith('exec-')) {
                      setNodes(nds => {
                        const result = triggerTypePropagationWithSignature(
                          nds, remainingEdges, currentFunction ?? undefined, getConstraintElements, pickConstraintName
                        );
                        return result.nodes;
                      });
                    }
                  }
                }}
                onDrop={(x, y, dataTransfer) => {
                  const functionData = dataTransfer.getData('application/reactflow-function');
                  if (functionData) {
                    try {
                      const functionCallData = JSON.parse(functionData);
                      const newNode: Node = {
                        id: generateNodeId(),
                        type: 'function-call',
                        position: { x, y },
                        data: functionCallData,
                      };
                      setNodes((nds) => [...nds, newNode]);
                      return;
                    } catch (err) {
                      console.error('Failed to parse dropped function:', err);
                    }
                  }

                  const data = dataTransfer.getData('application/json');
                  if (!data) return;

                  try {
                    const operation: OperationDef = JSON.parse(data);
                    const newNode: Node = {
                      id: generateNodeId(),
                      type: 'operation',
                      position: { x, y },
                      data: createBlueprintNodeData(operation),
                    };
                    setNodes((nds) => [...nds, newNode]);
                  } catch (err) {
                    console.error('Failed to parse dropped operation:', err);
                  }
                }}
              />
            ) : (
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={handleNodesChange}
                onEdgesChange={handleEdgesChange}
                onConnect={onConnect}
                isValidConnection={isValidConnection}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onEdgeDoubleClick={handleEdgeDoubleClick}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                defaultEdgeOptions={{
                  type: 'data',
                }}
                selectionOnDrag
                panOnDrag={[2]}
                selectionMode={SelectionMode.Partial}
                selectNodesOnDrag={false}
                edgesReconnectable
                fitView
                fitViewOptions={{ maxZoom: 1 }}
                minZoom={0.1}
                maxZoom={2}
                defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
                colorMode="dark"
              >
                <Background color="#444" gap={16} />
                <Controls />
                <MiniMap
                  nodeColor={(node) => {
                    switch (node.type) {
                      case 'function-entry': return '#22c55e';
                      case 'function-return': return '#ef4444';
                      case 'function-call': return '#a855f7';
                      default: return '#3b82f6';
                    }
                  }}
                />
              </ReactFlow>
            )}

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

/**
 * MainLayout Component
 * 
 * Provides the main application layout with:
 * - Left: Node palette (operations grouped by dialect)
 * - Center: Node editor (React Flow canvas)
 * - Right: Properties panel
 * - Bottom: Execution panel
 * 
 * Wrapped in ReactFlowProvider to enable useReactFlow hook.
 */
export function MainLayout(props: MainLayoutProps) {
  return (
    <ReactFlowProvider>
      <MainLayoutInner {...props} />
    </ReactFlowProvider>
  );
}

export default MainLayout;
