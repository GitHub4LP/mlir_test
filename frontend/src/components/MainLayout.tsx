/**
 * MainLayout Component
 * 
 * Implements the main application layout with:
 * - Left panel: Node palette for browsing/searching MLIR operations
 * - Center: Node editor canvas (React Flow)
 * - Right panel: Properties panel for editing selected node attributes
 * - Bottom: Execution panel for running MLIR code
 * 
 * Requirements: 2.1
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
  type NodeChange,
  type EdgeChange,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { NodePalette } from './NodePalette';
import { FunctionManager } from './FunctionManager';
import { ExecutionPanel } from './ExecutionPanel';
import { CreateProjectDialog, OpenProjectDialog, SaveProjectDialog } from './ProjectDialog';
import { nodeTypes } from './nodeTypes';
import { edgeTypes } from './edgeTypes';
import type { OperationDef, BlueprintNodeData, FunctionDef, GraphState, FunctionEntryData, FunctionCallData, Project } from '../types';
import { validateConnection, type ConnectionValidationResult } from '../services/connectionValidator';
import { getTypeColor } from '../services/typeSystem';
import { generateExecConfig, createExecIn } from '../services/operationClassifier';
import { useTypeStore } from '../stores/typeStore';
import { useProjectStore } from '../stores/projectStore';
import { useDialectStore } from '../stores/dialectStore';
import { buildPropagationGraph, propagateTypes, extractTypeSources, applyPropagationResult } from '../services/typePropagation/propagator';

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
 * Generates a unique node ID
 */
function generateNodeId(): string {
  return `node_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Creates BlueprintNodeData from an operation definition
 * 
 * Automatically generates execution pins based on operation classification:
 * - Terminator operations: exec-in only, no exec-out
 * - Control flow operations: exec-in + one exec-out per region
 * - Regular operations: exec-in + single exec-out
 */
function createBlueprintNodeData(operation: OperationDef): BlueprintNodeData {
  const attributes: Record<string, string> = {};
  const inputTypes: Record<string, string> = {};
  const outputTypes: Record<string, string> = {};
  
  for (const arg of operation.arguments) {
    if (arg.kind === 'operand') {
      inputTypes[arg.name] = arg.typeConstraint;
    }
  }
  
  for (const result of operation.results) {
    outputTypes[result.name] = result.typeConstraint;
  }
  
  // Generate execution pin configuration based on operation classification
  const execConfig = generateExecConfig(operation);
  
  return {
    operation,
    attributes,
    inputTypes,
    outputTypes,
    // Execution pins based on operation type
    execIn: execConfig.hasExecIn ? createExecIn() : undefined,
    execOuts: execConfig.execOuts,
    // Region data pins for control flow operations
    regionPins: execConfig.regionPins,
  };
}

/**
 * Connection error toast component
 */
interface ConnectionErrorToastProps {
  message: string;
  onClose: () => void;
}

function ConnectionErrorToast({ message, onClose }: ConnectionErrorToastProps) {
  return (
    <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-50 animate-fade-in">
      <div className="bg-red-600 text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 max-w-md">
        <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
        </svg>
        <span className="text-sm">{message}</span>
        <button 
          onClick={onClose}
          className="ml-2 text-white/80 hover:text-white"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>
      </div>
    </div>
  );
}

/**
 * Properties panel for editing selected node
 */
interface PropertiesPanelProps {
  selectedNode: Node | null;
  onNodeUpdate?: (nodeId: string, data: unknown) => void;
}

function PropertiesPanel({ selectedNode }: PropertiesPanelProps) {
  // 面板只在选中节点时显示，所以 selectedNode 不会为 null
  if (!selectedNode) return null;

  const nodeData = selectedNode.data as BlueprintNodeData | undefined;
  const operation = nodeData?.operation;

  return (
    <div className="p-4 overflow-y-auto h-full">
      <h2 className="text-lg font-semibold text-white mb-4">Properties</h2>
      
      {/* Node Info */}
      <div className="mb-4 p-3 bg-gray-700 rounded">
        <div className="text-sm text-gray-300">
          <span className="text-gray-500">ID:</span> {selectedNode.id}
        </div>
        {operation && (
          <>
            <div className="text-sm text-gray-300 mt-1">
              <span className="text-gray-500">Operation:</span> {operation.fullName}
            </div>
            {operation.summary && (
              <div className="text-xs text-gray-400 mt-2">
                {operation.summary}
              </div>
            )}
          </>
        )}
      </div>

      {/* Position */}
      <div className="mb-4">
        <h3 className="text-sm font-medium text-gray-300 mb-2">Position</h3>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-500">X</label>
            <input
              type="number"
              value={Math.round(selectedNode.position.x)}
              readOnly
              className="w-full bg-gray-700 text-white text-sm px-2 py-1 rounded border border-gray-600"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">Y</label>
            <input
              type="number"
              value={Math.round(selectedNode.position.y)}
              readOnly
              className="w-full bg-gray-700 text-white text-sm px-2 py-1 rounded border border-gray-600"
            />
          </div>
        </div>
      </div>

      {/* Attributes */}
      {nodeData?.attributes && Object.keys(nodeData.attributes).length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Attributes</h3>
          <div className="space-y-2">
            {Object.entries(nodeData.attributes).map(([key, value]) => (
              <div key={key} className="text-sm">
                <span className="text-gray-500">{key}:</span>{' '}
                <span className="text-gray-300">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Types */}
      {nodeData?.inputTypes && Object.keys(nodeData.inputTypes).length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Input Types</h3>
          <div className="space-y-1">
            {Object.entries(nodeData.inputTypes).map(([port, type]) => (
              <div key={port} className="text-xs flex justify-between">
                <span className="text-gray-400">{port}</span>
                <span className="text-blue-400">{type}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Output Types */}
      {nodeData?.outputTypes && Object.keys(nodeData.outputTypes).length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Output Types</h3>
          <div className="space-y-1">
            {Object.entries(nodeData.outputTypes).map(([port, type]) => (
              <div key={port} className="text-xs flex justify-between">
                <span className="text-gray-400">{port}</span>
                <span className="text-green-400">{type}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * MainLayout Inner Component
 * 
 * Contains the actual layout implementation.
 * Must be wrapped in ReactFlowProvider to use useReactFlow hook.
 */
/**
 * Clipboard data structure for copy/paste operations
 */
interface ClipboardData {
  nodes: Node[];
  edges: Edge[];
}

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
  
  // Get resolved types from type store for connection validation
  const resolvedTypes = useTypeStore(state => state.resolvedTypes);
  
  // Get project state and actions
  const project = useProjectStore(state => state.project);
  const currentFunctionId = useProjectStore(state => state.currentFunctionId);
  const createProject = useProjectStore(state => state.createProject);
  const updateFunctionGraph = useProjectStore(state => state.updateFunctionGraph);
  
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

  // Initialize a default project if none exists
  useEffect(() => {
    if (!project) {
      createProject('Untitled Project', './untitled_project', ['arith', 'func']);
    }
  }, [project, createProject]);

  // Track if we're loading from store to avoid triggering auto-save
  const isLoadingFromStoreRef = useRef(false);

  // Load graph when function changes or when project changes
  // Track function ID, project path, and project object reference to detect changes
  const lastFunctionIdRef = useRef<string | null>(null);
  const lastProjectPathRef = useRef<string | null>(null);
  const lastProjectRef = useRef<Project | null>(null);
  
  useEffect(() => {
    if (!currentFunction || !project) return;
    
    const currentGraph = currentFunction.graph;
    const isFunctionSwitch = lastFunctionIdRef.current !== currentFunctionId;
    const isProjectSwitch = lastProjectPathRef.current !== project.path;
    // Also detect when the same path is reloaded (project object reference changed)
    const isProjectReload = lastProjectRef.current !== project && lastProjectPathRef.current === project.path;
    
    // Reload graph when switching functions OR when project changes OR when project is reloaded
    // This ensures the graph is updated when opening a different project or reopening the same project
    if (isFunctionSwitch || isProjectSwitch || isProjectReload) {
      isLoadingFromStoreRef.current = true;
      const graphNodes = currentGraph.nodes as Node[];
      const graphEdges = currentGraph.edges as Edge[];
      setNodes(graphNodes);
      setEdges(graphEdges);
      // 传播模型：无需初始化，类型传播是无状态的
      lastFunctionIdRef.current = currentFunctionId;
      lastProjectPathRef.current = project.path;
      lastProjectRef.current = project;
      // Reset the flag after a microtask to allow React to batch the state updates
      queueMicrotask(() => {
        isLoadingFromStoreRef.current = false;
      });
    }
  }, [currentFunctionId, currentFunction, project, setNodes, setEdges]);

  // Derive selected node validity - if function changes, selected node should be null
  // This is computed during render, not in an effect
  const effectiveSelectedNode = selectedNode && nodes.find(n => n.id === selectedNode.id) 
    ? selectedNode 
    : null;

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
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle || '',
        targetHandle: e.targetHandle || '',
      })),
    };
  }, [nodes, edges]);

  // Auto-save graph changes (debounced)
  // Only save when persistable data changes (position, data), not UI state (selected)
  const lastSavedGraphRef = useRef<string>('');
  
  useEffect(() => {
    // Skip if we're loading from store
    if (isLoadingFromStoreRef.current) return;
    
    if (!currentFunctionId) return;
    
    // Create a serializable representation of persistable data only
    // Exclude UI state like 'selected', 'dragging', etc.
    const persistableData = JSON.stringify({
      nodes: nodes.map(n => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: n.data,
      })),
      edges: edges.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
      })),
    });
    
    // Skip if nothing changed
    if (persistableData === lastSavedGraphRef.current) return;
    
    const timer = setTimeout(() => {
      const graphState = convertToGraphState();
      updateFunctionGraph(currentFunctionId, graphState);
      lastSavedGraphRef.current = persistableData;
    }, 500);
    
    return () => clearTimeout(timer);
  }, [currentFunctionId, nodes, edges, convertToGraphState, updateFunctionGraph]);

  // Handle node selection
  const handleNodesChange = useCallback((changes: NodeChange[]) => {
    onNodesChange(changes);
    
    // Track selection changes
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

  // Handle edge changes
  const handleEdgesChange = useCallback((changes: EdgeChange[]) => {
    onEdgesChange(changes);
  }, [onEdgesChange]);

  /**
   * Copy selected nodes and their internal edges to clipboard
   * Preserves node configuration and connections between selected nodes
   */
  const copySelectedNodes = useCallback(() => {
    const selectedNodes = nodes.filter(n => n.selected);
    if (selectedNodes.length === 0) return;
    
    const selectedNodeIds = new Set(selectedNodes.map(n => n.id));
    
    // Only copy edges that connect selected nodes (internal edges)
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
   * Maintains internal connections between pasted nodes
   */
  const pasteNodes = useCallback(() => {
    if (!clipboardRef.current || clipboardRef.current.nodes.length === 0) return;
    
    const { nodes: copiedNodes, edges: copiedEdges } = clipboardRef.current;
    
    // Create ID mapping: old ID -> new ID
    const idMap = new Map<string, string>();
    const offset = { x: 50, y: 50 }; // Offset for pasted nodes
    
    // Create new nodes with new IDs (skip Entry nodes, allow Return nodes)
    const newNodes: Node[] = [];
    for (const node of copiedNodes) {
      // Don't allow copying Entry nodes (only one per function)
      if (node.type === 'function-entry') {
        continue;
      }
      
      const newId = generateNodeId();
      idMap.set(node.id, newId);
      
      newNodes.push({
        ...node,
        id: newId,
        position: {
          x: node.position.x + offset.x,
          y: node.position.y + offset.y,
        },
        selected: true, // Select pasted nodes
        data: { ...node.data }, // Deep copy data
      });
    }
    
    if (newNodes.length === 0) return;
    
    // Create new edges with updated source/target IDs
    const newEdges: Edge[] = copiedEdges
      .filter(edge => idMap.has(edge.source) && idMap.has(edge.target))
      .map(edge => ({
        ...edge,
        id: `edge-${idMap.get(edge.source)}-${edge.sourceHandle}-${idMap.get(edge.target)}-${edge.targetHandle}`,
        source: idMap.get(edge.source)!,
        target: idMap.get(edge.target)!,
        selected: false,
      }));
    
    // Deselect existing nodes and add new ones
    setNodes(nds => [
      ...nds.map(n => ({ ...n, selected: false })),
      ...newNodes,
    ]);
    
    setEdges(eds => [...eds, ...newEdges]);
    
    console.log(`Pasted ${newNodes.length} nodes and ${newEdges.length} edges`);
  }, [setNodes, setEdges]);

  /**
   * Delete selected nodes and edges
   * - Entry nodes cannot be deleted (only one per function)
   * - Return nodes can be deleted, but at least one must remain
   */
  const deleteSelected = useCallback(() => {
    // Count total Return nodes and selected Return nodes
    const allReturnNodes = nodes.filter(n => n.type === 'function-return');
    const selectedReturnNodes = allReturnNodes.filter(n => n.selected);
    
    // Calculate how many Return nodes we can delete (keep at least 1)
    const maxReturnNodesToDelete = Math.max(0, allReturnNodes.length - 1);
    const returnNodesToDelete = selectedReturnNodes.slice(0, maxReturnNodesToDelete);
    const returnNodeIdsToDelete = new Set(returnNodesToDelete.map(n => n.id));
    
    // Get selected nodes that can be deleted:
    // - Not Entry nodes
    // - Return nodes only if we can delete them (keeping at least 1)
    const selectedNodeIds = new Set(
      nodes
        .filter(n => {
          if (!n.selected) return false;
          if (n.type === 'function-entry') return false; // Never delete Entry
          if (n.type === 'function-return') return returnNodeIdsToDelete.has(n.id);
          return true;
        })
        .map(n => n.id)
    );
    
    // Get selected edges
    const selectedEdgeIds = new Set(edges.filter(e => e.selected).map(e => e.id));
    
    if (selectedNodeIds.size === 0 && selectedEdgeIds.size === 0) return;
    
    // Remove selected nodes
    if (selectedNodeIds.size > 0) {
      setNodes(nds => nds.filter(n => !selectedNodeIds.has(n.id)));
      // Also remove edges connected to deleted nodes
      setEdges(eds => eds.filter(e => 
        !selectedNodeIds.has(e.source) && !selectedNodeIds.has(e.target)
      ));
    }
    
    // Remove selected edges
    if (selectedEdgeIds.size > 0) {
      setEdges(eds => eds.filter(e => !selectedEdgeIds.has(e.id)));
    }
    
    // Log deletion info
    const skippedReturnNodes = selectedReturnNodes.length - returnNodesToDelete.length;
    if (skippedReturnNodes > 0) {
      console.log(`Kept ${skippedReturnNodes} Return node(s) to ensure at least one remains`);
    }
    console.log(`Deleted ${selectedNodeIds.size} nodes and ${selectedEdgeIds.size} edges`);
  }, [nodes, edges, setNodes, setEdges]);

  /**
   * Handle keyboard shortcuts for copy/paste/delete
   */
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Check if focus is on an input element
      const target = event.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return;
      }
      
      // Ctrl+C or Cmd+C: Copy
      if ((event.ctrlKey || event.metaKey) && event.key === 'c') {
        event.preventDefault();
        copySelectedNodes();
      }
      
      // Ctrl+V or Cmd+V: Paste
      if ((event.ctrlKey || event.metaKey) && event.key === 'v') {
        event.preventDefault();
        pasteNodes();
      }
      
      // Delete or Backspace: Delete selected
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
   * 
   * 删除边后，约束系统会：
   * 1. 移除连接约束
   * 2. 重置非钉住的端口到原始值域
   * 3. 重新传播约束
   * 
   * 钉住的端口（用户显式选择的类型）不会被重置
   */
  const handleEdgeDoubleClick = useCallback((_event: React.MouseEvent, edge: Edge) => {
    // 删除边
    const remainingEdges = edges.filter(e => e.id !== edge.id);
    setEdges(remainingEdges);
    
    // 传播模型：重新计算类型传播（数据边）
    if (!edge.sourceHandle?.startsWith('exec-')) {
      const graph = buildPropagationGraph(nodes, remainingEdges, currentFunction ?? undefined);
      const sources = extractTypeSources(nodes);
      const propagationResult = propagateTypes(graph, sources);
      
      console.log('Edge removed, propagation result:', {
        types: [...propagationResult.types.entries()]
      });
      
      // 统一更新所有节点的显示类型
      setNodes(nds => applyPropagationResult(nds, propagationResult));
    }
    
    console.log(`Deleted edge: ${edge.id}`);
  }, [edges, nodes, setEdges, setNodes, currentFunction]);

  // Handle function selection from FunctionManager
  const handleFunctionSelect = useCallback((functionId: string) => {
    console.log('Switched to function:', functionId);
  }, []);

  // Handle drag start from palette for operations
  const handleDragStart = useCallback((_event: React.DragEvent, operation: OperationDef) => {
    // Data is already set by NodePalette
    console.log('Dragging operation:', operation.fullName);
  }, []);

  // Handle drag start from palette for custom functions
  const handleFunctionDragStart = useCallback((_event: React.DragEvent, func: FunctionDef) => {
    console.log('Dragging function:', func.name);
  }, []);

  // Get React Flow instance for coordinate conversion
  const reactFlowInstance = useReactFlow();

  // Handle drop on canvas
  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      // Use screenToFlowPosition for correct coordinate conversion
      // This handles zoom and pan correctly
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      // Check if it's a function drop
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

      // Check if it's an operation drop
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
        // 传播模型：新节点没有 pinned 类型，不需要特殊处理
      } catch (err) {
        console.error('Failed to parse dropped operation:', err);
      }
    },
    [setNodes, reactFlowInstance]
  );

  // Handle drag over to allow drop
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  }, []);

  /**
   * Validates a connection before it's created
   * Requirements: 7.1, 7.2
   * 
   * Connection rules:
   * - Execution output (exec-out): max 1 connection
   * - Execution input (exec-in): unlimited connections
   * - Data output: unlimited connections
   * - Data input: max 1 connection
   */
  const isValidConnection = useCallback(
    (connection: Edge | Connection): boolean => {
      const normalizedConnection: Connection = {
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle ?? null,
        targetHandle: connection.targetHandle ?? null,
      };
      // Pass existing edges for connection count validation
      const existingEdges = edges.map(e => ({
        source: e.source,
        sourceHandle: e.sourceHandle ?? null,
        target: e.target,
        targetHandle: e.targetHandle ?? null,
      }));
      const result = validateConnection(normalizedConnection, nodes, resolvedTypes, existingEdges);
      return result.isValid;
    },
    [nodes, edges, resolvedTypes]
  );

  /**
   * Determines if a handle is an execution pin
   */
  const isExecHandle = useCallback((handleId: string | null | undefined): boolean => {
    if (!handleId) return false;
    return handleId.startsWith('exec-');
  }, []);

  /**
   * Gets the color for a data connection based on the source handle type
   */
  const getEdgeColor = useCallback((sourceNodeId: string, sourceHandleId: string | null | undefined): string => {
    if (!sourceHandleId) return '#4A90D9';
    
    // Get the resolved type for this port from typeStore
    const nodeTypesMap = resolvedTypes.get(sourceNodeId);
    if (nodeTypesMap) {
      const resolvedType = nodeTypesMap.get(sourceHandleId);
      if (resolvedType) {
        return getTypeColor(resolvedType);
      }
    }
    
    // Try to get type from node data
    const sourceNode = nodes.find(n => n.id === sourceNodeId);
    if (sourceNode) {
      // Handle FunctionEntryNode - outputs are in data.outputs array
      if (sourceNode.type === 'function-entry') {
        const entryData = sourceNode.data as FunctionEntryData;
        if (entryData.outputs) {
          const port = entryData.outputs.find(p => p.id === sourceHandleId);
          if (port) {
            return getTypeColor(port.concreteType || port.typeConstraint);
          }
        }
      }
      
      // Handle FunctionReturnNode - inputs are in data.inputs array (but Return node doesn't have source handles)
      // This case is unlikely since Return node only has target handles
      
      // Handle FunctionCallNode - outputs are in data.outputs array
      if (sourceNode.type === 'function-call') {
        const callData = sourceNode.data as FunctionCallData;
        if (callData.outputs) {
          const port = callData.outputs.find(p => p.id === sourceHandleId);
          if (port) {
            return getTypeColor(port.concreteType || port.typeConstraint);
          }
        }
      }
      
      // Handle BlueprintNode (operation) - outputTypes is a Record
      if (sourceNode.type === 'operation') {
        const nodeData = sourceNode.data as BlueprintNodeData;
        if (nodeData.outputTypes && sourceHandleId.startsWith('output-')) {
          const portName = sourceHandleId.replace('output-', '');
          const typeConstraint = nodeData.outputTypes[portName];
          if (typeConstraint) {
            return getTypeColor(typeConstraint);
          }
        }
      }
    }
    
    // Default blue for data connections
    return '#4A90D9';
  }, [resolvedTypes, nodes]);

  /**
   * Handles connection creation with validation
   * Requirements: 7.1, 7.2, 7.3
   */
  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      // Pass existing edges for connection count validation
      const existingEdges = edges.map(e => ({
        source: e.source,
        sourceHandle: e.sourceHandle ?? null,
        target: e.target,
        targetHandle: e.targetHandle ?? null,
      }));
      
      const validationResult: ConnectionValidationResult = validateConnection(
        connection, 
        nodes, 
        resolvedTypes,
        existingEdges
      );
      
      if (!validationResult.isValid) {
        setConnectionError(validationResult.errorMessage || 'Invalid connection');
        setTimeout(() => setConnectionError(null), 3000);
        return;
      }
      
      // Determine edge type and style based on handle type
      const isExec = isExecHandle(connection.sourceHandle) || isExecHandle(connection.targetHandle);
      
      const edgeWithStyle: Edge = {
        ...connection,
        id: `edge-${connection.source}-${connection.sourceHandle}-${connection.target}-${connection.targetHandle}`,
        type: isExec ? 'execution' : 'data',
        data: isExec ? undefined : { color: getEdgeColor(connection.source!, connection.sourceHandle) },
        animated: isExec,
      };
      
      const newEdges = [...edges, edgeWithStyle];
      setEdges(newEdges);
      
      // 传播模型：重新计算类型传播（数据边）
      if (!isExec) {
        const graph = buildPropagationGraph(nodes, newEdges, currentFunction ?? undefined);
        const sources = extractTypeSources(nodes);
        const propagationResult = propagateTypes(graph, sources);
        
        console.log('Edge added, propagation result:', {
          types: [...propagationResult.types.entries()]
        });
        
        // 统一更新所有节点的显示类型
        setNodes(nds => applyPropagationResult(nds, propagationResult));
      }
    },
    [nodes, edges, resolvedTypes, setEdges, setNodes, isExecHandle, getEdgeColor, currentFunction]
  );

  /**
   * Dismiss connection error toast
   */
  const dismissError = useCallback(() => {
    setConnectionError(null);
  }, []);

  /**
   * Toggle execution panel expansion
   */
  const toggleExecutionPanel = useCallback(() => {
    setExecutionPanelExpanded(prev => !prev);
  }, []);

  // Calculate execution panel height for layout
  const executionPanelHeight = executionPanelExpanded ? 256 : 32; // h-64 = 256px, h-8 = 32px

  // Project toolbar component
  const ProjectToolbar = (
    <div className="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4 gap-4">
      {/* Logo/Title */}
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center">
          <span className="text-white font-bold text-sm">ML</span>
        </div>
        <span className="text-white font-semibold">MLIR Blueprint Editor</span>
      </div>
      
      {/* Separator */}
      <div className="h-6 w-px bg-gray-600" />
      
      {/* Project Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setIsCreateDialogOpen(true)}
          className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors flex items-center gap-1.5"
          title="Create new project"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New
        </button>
        
        <button
          onClick={() => setIsOpenDialogOpen(true)}
          className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors flex items-center gap-1.5"
          title="Open existing project"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
          </svg>
          Open
        </button>
        
        <button
          onClick={() => setIsSaveDialogOpen(true)}
          disabled={!project}
          className="px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
          title="Save project"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
          </svg>
          Save
        </button>
      </div>
      
      {/* Separator */}
      <div className="h-6 w-px bg-gray-600" />
      
      {/* Current Project Info */}
      {project && (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-gray-500">Project:</span>
          <span className="text-gray-300">{project.name}</span>
          <span className="text-gray-600">|</span>
          <span className="text-gray-500 text-xs">{project.path}</span>
        </div>
      )}
      
      {/* Spacer */}
      <div className="flex-1" />
      
      {/* Status */}
      <div className="text-xs text-gray-500">
        {project ? `${project.customFunctions.length + 1} functions` : 'No project'}
      </div>
    </div>
  );

  return (
    <div className="w-full h-screen flex flex-col bg-gray-900">
      {/* Project Toolbar */}
      {ProjectToolbar}
      
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
          {/* Function Manager Section */}
          <div className="border-b border-gray-700 max-h-64 flex-shrink-0">
            <FunctionManager onFunctionSelect={handleFunctionSelect} />
          </div>
          
          {/* Node Palette Section */}
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
              // Enable multi-selection with box select and edge selection
              // Left-click drag = box select, Right-click/Middle-click drag = pan (like UE5 Blueprint)
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
            
            {/* Connection Error Toast */}
            {connectionError && (
              <ConnectionErrorToast 
                message={connectionError} 
                onClose={dismissError} 
              />
            )}
          </div>
        </div>

        {/* Right Panel - Properties (只在选中单个节点时显示) */}
        {effectiveSelectedNode && (
          <div className="w-72 bg-gray-800 border-l border-gray-700 flex-shrink-0 overflow-hidden">
            <PropertiesPanel selectedNode={effectiveSelectedNode} />
          </div>
        )}
      </div>

      {/* Bottom Panel - Execution (动态调整右边距) */}
      <div 
        className="flex-shrink-0 absolute bottom-0 left-64"
        style={{ 
          height: executionPanelHeight,
          right: effectiveSelectedNode ? 288 : 0, // w-72 = 288px
        }}
      >
        <ExecutionPanel
          projectPath={project?.path}
          isExpanded={executionPanelExpanded}
          onToggleExpand={toggleExecutionPanel}
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
      />
      <OpenProjectDialog
        isOpen={isOpenDialogOpen}
        onClose={() => setIsOpenDialogOpen(false)}
      />
      <SaveProjectDialog
        isOpen={isSaveDialogOpen}
        onClose={() => setIsSaveDialogOpen(false)}
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
