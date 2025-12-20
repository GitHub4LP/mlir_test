/**
 * useGraphEditor Hook
 * 
 * 封装图编辑器的核心逻辑：
 * - 节点/边状态管理（使用 editorStore）
 * - 连接验证和类型传播
 * - 复制/粘贴/删除
 * - 函数切换和图加载
 * - 拖放处理
 * 
 * 设计原则：
 * - 使用框架无关的 EditorNode/EditorEdge 类型
 * - 不依赖任何渲染框架（React Flow、Vue Flow 等）
 * - 通过 editorStore 管理状态
 */

import { useCallback, useRef, useEffect } from 'react';
import { useEditorStore } from '../../core/stores/editorStore';
import { useProjectStore } from '../../stores/projectStore';
import { useTypeConstraintStore } from '../../stores/typeConstraintStore';
import type { EditorNode, EditorEdge, EditorViewport, NodeChange, ConnectionRequest } from '../../editor/types';
import type { OperationDef, BlueprintNodeData, GraphState, FunctionEntryData, FunctionReturnData, DataPin } from '../../types';
import { validatePorts, type ConnectionValidationResult } from '../../editor/adapters/shared/ConnectionValidator';
import { triggerTypePropagationWithSignature } from '../../services/typePropagation';
import { dataInHandle, dataOutHandle } from '../../services/port';
import { getDisplayType } from '../../services/typeSelectorRenderer';
import {
  generateNodeId,
  generateEdgeId,
  convertGraphEdgeToReactFlowEdge,
  convertGraphNodeToReactFlowNode,
  createBlueprintNodeData,
  getEdgeColor,
  updateEdgeColors,
} from '../../utils';

/** 剪贴板数据 */
interface ClipboardData {
  nodes: EditorNode[];
  edges: EditorEdge[];
}

export interface UseGraphEditorReturn {
  // 状态
  nodes: EditorNode[];
  edges: EditorEdge[];
  viewport: EditorViewport;
  
  // 节点操作
  setNodes: (nodes: EditorNode[]) => void;
  setEdges: (edges: EditorEdge[]) => void;
  
  // 事件处理
  handleNodesChange: (changes: NodeChange[]) => void;
  handleConnect: (connection: ConnectionRequest) => { success: boolean; error?: string };
  handleDrop: (x: number, y: number, dataTransfer: DataTransfer) => void;
  handleEdgeDoubleClick: (edgeId: string) => void;
  
  // 编辑操作
  copySelectedNodes: () => void;
  pasteNodes: () => void;
  deleteSelected: () => void;
  
  // 函数操作
  handleFunctionSelect: (functionId: string) => void;
  saveCurrentGraph: (functionId: string) => void;
  loadFunctionGraph: (functionId: string) => void;
  
  // 视口
  setViewport: (viewport: EditorViewport) => void;
  
  // 类型传播
  triggerTypePropagation: () => void;
  
  // 工具函数
  isExecHandle: (handleId: string | null | undefined) => boolean;
  getEdgeColorForPort: (nodeId: string, handleId: string | null | undefined) => string;
}

export function useGraphEditor(): UseGraphEditorReturn {
  // Store 状态
  const nodes = useEditorStore(state => state.nodes);
  const edges = useEditorStore(state => state.edges);
  const viewport = useEditorStore(state => state.viewport);
  const setNodesStore = useEditorStore(state => state.setNodes);
  const setEdgesStore = useEditorStore(state => state.setEdges);
  const setViewportStore = useEditorStore(state => state.setViewport);
  const addEdge = useEditorStore(state => state.addEdge);
  const removeNodes = useEditorStore(state => state.removeNodes);
  const removeEdges = useEditorStore(state => state.removeEdges);
  const applyNodeChanges = useEditorStore(state => state.applyNodeChanges);
  const loadGraph = useEditorStore(state => state.loadGraph);
  
  // Project store
  const currentFunctionId = useProjectStore(state => state.currentFunctionId);
  const updateFunctionGraph = useProjectStore(state => state.updateFunctionGraph);
  const updateSignatureConstraints = useProjectStore(state => state.updateSignatureConstraints);
  const getFunctionById = useProjectStore(state => state.getFunctionById);
  const selectFunction = useProjectStore(state => state.selectFunction);
  
  // 获取当前函数
  const currentFunction = useProjectStore(state => {
    if (!state.project || !state.currentFunctionId) return null;
    if (state.project.mainFunction.id === state.currentFunctionId) {
      return state.project.mainFunction;
    }
    return state.project.customFunctions.find(f => f.id === state.currentFunctionId) || null;
  });
  
  // Type constraint store
  const getConstraintElements = useTypeConstraintStore(state => state.getConstraintElements);
  const pickConstraintName = useTypeConstraintStore(state => state.pickConstraintName);
  
  // 剪贴板
  const clipboardRef = useRef<ClipboardData | null>(null);
  
  // 是否正在从 store 加载（避免触发自动保存）
  const isLoadingFromStoreRef = useRef(false);
  
  // ============================================================
  // 工具函数
  // ============================================================
  
  const isExecHandle = useCallback((handleId: string | null | undefined): boolean => {
    if (!handleId) return false;
    return handleId.startsWith('exec-');
  }, []);
  
  const getEdgeColorForPort = useCallback((nodeId: string, handleId: string | null | undefined): string => {
    return getEdgeColor(nodes, nodeId, handleId);
  }, [nodes]);
  
  // ============================================================
  // 图状态转换
  // ============================================================
  
  const convertToGraphState = useCallback((): GraphState => {
    return {
      nodes: nodes.map(n => ({
        id: n.id,
        type: n.type,
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
  
  // ============================================================
  // 类型传播
  // ============================================================
  
  const triggerTypePropagation = useCallback(() => {
    if (!currentFunction) return;
    
    const result = triggerTypePropagationWithSignature(
      nodes, edges, currentFunction, getConstraintElements, pickConstraintName
    );
    
    setNodesStore(result.nodes);
    const updatedEdges = updateEdgeColors(result.nodes, edges);
    setEdgesStore(updatedEdges);
  }, [nodes, edges, currentFunction, getConstraintElements, pickConstraintName, setNodesStore, setEdgesStore]);
  
  // ============================================================
  // 保存/加载图
  // ============================================================
  
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
    
    const edgesWithColors = updateEdgeColors(result.nodes, graphEdges);
    
    loadGraph(result.nodes, edgesWithColors);
    
    queueMicrotask(() => {
      isLoadingFromStoreRef.current = false;
    });
  }, [getFunctionById, getConstraintElements, pickConstraintName, loadGraph]);
  
  // ============================================================
  // 函数切换
  // ============================================================
  
  const handleFunctionSelect = useCallback((functionId: string) => {
    if (currentFunctionId && currentFunctionId !== functionId) {
      saveCurrentGraph(currentFunctionId);
    }
    selectFunction(functionId);
    loadFunctionGraph(functionId);
  }, [currentFunctionId, saveCurrentGraph, selectFunction, loadFunctionGraph]);
  
  // ============================================================
  // 节点变更处理
  // ============================================================
  
  const handleNodesChange = useCallback((changes: NodeChange[]) => {
    applyNodeChanges(changes);
  }, [applyNodeChanges]);
  
  // ============================================================
  // 连接处理
  // ============================================================
  
  /**
   * 获取端口类型约束（用于连接验证）
   */
  const getPortType = useCallback((nodeId: string, portId: string): string | null => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return null;
    
    const data = node.data as BlueprintNodeData | FunctionEntryData | FunctionReturnData;
    
    // 根据节点类型获取端口类型
    if (node.type === 'function-entry') {
      const entryData = data as FunctionEntryData;
      if (portId.startsWith('data-out-') && Array.isArray(entryData.outputs)) {
        const portName = portId.replace('data-out-', '');
        const port = entryData.outputs.find(p => p.name === portName);
        return port ? (entryData.outputTypes?.[portName] || port.typeConstraint) : null;
      }
    } else if (node.type === 'function-return') {
      const returnData = data as FunctionReturnData;
      if (portId.startsWith('data-in-') && Array.isArray(returnData.inputs)) {
        const portName = portId.replace('data-in-', '');
        const port = returnData.inputs.find(p => p.name === portName);
        return port ? (returnData.inputTypes?.[portName] || port.typeConstraint) : null;
      }
    } else if (node.type === 'function-call') {
      const callData = data as { inputs?: Array<{ name: string; typeConstraint: string }>; outputs?: Array<{ name: string; typeConstraint: string }>; inputTypes?: Record<string, string>; outputTypes?: Record<string, string> };
      if (portId.startsWith('data-in-') && Array.isArray(callData.inputs)) {
        const portName = portId.replace('data-in-', '');
        const port = callData.inputs.find(p => p.name === portName);
        return port ? (callData.inputTypes?.[portName] || port.typeConstraint) : null;
      } else if (portId.startsWith('data-out-') && Array.isArray(callData.outputs)) {
        const portName = portId.replace('data-out-', '');
        const port = callData.outputs.find(p => p.name === portName);
        return port ? (callData.outputTypes?.[portName] || port.typeConstraint) : null;
      }
    } else {
      // Operation 节点
      const opData = data as { inputTypes?: Record<string, string>; outputTypes?: Record<string, string> };
      const portName = portId.replace(/^data-(in|out)-/, '').replace(/_\d+$/, '');
      
      if (portId.startsWith('data-out-')) {
        return opData.outputTypes?.[portName] || null;
      } else if (portId.startsWith('data-in-')) {
        return opData.inputTypes?.[portName] || null;
      }
    }
    
    return null;
  }, [nodes]);
  
  /**
   * 验证连接计数约束
   */
  const validateConnectionCount = useCallback((
    connection: ConnectionRequest,
    existingEdges: EditorEdge[]
  ): { isValid: boolean; errorMessage?: string } => {
    const { source, sourceHandle, target, targetHandle } = connection;
    
    const isSourceExec = sourceHandle.startsWith('exec-');
    const isTargetExec = targetHandle.startsWith('exec-');
    
    // 执行输出只能有 1 个连接
    if (isSourceExec) {
      const existingFromSource = existingEdges.filter(
        e => e.source === source && e.sourceHandle === sourceHandle
      );
      if (existingFromSource.length >= 1) {
        return { isValid: false, errorMessage: '执行输出只能有一个连接' };
      }
    }
    
    // 数据输入只能有 1 个连接
    if (!isTargetExec) {
      const existingToTarget = existingEdges.filter(
        e => e.target === target && e.targetHandle === targetHandle
      );
      if (existingToTarget.length >= 1) {
        return { isValid: false, errorMessage: '数据输入只能有一个连接' };
      }
    }
    
    return { isValid: true };
  }, []);
  
  const handleConnect = useCallback((connection: ConnectionRequest): { success: boolean; error?: string } => {
    const { source, sourceHandle, target, targetHandle } = connection;
    
    // 1. 基本验证
    if (!source || !sourceHandle || !target || !targetHandle) {
      return { success: false, error: '连接信息不完整' };
    }
    
    // 2. 连接计数验证
    const countResult = validateConnectionCount(connection, edges);
    if (!countResult.isValid) {
      return { success: false, error: countResult.errorMessage };
    }
    
    // 3. 使用 validatePorts 进行类型验证
    const validationResult: ConnectionValidationResult = validatePorts(
      source,
      sourceHandle,
      target,
      targetHandle,
      getPortType
    );
    
    if (!validationResult.isValid) {
      return { success: false, error: validationResult.errorMessage || 'Invalid connection' };
    }
    
    const isExec = isExecHandle(sourceHandle) || isExecHandle(targetHandle);
    
    const newEdge: EditorEdge = {
      id: generateEdgeId({ source, sourceHandle, target, targetHandle }),
      source,
      sourceHandle,
      target,
      targetHandle,
      type: isExec ? 'execution' : 'data',
      data: isExec ? undefined : { color: getEdgeColorForPort(source, sourceHandle) },
    };
    
    addEdge(newEdge);
    
    // 触发类型传播
    if (!isExec) {
      // 需要在下一个 tick 触发，因为 edge 刚添加
      queueMicrotask(() => {
        triggerTypePropagation();
      });
    }
    
    return { success: true };
  }, [edges, getPortType, validateConnectionCount, isExecHandle, getEdgeColorForPort, addEdge, triggerTypePropagation]);
  
  // ============================================================
  // 拖放处理
  // ============================================================
  
  const handleDrop = useCallback((x: number, y: number, dataTransfer: DataTransfer) => {
    // 处理函数调用节点
    const functionData = dataTransfer.getData('application/reactflow-function');
    if (functionData) {
      try {
        const functionCallData = JSON.parse(functionData);
        const newNode: EditorNode = {
          id: generateNodeId(),
          type: 'function-call',
          position: { x, y },
          data: functionCallData,
        };
        setNodesStore([...nodes, newNode]);
        return;
      } catch (err) {
        console.error('Failed to parse dropped function:', err);
      }
    }
    
    // 处理操作节点
    const data = dataTransfer.getData('application/json');
    if (!data) return;
    
    try {
      const operation: OperationDef = JSON.parse(data);
      const newNode: EditorNode = {
        id: generateNodeId(),
        type: 'operation',
        position: { x, y },
        data: createBlueprintNodeData(operation),
      };
      setNodesStore([...nodes, newNode]);
    } catch (err) {
      console.error('Failed to parse dropped operation:', err);
    }
  }, [nodes, setNodesStore]);
  
  // ============================================================
  // 边双击删除
  // ============================================================
  
  const handleEdgeDoubleClick = useCallback((edgeId: string) => {
    const edge = edges.find(e => e.id === edgeId);
    if (!edge) return;
    
    removeEdges([edgeId]);
    
    // 如果是数据边，触发类型传播
    if (!edge.sourceHandle?.startsWith('exec-')) {
      queueMicrotask(() => {
        triggerTypePropagation();
      });
    }
  }, [edges, removeEdges, triggerTypePropagation]);
  
  // ============================================================
  // 复制/粘贴/删除
  // ============================================================
  
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
    
    const newNodes: EditorNode[] = [];
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
        id: newId,
        type: node.type,
        position: {
          x: node.position.x + offset.x,
          y: node.position.y + offset.y,
        },
        data: { ...node.data as object },
        selected: true,
      });
    }
    
    if (newNodes.length === 0) return;
    
    const newEdges: EditorEdge[] = copiedEdges
      .filter(edge => idMap.has(edge.source) && idMap.has(edge.target))
      .map(edge => ({
        id: generateEdgeId({
          source: idMap.get(edge.source)!,
          sourceHandle: edge.sourceHandle,
          target: idMap.get(edge.target)!,
          targetHandle: edge.targetHandle,
        }),
        source: idMap.get(edge.source)!,
        sourceHandle: edge.sourceHandle,
        target: idMap.get(edge.target)!,
        targetHandle: edge.targetHandle,
        type: edge.type,
        data: edge.data,
      }));
    
    // 取消现有选择，添加新节点
    const updatedNodes = nodes.map(n => ({ ...n, selected: false }));
    setNodesStore([...updatedNodes, ...newNodes]);
    setEdgesStore([...edges, ...newEdges]);
    
    console.log(`Pasted ${newNodes.length} nodes and ${newEdges.length} edges`);
  }, [nodes, edges, setNodesStore, setEdgesStore]);
  
  const deleteSelected = useCallback(() => {
    const allReturnNodes = nodes.filter(n => n.type === 'function-return');
    const selectedReturnNodes = allReturnNodes.filter(n => n.selected);
    
    const maxReturnNodesToDelete = Math.max(0, allReturnNodes.length - 1);
    const returnNodesToDelete = selectedReturnNodes.slice(0, maxReturnNodesToDelete);
    const returnNodeIdsToDelete = new Set(returnNodesToDelete.map(n => n.id));
    
    const nodeIdsToDelete = nodes
      .filter(n => {
        if (!n.selected) return false;
        if (n.type === 'function-entry') return false;
        if (n.type === 'function-return') return returnNodeIdsToDelete.has(n.id);
        return true;
      })
      .map(n => n.id);
    
    const edgeIdsToDelete = edges
      .filter(e => e.selected)
      .map(e => e.id);
    
    if (nodeIdsToDelete.length === 0 && edgeIdsToDelete.length === 0) return;
    
    if (nodeIdsToDelete.length > 0) {
      removeNodes(nodeIdsToDelete);
    }
    
    if (edgeIdsToDelete.length > 0) {
      removeEdges(edgeIdsToDelete);
    }
    
    // 触发类型传播
    queueMicrotask(() => {
      triggerTypePropagation();
    });
    
    console.log(`Deleted ${nodeIdsToDelete.length} nodes and ${edgeIdsToDelete.length} edges`);
  }, [nodes, edges, removeNodes, removeEdges, triggerTypePropagation]);
  
  // ============================================================
  // 签名同步
  // ============================================================
  
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
  
  return {
    nodes,
    edges,
    viewport,
    setNodes: setNodesStore,
    setEdges: setEdgesStore,
    handleNodesChange,
    handleConnect,
    handleDrop,
    handleEdgeDoubleClick,
    copySelectedNodes,
    pasteNodes,
    deleteSelected,
    handleFunctionSelect,
    saveCurrentGraph,
    loadFunctionGraph,
    setViewport: setViewportStore,
    triggerTypePropagation,
    isExecHandle,
    getEdgeColorForPort,
  };
}
