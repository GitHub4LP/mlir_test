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
import { getPlatformBridge } from '../../platform';
import type { EditorNode, EditorEdge, EditorViewport, NodeChange, ConnectionRequest } from '../../editor/types';
import type { OperationDef, BlueprintNodeData, GraphState, FunctionEntryData, FunctionReturnData, DataPin } from '../../types';
import { validateConnection } from '../../editor/adapters/reactflow/connectionUtils';
import { triggerTypePropagationWithSignature, type DialectFilterConfig } from '../../services/typePropagation';
import { inferFunctionTraits } from '../../services/typePropagation/traitsInference';
import { computeReachableDialects } from '../../services/dialectDependency';
import { dataInHandle, dataOutHandle } from '../../services/port';
import { getDisplayType } from '../../services/typeSelectorRenderer';
import { generateFunctionCallData } from '../../services/functionNodeGenerator';
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
  handleDrop: (x: number, y: number, dataTransfer: DataTransfer) => void | Promise<void>;
  handleEdgeDoubleClick: (edgeId: string) => void;
  
  // 编辑操作
  copySelectedNodes: () => void;
  pasteNodes: () => void;
  deleteSelected: () => void;
  
  // 函数操作
  handleFunctionSelect: (functionName: string) => void;
  saveCurrentGraph: (functionName: string) => void;
  loadFunctionGraph: (functionName: string) => void;
  
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
  const currentFunctionName = useProjectStore(state => state.currentFunctionName);
  const updateFunctionGraph = useProjectStore(state => state.updateFunctionGraph);
  const updateSignatureConstraints = useProjectStore(state => state.updateSignatureConstraints);
  const setFunctionTraits = useProjectStore(state => state.setFunctionTraits);
  const getFunctionByName = useProjectStore(state => state.getFunctionByName);
  const selectFunction = useProjectStore(state => state.selectFunction);
  
  // 获取当前函数
  const currentFunction = useProjectStore(state => {
    if (!state.project || !state.currentFunctionName) return null;
    if (state.project.mainFunction.name === state.currentFunctionName) {
      return state.project.mainFunction;
    }
    return state.project.customFunctions.find(f => f.name === state.currentFunctionName) || null;
  });
  
  // Type constraint store
  const getConstraintElements = useTypeConstraintStore(state => state.getConstraintElements);
  const pickConstraintName = useTypeConstraintStore(state => state.pickConstraintName);
  const findSubsetConstraints = useTypeConstraintStore(state => state.findSubsetConstraints);
  const filterConstraintsByDialects = useTypeConstraintStore(state => state.filterConstraintsByDialects);
  
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
    
    // 从 store 获取最新状态，避免闭包捕获旧值
    const latestNodes = useEditorStore.getState().nodes;
    const latestEdges = useEditorStore.getState().edges;
    const latestProject = useProjectStore.getState().project;
    const latestGetFunctionByName = useProjectStore.getState().getFunctionByName;
    
    // 构建方言过滤配置
    const dialectFilter: DialectFilterConfig | undefined = latestProject ? {
      getReachableDialects: (functionName: string) => computeReachableDialects(functionName, latestProject),
      filterConstraintsByDialects,
    } : undefined;
    
    const result = triggerTypePropagationWithSignature(
      latestNodes, latestEdges, currentFunction, getConstraintElements, pickConstraintName, findSubsetConstraints, dialectFilter, latestGetFunctionByName
    );
    
    setNodesStore(result.nodes);
    const updatedEdges = updateEdgeColors(result.nodes, latestEdges);
    setEdgesStore(updatedEdges);
    
    // 更新推断的 traits（仅非 main 函数）
    if (currentFunction.name !== 'main' && result.inferredTraits) {
      setFunctionTraits(currentFunction.name, result.inferredTraits);
    }
  }, [currentFunction, getConstraintElements, pickConstraintName, findSubsetConstraints, filterConstraintsByDialects, setNodesStore, setEdgesStore, setFunctionTraits]);
  
  // 监听函数参数/返回值变化，自动触发类型传播
  // 这确保添加/删除参数后，新端口能获得正确的 portStates
  const prevParamsLengthRef = useRef<number>(0);
  const prevReturnTypesLengthRef = useRef<number>(0);
  
  useEffect(() => {
    if (!currentFunction) return;
    
    const paramsLength = currentFunction.parameters.length;
    const returnTypesLength = currentFunction.returnTypes.length;
    
    // 检查是否有变化（只检查数量变化，避免不必要的传播）
    if (paramsLength !== prevParamsLengthRef.current || 
        returnTypesLength !== prevReturnTypesLengthRef.current) {
      prevParamsLengthRef.current = paramsLength;
      prevReturnTypesLengthRef.current = returnTypesLength;
      
      // 使用 setTimeout 延迟触发，确保 node.data.outputs/inputs 同步完成
      // （FunctionEntryNode/FunctionReturnNode 的 useEffect 会先执行）
      setTimeout(() => {
        triggerTypePropagation();
      }, 0);
    }
  }, [currentFunction, triggerTypePropagation]);
  
  // ============================================================
  // 保存/加载图
  // ============================================================
  
  const saveCurrentGraph = useCallback((functionName: string) => {
    const graphState = convertToGraphStateRef.current();
    if (graphState.nodes.length > 0) {
      updateFunctionGraph(functionName, graphState);
      
      // 保存时也更新 traits（仅非 main 函数）
      const func = getFunctionByName(functionName);
      if (func && func.name !== 'main') {
        const latestNodes = useEditorStore.getState().nodes;
        const latestEdges = useEditorStore.getState().edges;
        const inferredTraits = inferFunctionTraits(latestNodes, latestEdges, func);
        setFunctionTraits(functionName, inferredTraits);
      }
    }
  }, [updateFunctionGraph, getFunctionByName, setFunctionTraits]);
  
  const loadFunctionGraph = useCallback((functionName: string) => {
    const func = getFunctionByName(functionName);
    if (!func) return;
    
    isLoadingFromStoreRef.current = true;
    
    const graphNodes = func.graph.nodes.map(convertGraphNodeToReactFlowNode);
    const graphEdges = func.graph.edges.map(convertGraphEdgeToReactFlowEdge);
    
    // 获取最新的 project 用于方言过滤
    const latestProject = useProjectStore.getState().project;
    const latestGetFunctionByName = useProjectStore.getState().getFunctionByName;
    const dialectFilter: DialectFilterConfig | undefined = latestProject ? {
      getReachableDialects: (fName: string) => computeReachableDialects(fName, latestProject),
      filterConstraintsByDialects,
    } : undefined;
    
    const result = triggerTypePropagationWithSignature(
      graphNodes, graphEdges, func, getConstraintElements, pickConstraintName, findSubsetConstraints, dialectFilter, latestGetFunctionByName
    );
    
    const edgesWithColors = updateEdgeColors(result.nodes, graphEdges);
    
    loadGraph(result.nodes, edgesWithColors);
    
    // 更新推断的 traits（仅非 main 函数）
    if (func.name !== 'main' && result.inferredTraits) {
      setFunctionTraits(func.name, result.inferredTraits);
    }
    
    queueMicrotask(() => {
      isLoadingFromStoreRef.current = false;
    });
  }, [getFunctionByName, getConstraintElements, pickConstraintName, findSubsetConstraints, filterConstraintsByDialects, loadGraph, setFunctionTraits]);
  
  // ============================================================
  // 函数切换
  // ============================================================
  
  const handleFunctionSelect = useCallback((functionName: string) => {
    if (currentFunctionName && currentFunctionName !== functionName) {
      saveCurrentGraph(currentFunctionName);
    }
    selectFunction(functionName);
    loadFunctionGraph(functionName);
  }, [currentFunctionName, saveCurrentGraph, selectFunction, loadFunctionGraph]);
  
  // ============================================================
  // 节点变更处理
  // ============================================================
  
  const handleNodesChange = useCallback((changes: NodeChange[]) => {
    applyNodeChanges(changes);
  }, [applyNodeChanges]);
  
  // ============================================================
  // 连接处理
  // ============================================================
  
  const handleConnect = useCallback((connection: ConnectionRequest): { success: boolean; error?: string } => {
    const { source, sourceHandle, target, targetHandle } = connection;
    
    // 1. 基本验证
    if (!source || !sourceHandle || !target || !targetHandle) {
      return { success: false, error: '连接信息不完整' };
    }
    
    // 2. 使用 validateConnection 进行完整验证（包括类型和计数）
    // 将 EditorNode 转换为 React Flow Node 格式
    const rfNodes = nodes.map(n => ({
      id: n.id,
      type: n.type,
      position: n.position,
      data: n.data as Record<string, unknown>,
    }));
    
    const validationResult = validateConnection(
      { source, sourceHandle, target, targetHandle },
      rfNodes,
      edges.map(e => ({
        source: e.source,
        sourceHandle: e.sourceHandle,
        target: e.target,
        targetHandle: e.targetHandle,
      }))
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
  }, [nodes, edges, isExecHandle, getEdgeColorForPort, addEdge, triggerTypePropagation]);
  
  // ============================================================
  // 拖放处理
  // ============================================================
  
  const handleDrop = useCallback(async (x: number, y: number, dataTransfer: DataTransfer) => {
    const bridge = getPlatformBridge();
    
    // 1. 处理函数调用节点
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
        queueMicrotask(() => {
          triggerTypePropagation();
        });
        return;
      } catch (err) {
        console.error('Failed to parse dropped function:', err);
      }
    }
    
    // 2. 尝试直接获取操作数据（Web 模式）
    let operationData = dataTransfer.getData('application/json');
    
    // 3. 如果没有，尝试 VS Code TreeView 数据或文件拖放
    // VS Code TreeView 拖放时，text/uri-list 包含我们设置的 resourceUri: mlir-op://arith.addi 或 mlir-func://functionId
    // 文件拖放时，text/uri-list 包含 file:///path/to/file.mlir.json
    if (!operationData && bridge.platform === 'vscode') {
      const uriList = dataTransfer.getData('text/uri-list');
      if (uriList) {
        try {
          // 处理操作拖放: mlir-op://arith.addi -> arith.addi
          if (uriList.startsWith('mlir-op://')) {
            const fullName = uriList.replace('mlir-op://', '');
            if (fullName) {
              const operation = await bridge.resolveOperationData(fullName);
              if (operation) {
                operationData = JSON.stringify(operation);
              }
            }
          }
          // 处理函数拖放: mlir-func://functionName -> functionName
          else if (uriList.startsWith('mlir-func://')) {
            const functionName = uriList.replace('mlir-func://', '');
            if (functionName) {
              const funcInfo = await bridge.resolveFunctionData(functionName);
              if (funcInfo) {
                // 从 projectStore 获取完整的函数定义
                const func = getFunctionByName(functionName);
                if (func) {
                  const functionCallData = generateFunctionCallData(func);
                  const newNode: EditorNode = {
                    id: generateNodeId(),
                    type: 'function-call',
                    position: { x, y },
                    data: functionCallData,
                  };
                  setNodesStore([...nodes, newNode]);
                  queueMicrotask(() => {
                    triggerTypePropagation();
                  });
                  return;
                }
              }
            }
          }
          // 处理文件拖放: file:///path/to/function.mlir.json
          else if (uriList.startsWith('file://')) {
            // 解析文件路径
            let filePath = uriList.replace('file:///', '').replace('file://', '');
            // Windows 路径处理
            filePath = decodeURIComponent(filePath);
            
            // 检查是否是 .mlir.json 文件
            if (filePath.endsWith('.mlir.json')) {
              // 提取函数名（去掉 .mlir.json 后缀）
              const fileName = filePath.split(/[/\\]/).pop() || '';
              const functionName = fileName.replace('.mlir.json', '');
              
              // 不能拖放 main 函数
              if (functionName === 'main') {
                console.warn('Cannot create Call node for main function');
                return;
              }
              
              // 不能拖放当前正在编辑的函数（避免递归调用）
              if (functionName === currentFunctionName) {
                console.warn('Cannot create Call node for current function (recursive call)');
                return;
              }
              
              // 从 projectStore 获取函数定义
              const func = getFunctionByName(functionName);
              if (func) {
                const functionCallData = generateFunctionCallData(func);
                const newNode: EditorNode = {
                  id: generateNodeId(),
                  type: 'function-call',
                  position: { x, y },
                  data: functionCallData,
                };
                setNodesStore([...nodes, newNode]);
                queueMicrotask(() => {
                  triggerTypePropagation();
                });
                return;
              } else {
                console.warn(`Function not found: ${functionName}`);
              }
            }
          }
        } catch (err) {
          console.error('Failed to resolve data from TreeView:', err);
        }
      }
    }
    
    // 4. 创建节点
    if (!operationData) return;
    
    try {
      const operation: OperationDef = JSON.parse(operationData);
      const newNode: EditorNode = {
        id: generateNodeId(),
        type: 'operation',
        position: { x, y },
        data: createBlueprintNodeData(operation),
      };
      setNodesStore([...nodes, newNode]);
      queueMicrotask(() => {
        triggerTypePropagation();
      });
    } catch (err) {
      console.error('Failed to parse dropped operation:', err);
    }
  }, [nodes, setNodesStore, triggerTypePropagation, getFunctionByName, currentFunctionName]);
  
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
    if (!currentFunctionName || !currentFunction || isLoadingFromStoreRef.current) return;
    if (currentFunction.name === 'main') return;
    
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
    
    updateSignatureConstraints(currentFunctionName, parameterConstraints, returnTypeConstraints);
  }, [nodes, currentFunctionName, currentFunction, updateSignatureConstraints]);
  
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
