/**
 * ReactFlow 编辑器包装组件
 * 
 * 将 React Flow 包装为符合 INodeEditor 接口的 React 组件。
 * 这是一个"薄"包装，主要做接口转换，React Flow 的能力完全保留。
 */

import { useCallback, useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import {
  ReactFlow,
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
  type NodeChange as RFNodeChange,
  type EdgeChange as RFEdgeChange,
  type NodeTypes,
  type EdgeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
  NodeChange,
  EdgeChange,
} from '../types';
import {
  toReactFlowNode,
  toReactFlowEdge,
  toReactFlowViewport,
  fromReactFlowViewport,
  convertNodeChanges,
  convertEdgeChanges,
  extractSelection,
  applySelectionToNodes,
  applySelectionToEdges,
  toConnectionRequest,
} from './ReactFlowAdapter';

// ============================================================
// 组件 Props
// ============================================================

export interface ReactFlowEditorWrapperProps {
  /** 节点类型映射 */
  nodeTypes: NodeTypes;
  /** 边类型映射 */
  edgeTypes: EdgeTypes;
  /** 连接验证函数 */
  isValidConnection?: (connection: Edge | Connection) => boolean;
  
  // 事件回调
  onNodesChange?: (changes: NodeChange[]) => void;
  onEdgesChange?: (changes: EdgeChange[]) => void;
  onSelectionChange?: (selection: EditorSelection) => void;
  onViewportChange?: (viewport: EditorViewport) => void;
  onConnect?: (request: ConnectionRequest) => void;
  onNodeDoubleClick?: (nodeId: string) => void;
  onEdgeDoubleClick?: (edgeId: string) => void;
  onDrop?: (x: number, y: number, dataTransfer: DataTransfer) => void;
  onDeleteRequest?: (nodeIds: string[], edgeIds: string[]) => void;
}

/** 暴露给父组件的命令式 API */
export interface ReactFlowEditorHandle {
  setNodes(nodes: EditorNode[]): void;
  setEdges(edges: EditorEdge[]): void;
  setSelection(selection: EditorSelection): void;
  setViewport(viewport: EditorViewport): void;
  fitView(options?: { padding?: number; maxZoom?: number }): void;
  getViewport(): EditorViewport;
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number };
}

// ============================================================
// 内部组件（需要 useReactFlow）
// ============================================================

interface InnerProps extends ReactFlowEditorWrapperProps {
  onHandle: (handle: ReactFlowEditorHandle) => void;
}

function ReactFlowEditorInner({
  nodeTypes,
  edgeTypes,
  isValidConnection,
  onNodesChange,
  onEdgesChange,
  onSelectionChange,
  onViewportChange,
  onConnect,
  onNodeDoubleClick,
  onEdgeDoubleClick,
  onDrop,
  onDeleteRequest,
  onHandle,
}: InnerProps) {
  const [nodes, setNodes, onRFNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onRFEdgesChange] = useEdgesState<Edge>([]);
  const reactFlowInstance = useReactFlow();
  
  // 用 ref 存储回调，避免闭包问题
  const callbacksRef = useRef({
    onNodesChange,
    onEdgesChange,
    onSelectionChange,
    onViewportChange,
    onConnect,
    onNodeDoubleClick,
    onEdgeDoubleClick,
    onDrop,
    onDeleteRequest,
  });
  
  useEffect(() => {
    callbacksRef.current = {
      onNodesChange,
      onEdgesChange,
      onSelectionChange,
      onViewportChange,
      onConnect,
      onNodeDoubleClick,
      onEdgeDoubleClick,
      onDrop,
      onDeleteRequest,
    };
  }, [onNodesChange, onEdgesChange, onSelectionChange, onViewportChange, onConnect, onNodeDoubleClick, onEdgeDoubleClick, onDrop, onDeleteRequest]);

  // 暴露命令式 API
  useEffect(() => {
    const handle: ReactFlowEditorHandle = {
      setNodes: (editorNodes: EditorNode[]) => {
        setNodes(editorNodes.map(toReactFlowNode));
      },
      setEdges: (editorEdges: EditorEdge[]) => {
        setEdges(editorEdges.map(toReactFlowEdge));
      },
      setSelection: (selection: EditorSelection) => {
        setNodes(nds => applySelectionToNodes(nds, selection));
        setEdges(eds => applySelectionToEdges(eds, selection));
      },
      setViewport: (viewport: EditorViewport) => {
        reactFlowInstance.setViewport(toReactFlowViewport(viewport));
      },
      fitView: (options) => {
        reactFlowInstance.fitView({
          padding: options?.padding ?? 0.1,
          maxZoom: options?.maxZoom ?? 1,
        });
      },
      getViewport: () => {
        return fromReactFlowViewport(reactFlowInstance.getViewport());
      },
      screenToCanvas: (screenX: number, screenY: number) => {
        const pos = reactFlowInstance.screenToFlowPosition({ x: screenX, y: screenY });
        return { x: pos.x, y: pos.y };
      },
    };
    onHandle(handle);
  }, [reactFlowInstance, setNodes, setEdges, onHandle]);

  // 处理节点变更
  const handleNodesChange = useCallback((changes: RFNodeChange[]) => {
    // 先应用到 React Flow
    onRFNodesChange(changes);
    
    // 转换并通知外部
    const editorChanges = convertNodeChanges(changes);
    if (editorChanges.length > 0) {
      callbacksRef.current.onNodesChange?.(editorChanges);
    }
    
    // 检查选择变化
    const selectChanges = changes.filter(c => c.type === 'select');
    if (selectChanges.length > 0) {
      // 延迟获取最新状态
      setTimeout(() => {
        const selection = extractSelection(
          reactFlowInstance.getNodes(),
          reactFlowInstance.getEdges()
        );
        callbacksRef.current.onSelectionChange?.(selection);
      }, 0);
    }
  }, [onRFNodesChange, reactFlowInstance]);

  // 处理边变更
  const handleEdgesChange = useCallback((changes: RFEdgeChange[]) => {
    // 先应用到 React Flow
    onRFEdgesChange(changes);
    
    // 转换并通知外部
    const editorChanges = convertEdgeChanges(changes);
    if (editorChanges.length > 0) {
      callbacksRef.current.onEdgesChange?.(editorChanges);
    }
    
    // 检查选择变化
    const selectChanges = changes.filter(c => c.type === 'select');
    if (selectChanges.length > 0) {
      setTimeout(() => {
        const selection = extractSelection(
          reactFlowInstance.getNodes(),
          reactFlowInstance.getEdges()
        );
        callbacksRef.current.onSelectionChange?.(selection);
      }, 0);
    }
  }, [onRFEdgesChange, reactFlowInstance]);

  // 处理连接
  const handleConnect = useCallback((connection: Connection) => {
    const request = toConnectionRequest(connection);
    if (request) {
      callbacksRef.current.onConnect?.(request);
    }
  }, []);

  // 处理边双击
  const handleEdgeDoubleClick = useCallback((_event: React.MouseEvent, edge: Edge) => {
    callbacksRef.current.onEdgeDoubleClick?.(edge.id);
  }, []);

  // 处理节点双击
  const handleNodeDoubleClick = useCallback((_event: React.MouseEvent, node: Node) => {
    callbacksRef.current.onNodeDoubleClick?.(node.id);
  }, []);

  // 处理拖放
  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    const position = reactFlowInstance.screenToFlowPosition({
      x: event.clientX,
      y: event.clientY,
    });
    callbacksRef.current.onDrop?.(position.x, position.y, event.dataTransfer);
  }, [reactFlowInstance]);

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  }, []);

  // 处理视口变化
  const handleMoveEnd = useCallback(() => {
    const viewport = fromReactFlowViewport(reactFlowInstance.getViewport());
    callbacksRef.current.onViewportChange?.(viewport);
  }, [reactFlowInstance]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={handleNodesChange}
      onEdgesChange={handleEdgesChange}
      onConnect={handleConnect}
      isValidConnection={isValidConnection}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onEdgeDoubleClick={handleEdgeDoubleClick}
      onNodeDoubleClick={handleNodeDoubleClick}
      onMoveEnd={handleMoveEnd}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      defaultEdgeOptions={{
        type: 'data',
      }}
      // 匹配 UE5 蓝图行为：左键拖拽 = 框选，右键/中键拖拽 = 平移
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
  );
}

// ============================================================
// 导出组件
// ============================================================

/**
 * ReactFlow 编辑器包装组件
 * 
 * 使用 forwardRef 暴露命令式 API
 */
export const ReactFlowEditorWrapper = forwardRef<ReactFlowEditorHandle, ReactFlowEditorWrapperProps>(
  function ReactFlowEditorWrapper(props, ref) {
    const handleRef = useRef<ReactFlowEditorHandle | null>(null);
    
    useImperativeHandle(ref, () => ({
      setNodes: (nodes) => handleRef.current?.setNodes(nodes),
      setEdges: (edges) => handleRef.current?.setEdges(edges),
      setSelection: (selection) => handleRef.current?.setSelection(selection),
      setViewport: (viewport) => handleRef.current?.setViewport(viewport),
      fitView: (options) => handleRef.current?.fitView(options),
      getViewport: () => handleRef.current?.getViewport() ?? { x: 0, y: 0, zoom: 1 },
      screenToCanvas: (x, y) => handleRef.current?.screenToCanvas(x, y) ?? { x: 0, y: 0 },
    }));
    
    const handleHandle = useCallback((handle: ReactFlowEditorHandle) => {
      handleRef.current = handle;
    }, []);
    
    return <ReactFlowEditorInner {...props} onHandle={handleHandle} />;
  }
);

export default ReactFlowEditorWrapper;
