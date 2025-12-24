/**
 * ReactFlowCanvas
 * 
 * React Flow 画布组件，供 ReactFlowNodeEditor 内部使用。
 * 封装 React Flow 的所有交互逻辑。
 */

import { useCallback, useEffect, useRef } from 'react';
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
  type NodeChange as RFNodeChange,
  type EdgeChange as RFEdgeChange,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  NodeChange,
  EdgeChange,
  ConnectionRequest,
} from '../../types';
import { nodeTypes } from './nodes';
import { edgeTypes } from './edges';
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
} from '../ReactFlowAdapter';

// ============================================================
// 类型定义
// ============================================================

export interface ReactFlowCanvasHandle {
  setNodes(nodes: EditorNode[]): void;
  setEdges(edges: EditorEdge[]): void;
  setSelection(selection: EditorSelection): void;
  setViewport(viewport: EditorViewport): void;
  fitView(options?: { padding?: number; maxZoom?: number }): void;
  getViewport(): EditorViewport;
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number };
}

export interface ReactFlowCanvasProps {
  initialNodes?: EditorNode[];
  initialEdges?: EditorEdge[];
  initialViewport?: EditorViewport;
  onReady?: (handle: ReactFlowCanvasHandle) => void;
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

// ============================================================
// 内部组件（需要 useReactFlow）
// ============================================================

interface InnerProps extends ReactFlowCanvasProps {
  onHandle: (handle: ReactFlowCanvasHandle) => void;
}

function ReactFlowCanvasInner({
  initialNodes = [],
  initialEdges = [],
  initialViewport,
  onHandle,
  onNodesChange,
  onEdgesChange,
  onSelectionChange,
  onViewportChange,
  onConnect,
  onNodeDoubleClick,
  onEdgeDoubleClick,
  onDrop,
  onDeleteRequest,
}: InnerProps) {
  const [nodes, setNodes, onRFNodesChange] = useNodesState<Node>(
    initialNodes.map(toReactFlowNode)
  );
  const [edges, setEdges, onRFEdgesChange] = useEdgesState<Edge>(
    initialEdges.map(toReactFlowEdge)
  );
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
    const handle: ReactFlowCanvasHandle = {
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
    onRFEdgesChange(changes);
    
    const editorChanges = convertEdgeChanges(changes);
    if (editorChanges.length > 0) {
      callbacksRef.current.onEdgesChange?.(editorChanges);
    }
    
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
        selectionOnDrag
        panOnDrag={[2]}
        selectionMode={SelectionMode.Partial}
        selectNodesOnDrag={false}
        edgesReconnectable
        minZoom={0.1}
        maxZoom={2}
        defaultViewport={initialViewport ?? { x: 0, y: 0, zoom: 0.8 }}
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
 * ReactFlow 画布组件
 * 
 * 包装在 ReactFlowProvider 中，通过 onReady 回调暴露命令式 API
 */
export function ReactFlowCanvas({ onReady, ...restProps }: ReactFlowCanvasProps) {
  const handleRef = useRef<ReactFlowCanvasHandle | null>(null);
  
  const handleHandle = useCallback((handle: ReactFlowCanvasHandle) => {
    handleRef.current = handle;
    onReady?.(handle);
  }, [onReady]);
  
  return (
    <ReactFlowProvider>
      <ReactFlowCanvasInner 
        {...restProps} 
        onReady={onReady} 
        onHandle={handleHandle}
      />
    </ReactFlowProvider>
  );
}

export default ReactFlowCanvas;
