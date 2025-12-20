/**
 * Canvas 编辑器包装组件
 * 
 * 类似 ReactFlowEditorWrapper，提供 React 组件接口。
 * 内部使用 CanvasNodeEditor 实现 INodeEditor 接口。
 */

import { useEffect, useRef, useImperativeHandle, forwardRef, useCallback } from 'react';
import { CanvasNodeEditor } from './CanvasNodeEditor';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
  NodeChange,
  EdgeChange,
} from '../types';
import { PerformanceOverlay } from '../../components/PerformanceOverlay';
import { useRendererStore } from '../../stores/rendererStore';

/** Canvas 编辑器包装组件 Props */
export interface CanvasEditorWrapperProps {
  /** 初始节点 */
  nodes: EditorNode[];
  /** 初始边 */
  edges: EditorEdge[];
  /** 节点变更回调 */
  onNodesChange?: (changes: NodeChange[]) => void;
  /** 边变更回调 */
  onEdgesChange?: (changes: EdgeChange[]) => void;
  /** 选择变更回调 */
  onSelectionChange?: (selection: EditorSelection) => void;
  /** 视口变更回调 */
  onViewportChange?: (viewport: EditorViewport) => void;
  /** 连接请求回调 */
  onConnect?: (request: ConnectionRequest) => void;
  /** 节点双击回调 */
  onNodeDoubleClick?: (nodeId: string) => void;
  /** 边双击回调 */
  onEdgeDoubleClick?: (edgeId: string) => void;
  /** 拖放回调 */
  onDrop?: (x: number, y: number, dataTransfer: DataTransfer) => void;
  /** 删除请求回调 */
  onDeleteRequest?: (nodeIds: string[], edgeIds: string[]) => void;
}

/** Canvas 编辑器命令式 API */
export interface CanvasEditorHandle {
  setNodes(nodes: EditorNode[]): void;
  setEdges(edges: EditorEdge[]): void;
  setSelection(selection: EditorSelection): void;
  setViewport(viewport: EditorViewport): void;
  fitView(options?: { padding?: number; maxZoom?: number }): void;
  getViewport(): EditorViewport;
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number };
}

/**
 * Canvas 编辑器包装组件
 */
export const CanvasEditorWrapper = forwardRef<CanvasEditorHandle, CanvasEditorWrapperProps>(
  function CanvasEditorWrapper(props, ref) {
    const {
      nodes,
      edges,
      onNodesChange,
      onEdgesChange,
      onSelectionChange,
      onViewportChange,
      onConnect,
      onNodeDoubleClick,
      onEdgeDoubleClick,
      onDrop,
      onDeleteRequest,
    } = props;

    const containerRef = useRef<HTMLDivElement>(null);
    const editorRef = useRef<CanvasNodeEditor | null>(null);
    const initializedRef = useRef(false);
    
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
    
    // 更新回调 ref
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

    const showPerformanceOverlay = useRendererStore(state => state.showPerformanceOverlay);
    const togglePerformanceOverlay = useRendererStore(state => state.togglePerformanceOverlay);

    // 初始化编辑器
    useEffect(() => {
      if (!containerRef.current || initializedRef.current) return;
      
      const editor = new CanvasNodeEditor();
      editorRef.current = editor;
      initializedRef.current = true;
      
      // 设置回调（使用 ref 包装）
      editor.onNodesChange = (changes) => callbacksRef.current.onNodesChange?.(changes);
      editor.onEdgesChange = (changes) => callbacksRef.current.onEdgesChange?.(changes);
      editor.onSelectionChange = (selection) => callbacksRef.current.onSelectionChange?.(selection);
      editor.onViewportChange = (viewport) => callbacksRef.current.onViewportChange?.(viewport);
      editor.onConnect = (request) => callbacksRef.current.onConnect?.(request);
      editor.onNodeDoubleClick = (nodeId) => callbacksRef.current.onNodeDoubleClick?.(nodeId);
      editor.onEdgeDoubleClick = (edgeId) => callbacksRef.current.onEdgeDoubleClick?.(edgeId);
      editor.onDrop = (x, y, dataTransfer) => callbacksRef.current.onDrop?.(x, y, dataTransfer);
      editor.onDeleteRequest = (nodeIds, edgeIds) => callbacksRef.current.onDeleteRequest?.(nodeIds, edgeIds);
      
      editor.mount(containerRef.current);
      
      return () => {
        editor.unmount();
        editorRef.current = null;
        initializedRef.current = false;
      };
    }, []);

    // 同步 nodes
    useEffect(() => {
      editorRef.current?.setNodes(nodes);
    }, [nodes]);

    // 同步 edges
    useEffect(() => {
      editorRef.current?.setEdges(edges);
    }, [edges]);

    // 暴露命令式 API
    useImperativeHandle(ref, () => ({
      setNodes: (nodes) => editorRef.current?.setNodes(nodes),
      setEdges: (edges) => editorRef.current?.setEdges(edges),
      setSelection: (selection) => editorRef.current?.setSelection(selection),
      setViewport: (viewport) => editorRef.current?.setViewport(viewport),
      fitView: (options) => editorRef.current?.fitView(options),
      getViewport: () => editorRef.current?.getViewport() ?? { x: 0, y: 0, zoom: 1 },
      screenToCanvas: (x, y) => editorRef.current?.screenToCanvas(x, y) ?? { x: 0, y: 0 },
    }), []);

    const handleFitView = useCallback(() => {
      editorRef.current?.fitView();
    }, []);

    return (
      <div className="w-full h-full relative bg-gray-950">
        {/* Canvas 容器 */}
        <div ref={containerRef} className="w-full h-full" />
        
        {/* 工具栏 */}
        <div className="absolute top-2 right-2 flex items-center gap-2">
          {/* 性能监控开关 */}
          <button
            onClick={togglePerformanceOverlay}
            className={`text-xs px-2 py-1 rounded border transition-colors ${
              showPerformanceOverlay
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-gray-800/80 border-gray-600 text-gray-400 hover:text-white'
            }`}
            title="Toggle performance overlay"
          >
            📊
          </button>
          {/* 适应视口 */}
          <button
            onClick={handleFitView}
            className="text-xs px-2 py-1 rounded border bg-gray-800/80 border-gray-600 text-gray-400 hover:text-white transition-colors"
            title="Fit view"
          >
            ⊞
          </button>
        </div>
        
        {/* 性能监控覆盖层 */}
        {showPerformanceOverlay && <PerformanceOverlay />}
        
        {/* 提示信息 */}
        <div className="absolute bottom-2 left-2 text-xs text-gray-500 bg-gray-900/50 px-2 py-1 rounded">
          Canvas 2D • Scroll to zoom • Middle-drag to pan • Drag nodes to move
        </div>
      </div>
    );
  }
);

export default CanvasEditorWrapper;
