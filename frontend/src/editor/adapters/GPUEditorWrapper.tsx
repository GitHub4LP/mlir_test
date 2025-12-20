/**
 * GPU 编辑器 React 包装组件
 * 
 * 将 GPUNodeEditor 集成到 React 生命周期中。
 * 处理 canvas 创建、尺寸调整、数据同步。
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { GPUNodeEditor } from './gpu/GPUNodeEditor';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
  NodeChange,
  EdgeChange,
} from '../types';

export interface GPUEditorWrapperProps {
  /** 节点列表 */
  nodes: EditorNode[];
  /** 边列表 */
  edges: EditorEdge[];
  /** 选择状态 */
  selection?: EditorSelection;
  /** 初始视口 */
  defaultViewport?: EditorViewport;
  /** 是否优先使用 WebGPU */
  preferWebGPU?: boolean;
  
  // 回调
  onNodesChange?: (changes: NodeChange[]) => void;
  onEdgesChange?: (changes: EdgeChange[]) => void;
  onSelectionChange?: (selection: EditorSelection) => void;
  onViewportChange?: (viewport: EditorViewport) => void;
  onConnect?: (request: ConnectionRequest) => void;
  onNodeDoubleClick?: (nodeId: string) => void;
  onEdgeDoubleClick?: (edgeId: string) => void;
  onDrop?: (x: number, y: number, dataTransfer: DataTransfer) => void;
  onDeleteRequest?: (nodeIds: string[], edgeIds: string[]) => void;
  
  /** 自定义类名 */
  className?: string;
  /** 自定义样式 */
  style?: React.CSSProperties;
}


/**
 * GPU 编辑器包装组件
 */
export function GPUEditorWrapper({
  nodes,
  edges,
  selection,
  defaultViewport,
  preferWebGPU = true,
  onNodesChange,
  onEdgesChange,
  onSelectionChange,
  onViewportChange,
  onConnect,
  onNodeDoubleClick,
  onEdgeDoubleClick,
  onDrop,
  onDeleteRequest,
  className,
  style,
}: GPUEditorWrapperProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<GPUNodeEditor | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [backendName, setBackendName] = useState<string>('');
  
  // 初始化编辑器
  useEffect(() => {
    if (!containerRef.current) return;
    
    const container = containerRef.current;
    
    const editor = new GPUNodeEditor(preferWebGPU);
    editorRef.current = editor;
    
    // 挂载
    editor.mount(container);
    
    // 使用 requestAnimationFrame 延迟状态更新
    requestAnimationFrame(() => {
      setBackendName(editor.getName());
      setIsReady(true);
      
      // 首次就绪时应用初始视口
      if (defaultViewport) {
        editor.setViewport(defaultViewport);
      }
    });
    
    return () => {
      // 必须调用 unmount 清理 canvas 和事件监听器
      editor.unmount();
      editorRef.current = null;
      setIsReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preferWebGPU]); // 注意：不依赖 defaultViewport，只在初始化时使用
  
  // 同步回调
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    
    editor.onNodesChange = onNodesChange ?? null;
    editor.onEdgesChange = onEdgesChange ?? null;
    editor.onSelectionChange = onSelectionChange ?? null;
    editor.onViewportChange = onViewportChange ?? null;
    editor.onConnect = onConnect ?? null;
    editor.onNodeDoubleClick = onNodeDoubleClick ?? null;
    editor.onEdgeDoubleClick = onEdgeDoubleClick ?? null;
    editor.onDrop = onDrop ?? null;
    editor.onDeleteRequest = onDeleteRequest ?? null;
  }, [
    onNodesChange,
    onEdgesChange,
    onSelectionChange,
    onViewportChange,
    onConnect,
    onNodeDoubleClick,
    onEdgeDoubleClick,
    onDrop,
    onDeleteRequest,
  ]);
  
  // 跟踪是否已经应用过初始视口
  const hasAppliedInitialViewportRef = useRef(false);
  
  // 同步节点数据
  useEffect(() => {
    if (editorRef.current && isReady) {
      editorRef.current.setNodes(nodes);
      
      // 首次就绪时确保视口已应用
      if (!hasAppliedInitialViewportRef.current && defaultViewport) {
        hasAppliedInitialViewportRef.current = true;
        requestAnimationFrame(() => {
          editorRef.current?.setViewport(defaultViewport);
        });
      }
    }
  }, [nodes, isReady, defaultViewport]);
  
  // 同步边数据
  useEffect(() => {
    if (editorRef.current && isReady) {
      editorRef.current.setEdges(edges);
    }
  }, [edges, isReady]);
  
  // 同步外部视口变化（从 store 来的）
  useEffect(() => {
    if (editorRef.current && isReady && defaultViewport) {
      const current = editorRef.current.getViewport();
      // 只有当视口确实不同时才更新，避免循环
      if (Math.abs(current.x - defaultViewport.x) > 0.1 ||
          Math.abs(current.y - defaultViewport.y) > 0.1 ||
          Math.abs(current.zoom - defaultViewport.zoom) > 0.001) {
        editorRef.current.setViewport(defaultViewport);
      }
    }
  }, [defaultViewport, isReady]);
  
  // 同步选择状态
  useEffect(() => {
    if (editorRef.current && isReady && selection) {
      editorRef.current.setSelection(selection);
    }
  }, [selection, isReady]);
  
  // 暴露 fitView 方法
  const fitView = useCallback((options?: { padding?: number; maxZoom?: number }) => {
    editorRef.current?.fitView(options);
  }, []);
  
  // 暴露 getViewport 方法
  const getViewport = useCallback(() => {
    return editorRef.current?.getViewport() ?? { x: 0, y: 0, zoom: 1 };
  }, []);
  
  // 暴露 screenToCanvas 方法
  const screenToCanvas = useCallback((screenX: number, screenY: number) => {
    return editorRef.current?.screenToCanvas(screenX, screenY) ?? { x: 0, y: 0 };
  }, []);
  
  // 将方法暴露给父组件（通过 ref 或 context）
  // 这里暂时不使用，但保留以备将来使用
  void fitView;
  void getViewport;
  void screenToCanvas;
  
  return (
    <div
      className={`w-full h-full relative bg-gray-950 ${className ?? ''}`}
      style={style}
    >
      {/* Canvas 容器 - React 不会触碰这个 div 的内容 */}
      <div
        ref={containerRef}
        style={{ width: '100%', height: '100%' }}
      />
      {/* 后端名称指示器（调试用） */}
      {backendName && (
        <div
          style={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            padding: '2px 6px',
            fontSize: 10,
            color: 'rgba(255,255,255,0.5)',
            backgroundColor: 'rgba(0,0,0,0.3)',
            borderRadius: 4,
            pointerEvents: 'none',
            zIndex: 1000,
          }}
        >
          {backendName}
        </div>
      )}
    </div>
  );
}

export default GPUEditorWrapper;
