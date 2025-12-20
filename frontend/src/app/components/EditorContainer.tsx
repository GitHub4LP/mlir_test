/**
 * EditorContainer 组件
 * 
 * 编辑器容器，通过 INodeEditor 接口与各种渲染器交互。
 * 此组件不直接导入 React Flow，实现 app 层与渲染器的解耦。
 * 
 * 职责：
 * - 管理编辑器实例的生命周期
 * - 同步 editorStore 状态到编辑器
 * - 处理编辑器事件并更新 store
 */

import { useEffect, useRef, useCallback } from 'react';
import { useEditorStore } from '../../core/stores/editorStore';
import type { INodeEditor } from '../../editor/INodeEditor';
import type {
  EditorViewport,
  EditorSelection,
  NodeChange,
  ConnectionRequest,
} from '../../editor/types';

export type RendererType = 'reactflow' | 'canvas' | 'webgl' | 'webgpu' | 'vueflow';

export interface EditorContainerProps {
  /** 渲染器类型 */
  rendererType: RendererType;
  
  /** 创建编辑器实例的工厂函数 */
  createEditor: (type: RendererType) => INodeEditor | null;
  
  /** 连接请求处理（验证并添加边） */
  onConnect?: (request: ConnectionRequest) => void;
  
  /** 拖放处理（添加节点） */
  onDrop?: (x: number, y: number, dataTransfer: DataTransfer) => void;
  
  /** 删除请求处理 */
  onDeleteRequest?: (nodeIds: string[], edgeIds: string[]) => void;
  
  /** 边双击处理 */
  onEdgeDoubleClick?: (edgeId: string) => void;
  
  /** 节点双击处理 */
  onNodeDoubleClick?: (nodeId: string) => void;
  
  /** 选择变化处理 */
  onSelectionChange?: (selection: EditorSelection) => void;
  
  /** 视口变化处理 */
  onViewportChange?: (viewport: EditorViewport) => void;
  
  /** 编辑器就绪回调 */
  onEditorReady?: (editor: INodeEditor) => void;
}

/**
 * 编辑器容器组件
 * 
 * 通过 INodeEditor 接口与编辑器交互，不直接依赖任何具体渲染器。
 */
export function EditorContainer({
  rendererType,
  createEditor,
  onConnect,
  onDrop,
  onDeleteRequest,
  onEdgeDoubleClick,
  onNodeDoubleClick,
  onSelectionChange: onSelectionChangeProp,
  onViewportChange: onViewportChangeProp,
  onEditorReady,
}: EditorContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<INodeEditor | null>(null);
  
  // 使用 ref 存储回调，避免 useEffect 依赖变化导致编辑器重建
  const callbacksRef = useRef({
    onConnect,
    onDrop,
    onDeleteRequest,
    onEdgeDoubleClick,
    onNodeDoubleClick,
    onSelectionChangeProp,
    onViewportChangeProp,
    onEditorReady,
  });
  
  // 更新回调 ref
  useEffect(() => {
    callbacksRef.current = {
      onConnect,
      onDrop,
      onDeleteRequest,
      onEdgeDoubleClick,
      onNodeDoubleClick,
      onSelectionChangeProp,
      onViewportChangeProp,
      onEditorReady,
    };
  });
  
  // 从 store 获取 actions（这些是稳定的）
  const applyNodeChanges = useEditorStore(state => state.applyNodeChanges);
  const setSelection = useEditorStore(state => state.setSelection);
  const setViewport = useEditorStore(state => state.setViewport);
  
  // 稳定的事件处理函数（通过 ref 访问最新回调）
  const handleNodesChange = useCallback((changes: NodeChange[]) => {
    applyNodeChanges(changes);
  }, [applyNodeChanges]);
  
  const handleSelectionChange = useCallback((selection: EditorSelection) => {
    setSelection(selection);
    callbacksRef.current.onSelectionChangeProp?.(selection);
  }, [setSelection]);
  
  const handleViewportChange = useCallback((newViewport: EditorViewport) => {
    setViewport(newViewport);
    callbacksRef.current.onViewportChangeProp?.(newViewport);
  }, [setViewport]);
  
  const handleConnect = useCallback((request: ConnectionRequest) => {
    callbacksRef.current.onConnect?.(request);
  }, []);
  
  const handleDrop = useCallback((x: number, y: number, dataTransfer: DataTransfer) => {
    callbacksRef.current.onDrop?.(x, y, dataTransfer);
  }, []);
  
  const handleDeleteRequest = useCallback((nodeIds: string[], edgeIds: string[]) => {
    callbacksRef.current.onDeleteRequest?.(nodeIds, edgeIds);
  }, []);
  
  const handleEdgeDoubleClick = useCallback((edgeId: string) => {
    callbacksRef.current.onEdgeDoubleClick?.(edgeId);
  }, []);
  
  const handleNodeDoubleClick = useCallback((nodeId: string) => {
    callbacksRef.current.onNodeDoubleClick?.(nodeId);
  }, []);
  
  // 创建/切换编辑器 - 只依赖 rendererType
  useEffect(() => {
    if (!containerRef.current) return;
    
    // 卸载旧编辑器
    if (editorRef.current) {
      editorRef.current.unmount();
      editorRef.current = null;
    }
    
    // 清空容器内容
    containerRef.current.innerHTML = '';
    
    // 创建新的子容器（避免 createRoot 冲突）
    const subContainer = document.createElement('div');
    subContainer.style.width = '100%';
    subContainer.style.height = '100%';
    containerRef.current.appendChild(subContainer);
    
    // 创建新编辑器
    const editor = createEditor(rendererType);
    if (!editor) {
      console.warn(`EditorContainer: failed to create editor for type "${rendererType}"`);
      return;
    }
    
    editorRef.current = editor;
    
    // 绑定事件回调
    editor.onNodesChange = handleNodesChange;
    editor.onSelectionChange = handleSelectionChange;
    editor.onViewportChange = handleViewportChange;
    editor.onConnect = handleConnect;
    editor.onDrop = handleDrop;
    editor.onDeleteRequest = handleDeleteRequest;
    editor.onEdgeDoubleClick = handleEdgeDoubleClick;
    editor.onNodeDoubleClick = handleNodeDoubleClick;
    
    // 挂载到子容器
    editor.mount(subContainer);
    
    // 同步当前数据到新编辑器
    const state = useEditorStore.getState();
    editor.setNodes(state.nodes);
    editor.setEdges(state.edges);
    if (state.viewport.x !== 0 || state.viewport.y !== 0 || state.viewport.zoom !== 1) {
      editor.setViewport(state.viewport);
    }
    // 同步选择状态
    if (state.selection.nodeIds.length > 0 || state.selection.edgeIds.length > 0) {
      editor.setSelection(state.selection);
    }
    
    // 通知就绪
    callbacksRef.current.onEditorReady?.(editor);
    
    return () => {
      if (editorRef.current) {
        editorRef.current.unmount();
        editorRef.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rendererType]); // 只在 rendererType 变化时重建编辑器
  
  // 从 store 订阅状态变化并同步到编辑器
  const nodes = useEditorStore(state => state.nodes);
  const edges = useEditorStore(state => state.edges);
  const viewport = useEditorStore(state => state.viewport);
  
  // 同步节点到编辑器
  useEffect(() => {
    editorRef.current?.setNodes(nodes);
  }, [nodes]);
  
  // 同步边到编辑器
  useEffect(() => {
    editorRef.current?.setEdges(edges);
  }, [edges]);
  
  // 同步视口到编辑器（仅当外部改变时）
  const lastViewportRef = useRef(viewport);
  useEffect(() => {
    const last = lastViewportRef.current;
    const diff = Math.abs(viewport.x - last.x) + Math.abs(viewport.y - last.y) + Math.abs(viewport.zoom - last.zoom);
    if (diff > 0.01) {
      editorRef.current?.setViewport(viewport);
      lastViewportRef.current = viewport;
    }
  }, [viewport]);
  
  return (
    <div 
      ref={containerRef} 
      className="w-full h-full"
      style={{ minHeight: '100%' }}
    />
  );
}

export default EditorContainer;
