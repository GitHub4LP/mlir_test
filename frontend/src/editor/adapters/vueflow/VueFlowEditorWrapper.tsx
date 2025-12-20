/**
 * Vue Flow 编辑器 React 包装组件
 * 
 * 直接使用 VueFlowBridge，不通过 INodeEditor 接口。
 * 这样避免了在 React 渲染期间创建/销毁独立 React Root 的问题。
 */

import { useEffect, useRef, useCallback } from 'react';
import { VueFlowBridge, type VueFlowBridgeHandle } from './VueFlowBridge';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
  NodeChange,
} from '../../types';

export interface VueFlowEditorWrapperProps {
  /** 节点列表 */
  nodes: EditorNode[];
  /** 边列表 */
  edges: EditorEdge[];
  /** 选择状态 */
  selection?: EditorSelection;
  /** 初始视口 */
  defaultViewport?: EditorViewport;
  
  // 回调
  onNodesChange?: (changes: NodeChange[]) => void;
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
 * Vue Flow 编辑器包装组件
 * 
 * 直接渲染 VueFlowBridge，避免 INodeEditor 的 createRoot 问题。
 */
export function VueFlowEditorWrapper({
  nodes,
  edges,
  selection,
  defaultViewport,
  onNodesChange,
  onSelectionChange,
  onViewportChange,
  onConnect,
  onNodeDoubleClick,
  onEdgeDoubleClick,
  onDrop,
  onDeleteRequest,
  className,
  style,
}: VueFlowEditorWrapperProps) {
  const bridgeRef = useRef<VueFlowBridgeHandle>(null);
  
  // 使用 defaultViewport 作为初始值，后续通过 bridge 同步
  const initialViewport = defaultViewport || { x: 0, y: 0, zoom: 1 };
  
  // 同步外部视口变化到 bridge（不更新本地状态，直接传递给 bridge）
  useEffect(() => {
    if (defaultViewport) {
      bridgeRef.current?.setViewport(defaultViewport);
    }
  }, [defaultViewport]);
  
  // 处理视口变化
  const handleViewportChange = useCallback((viewport: EditorViewport) => {
    onViewportChange?.(viewport);
  }, [onViewportChange]);
  
  // 暴露 fitView 方法
  const fitView = useCallback(() => {
    bridgeRef.current?.fitView();
  }, []);
  
  void fitView; // 保留以备将来使用
  
  return (
    <div
      className={`w-full h-full relative bg-gray-950 ${className ?? ''}`}
      style={style}
    >
      <VueFlowBridge
        ref={bridgeRef}
        nodes={nodes}
        edges={edges}
        viewport={initialViewport}
        selection={selection}
        onNodesChange={onNodesChange}
        onSelectionChange={onSelectionChange}
        onViewportChange={handleViewportChange}
        onConnect={onConnect}
        onNodeDoubleClick={onNodeDoubleClick}
        onEdgeDoubleClick={onEdgeDoubleClick}
        onDrop={onDrop}
        onDeleteRequest={onDeleteRequest}
      />
      {/* 后端名称指示器 */}
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
        Vue Flow
      </div>
    </div>
  );
}

export default VueFlowEditorWrapper;
