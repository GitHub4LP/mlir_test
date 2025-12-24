/**
 * Canvas 编辑器包装组件
 * 
 * 类似 ReactFlowEditorWrapper，提供 React 组件接口。
 * 内部使用 CanvasNodeEditor 实现 INodeEditor 接口。
 * 
 * 使用原生 Canvas UI 组件（TypeSelector 等），不使用 DOM overlay。
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
import { getPortTypeInfo } from './shared/PortTypeInfo';
import { useTypeConstraintStore } from '../../stores/typeConstraintStore';
import { computeTypeSelectorData, computeTypeGroups } from '../../services/typeSelectorService';
import type { TypeOption } from './canvas/ui/TypeSelector';

/** Canvas 编辑器包装组件 Props */
export interface CanvasEditorWrapperProps {
  nodes: EditorNode[];
  edges: EditorEdge[];
  defaultViewport?: EditorViewport;
  onNodesChange?: (changes: NodeChange[]) => void;
  onEdgesChange?: (changes: EdgeChange[]) => void;
  onSelectionChange?: (selection: EditorSelection) => void;
  onViewportChange?: (viewport: EditorViewport) => void;
  onConnect?: (request: ConnectionRequest) => void;
  onNodeDoubleClick?: (nodeId: string) => void;
  onEdgeDoubleClick?: (edgeId: string) => void;
  onDrop?: (x: number, y: number, dataTransfer: DataTransfer) => void;
  onDeleteRequest?: (nodeIds: string[], edgeIds: string[]) => void;
  onTypeSelect?: (nodeId: string, handleId: string, type: string) => void;
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
      defaultViewport,
      onNodesChange,
      onEdgesChange,
      onSelectionChange,
      onViewportChange,
      onConnect,
      onNodeDoubleClick,
      onEdgeDoubleClick,
      onDrop,
      onDeleteRequest,
      onTypeSelect,
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
      onTypeSelect,
    });
    
    // 用 ref 存储 nodes，供类型选择器回调使用
    const nodesRef = useRef(nodes);
    useEffect(() => {
      nodesRef.current = nodes;
    }, [nodes]);
    
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
        onTypeSelect,
      };
    }, [onNodesChange, onEdgesChange, onSelectionChange, onViewportChange, onConnect, onNodeDoubleClick, onEdgeDoubleClick, onDrop, onDeleteRequest, onTypeSelect]);

    const showPerformanceOverlay = useRendererStore(state => state.showPerformanceOverlay);
    const togglePerformanceOverlay = useRendererStore(state => state.togglePerformanceOverlay);

    // 初始化编辑器
    useEffect(() => {
      if (!containerRef.current || initializedRef.current) return;
      
      const editor = new CanvasNodeEditor();
      editorRef.current = editor;
      initializedRef.current = true;
      
      // 设置回调
      editor.onNodesChange = (changes) => callbacksRef.current.onNodesChange?.(changes);
      editor.onEdgesChange = (changes) => callbacksRef.current.onEdgesChange?.(changes);
      editor.onSelectionChange = (selection) => callbacksRef.current.onSelectionChange?.(selection);
      editor.onViewportChange = (viewport) => {
        callbacksRef.current.onViewportChange?.(viewport);
      };
      editor.onConnect = (request) => callbacksRef.current.onConnect?.(request);
      editor.onNodeDoubleClick = (nodeId) => callbacksRef.current.onNodeDoubleClick?.(nodeId);
      editor.onEdgeDoubleClick = (edgeId) => callbacksRef.current.onEdgeDoubleClick?.(edgeId);
      editor.onDrop = (x, y, dataTransfer) => callbacksRef.current.onDrop?.(x, y, dataTransfer);
      editor.onDeleteRequest = (nodeIds, edgeIds) => callbacksRef.current.onDeleteRequest?.(nodeIds, edgeIds);
      
      // 类型标签点击回调 - 使用原生 Canvas TypeSelector
      editor.onTypeLabelClick = (nodeId, handleId, canvasX, canvasY) => {
        const typeInfo = getPortTypeInfo(nodesRef.current, nodeId, handleId);
        if (!typeInfo) return;
        
        // 获取类型约束 store 数据
        const state = useTypeConstraintStore.getState();
        const { buildableTypes, constraintDefs, getConstraintElements, isShapedConstraint, getAllowedContainers } = state;
        
        // 计算类型选项
        const selectorData = computeTypeSelectorData({
          constraint: typeInfo.constraint,
          allowedTypes: typeInfo.allowedTypes,
          buildableTypes,
          constraintDefs,
          getConstraintElements,
          isShapedConstraint,
          getAllowedContainers,
        });
        
        // 计算类型分组
        const typeGroups = computeTypeGroups(
          selectorData,
          { searchText: '', showConstraints: true, showTypes: true, useRegex: false },
          typeInfo.constraint,
          buildableTypes,
          constraintDefs,
          getConstraintElements
        );
        
        // 转换为 TypeOption 格式
        const options: TypeOption[] = [];
        for (const group of typeGroups) {
          for (const item of group.items) {
            options.push({
              name: item,
              label: item,
              group: group.label,
            });
          }
        }
        
        // 转换画布坐标到屏幕坐标
        const viewport = editor.getViewport();
        const screenX = canvasX * viewport.zoom + viewport.x;
        const screenY = canvasY * viewport.zoom + viewport.y;
        
        // 显示原生 Canvas TypeSelector
        editor.showTypeSelector(nodeId, handleId, screenX, screenY, options, typeInfo.currentType);
      };
      
      // 设置类型选择回调
      editor.setTypeSelectCallback((nodeId, handleId, type) => {
        callbacksRef.current.onTypeSelect?.(nodeId, handleId, type);
      });
      
      editor.mount(containerRef.current);
      
      // 应用初始视口
      if (defaultViewport) {
        editor.setViewport(defaultViewport);
      }
      
      return () => {
        editor.unmount();
        editorRef.current = null;
        initializedRef.current = false;
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // 同步 nodes
    useEffect(() => {
      editorRef.current?.setNodes(nodes);
    }, [nodes]);

    // 同步 edges
    useEffect(() => {
      editorRef.current?.setEdges(edges);
    }, [edges]);

    // 同步外部视口变化
    useEffect(() => {
      if (defaultViewport && editorRef.current) {
        const current = editorRef.current.getViewport();
        if (Math.abs(current.x - defaultViewport.x) > 0.1 ||
            Math.abs(current.y - defaultViewport.y) > 0.1 ||
            Math.abs(current.zoom - defaultViewport.zoom) > 0.001) {
          editorRef.current.setViewport(defaultViewport);
        }
      }
    }, [defaultViewport]);

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
        <div className="absolute top-2 right-2 flex items-center gap-2 pointer-events-auto">
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
        <div className="absolute bottom-2 left-2 text-xs text-gray-500 bg-gray-900/50 px-2 py-1 rounded pointer-events-none">
          Canvas 2D • Scroll to zoom • Middle-drag to pan • Click type labels to select
        </div>
      </div>
    );
  }
);

export default CanvasEditorWrapper;
