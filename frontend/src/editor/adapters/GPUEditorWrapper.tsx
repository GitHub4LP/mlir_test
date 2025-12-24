/**
 * GPU 编辑器 React 包装组件
 * 
 * 将 GPUNodeEditor 集成到 React 生命周期中。
 * 处理 canvas 创建、尺寸调整、数据同步。
 * 
 * 使用原生 Canvas UI 组件（TypeSelector 等），不使用 DOM overlay。
 */

import { useEffect, useRef, useState } from 'react';
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
import { getPortTypeInfo } from './shared/PortTypeInfo';
import { useTypeConstraintStore } from '../../stores/typeConstraintStore';
import { computeTypeSelectorData, computeTypeGroups } from '../../services/typeSelectorService';
import type { TypeOption } from './canvas/ui/TypeSelector';

export interface GPUEditorWrapperProps {
  nodes: EditorNode[];
  edges: EditorEdge[];
  selection?: EditorSelection;
  defaultViewport?: EditorViewport;
  preferWebGPU?: boolean;
  
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
  
  className?: string;
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
  onTypeSelect,
  className,
  style,
}: GPUEditorWrapperProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<GPUNodeEditor | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [backendName, setBackendName] = useState<string>('');
  
  // 用 ref 存储回调
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
  
  // 初始化编辑器
  useEffect(() => {
    if (!containerRef.current) return;
    
    const container = containerRef.current;
    const editor = new GPUNodeEditor(preferWebGPU);
    editorRef.current = editor;
    
    editor.mount(container);
    
    // 等待 GPU 后端真正就绪
    editor.waitForReady().then(() => {
      setBackendName(editor.getName());
      setIsReady(true);
      
      if (defaultViewport) {
        editor.setViewport(defaultViewport);
      }
      
      // 初始渲染
      if (nodes.length > 0) {
        editor.setNodes(nodes);
      }
      if (edges.length > 0) {
        editor.setEdges(edges);
      }
    });
    
    return () => {
      editor.unmount();
      editorRef.current = null;
      setIsReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preferWebGPU]);
  
  // 同步回调
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    
    editor.onNodesChange = callbacksRef.current.onNodesChange ?? null;
    editor.onEdgesChange = callbacksRef.current.onEdgesChange ?? null;
    editor.onSelectionChange = callbacksRef.current.onSelectionChange ?? null;
    editor.onViewportChange = (viewport) => {
      callbacksRef.current.onViewportChange?.(viewport);
    };
    editor.onConnect = callbacksRef.current.onConnect ?? null;
    editor.onNodeDoubleClick = callbacksRef.current.onNodeDoubleClick ?? null;
    editor.onEdgeDoubleClick = callbacksRef.current.onEdgeDoubleClick ?? null;
    editor.onDrop = callbacksRef.current.onDrop ?? null;
    editor.onDeleteRequest = callbacksRef.current.onDeleteRequest ?? null;
    
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
  }, [nodes]);
  
  const hasAppliedInitialViewportRef = useRef(false);
  
  // 同步节点数据
  useEffect(() => {
    if (editorRef.current && isReady) {
      editorRef.current.setNodes(nodes);
      
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
  
  // 同步外部视口变化
  useEffect(() => {
    if (editorRef.current && isReady && defaultViewport) {
      const current = editorRef.current.getViewport();
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
  
  return (
    <div
      className={`w-full h-full relative bg-gray-950 ${className ?? ''}`}
      style={style}
    >
      {/* Canvas 容器 */}
      <div
        ref={containerRef}
        style={{ width: '100%', height: '100%' }}
      />
      
      {/* 后端名称指示器 */}
      {backendName && (
        <div className="absolute bottom-2 left-2 text-xs text-gray-500 bg-gray-900/50 px-2 py-1 rounded pointer-events-none">
          {backendName} • Scroll to zoom • Middle-drag to pan • Click type labels to select
        </div>
      )}
    </div>
  );
}

export default GPUEditorWrapper;
