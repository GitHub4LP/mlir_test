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
 * - 渲染类型选择器覆盖层（仅 ReactFlow/VueFlow）
 * - Canvas/GPU 渲染器使用原生 Canvas TypeSelector
 */

import { useEffect, useRef, useCallback, useReducer, useState } from 'react';
import { useEditorStore } from '../../core/stores/editorStore';
import type { INodeEditor } from '../../editor/INodeEditor';
import type {
  EditorViewport,
  EditorSelection,
  NodeChange,
  ConnectionRequest,
} from '../../editor/types';
import { EditorOverlay } from '../../editor/adapters/shared/EditorOverlay';
import { overlayReducer, type OverlayState } from '../../editor/adapters/shared/overlayTypes';
import { getPortTypeInfo } from '../../editor/adapters/shared/PortTypeInfo';
import { UnifiedTypeSelector } from '../../components/UnifiedTypeSelector';
import { useTypeConstraintStore } from '../../stores/typeConstraintStore';
import { computeTypeSelectorData, computeTypeGroups } from '../../services/typeSelectorService';
import { PerformanceOverlay } from '../../components/PerformanceOverlay';
import { MiniMap, type NodeSize } from '../../editor/adapters/shared/MiniMap';
import { performanceMonitor } from '../../editor/adapters/canvas/PerformanceMonitor';
import { computeNodeLayoutBox } from '../../editor/core/layout';
import type { TypeOption } from '../../editor/adapters/canvas/ui/TypeSelector';
import type { RendererType, CanvasBackendType } from '../../stores/rendererStore';
import type { EditorNode } from '../../editor/types';

export type { RendererType, CanvasBackendType };

export interface EditorContainerProps {
  /** 渲染器类型 */
  rendererType: RendererType;
  
  /** Canvas 图形后端（仅 rendererType='canvas' 时有效） */
  canvasBackend?: CanvasBackendType;
  
  /** 创建编辑器实例的工厂函数 */
  createEditor: (type: RendererType, canvasBackend?: CanvasBackendType) => INodeEditor | null;
  
  /** 连接请求处理（验证并添加边） */
  onConnect?: (request: ConnectionRequest) => void;
  
  /** 拖放处理（添加节点） */
  onDrop?: (x: number, y: number, dataTransfer: DataTransfer) => void;
  
  /** 删除请求处理 */
  onDeleteRequest?: (nodeIds: string[], edgeIds: string[]) => void;
  
  /** 边双击处理 */
  onEdgeDoubleClick?: (edgeId: string) => void;
  
  /** 选择变化处理 */
  onSelectionChange?: (selection: EditorSelection) => void;
  
  /** 视口变化处理 */
  onViewportChange?: (viewport: EditorViewport) => void;
  
  /** 编辑器就绪回调 */
  onEditorReady?: (editor: INodeEditor) => void;
  
  /** 类型选择回调 */
  onTypeSelect?: (nodeId: string, handleId: string, type: string) => void;
  
  /** 参数重命名回调 */
  onParameterRename?: (functionId: string, oldName: string, newName: string) => void;
  
  /** 返回值重命名回调 */
  onReturnTypeRename?: (functionId: string, oldName: string, newName: string) => void;
}

/**
 * 编辑器容器组件
 * 
 * 通过 INodeEditor 接口与编辑器交互，不直接依赖任何具体渲染器。
 */
export function EditorContainer({
  rendererType,
  canvasBackend,
  createEditor,
  onConnect,
  onDrop,
  onDeleteRequest,
  onEdgeDoubleClick,
  onSelectionChange: onSelectionChangeProp,
  onViewportChange: onViewportChangeProp,
  onEditorReady,
  onTypeSelect,
  onParameterRename,
  onReturnTypeRename,
}: EditorContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<INodeEditor | null>(null);
  
  // 覆盖层状态
  const [overlayState, dispatchOverlay] = useReducer(overlayReducer, null as OverlayState);
  const viewportRef = useRef<EditorViewport>({ x: 0, y: 0, zoom: 1 });
  
  // 使用 ref 存储回调，避免 useEffect 依赖变化导致编辑器重建
  const callbacksRef = useRef({
    onConnect,
    onDrop,
    onDeleteRequest,
    onEdgeDoubleClick,
    onSelectionChangeProp,
    onViewportChangeProp,
    onEditorReady,
    onTypeSelect,
    onParameterRename,
    onReturnTypeRename,
  });
  
  // 更新回调 ref
  useEffect(() => {
    callbacksRef.current = {
      onConnect,
      onDrop,
      onDeleteRequest,
      onEdgeDoubleClick,
      onSelectionChangeProp,
      onViewportChangeProp,
      onEditorReady,
      onTypeSelect,
      onParameterRename,
      onReturnTypeRename,
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
    viewportRef.current = newViewport;
    setViewport(newViewport);
    callbacksRef.current.onViewportChangeProp?.(newViewport);
  }, [setViewport]);
  
  // 类型标签点击处理
  // 对于 Canvas 渲染器，使用原生 Canvas TypeSelector
  // 对于 ReactFlow/VueFlow，使用 DOM overlay
  const handleTypeLabelClick = useCallback((nodeId: string, handleId: string, canvasX: number, canvasY: number) => {
    // 从节点的 data.portStates 检查端口是否可编辑（统一数据源）
    const state = useEditorStore.getState();
    const node = state.nodes.find(n => n.id === nodeId);
    const portStates = (node?.data as { portStates?: Record<string, { canEdit?: boolean }> })?.portStates;
    const portState = portStates?.[handleId];
    if (portState && !portState.canEdit) {
      // 不可编辑，不弹出选择器
      return;
    }
    
    const typeInfo = getPortTypeInfo(state.nodes, nodeId, handleId);
    if (!typeInfo) return;
    
    // Canvas 渲染器使用原生 Canvas UI
    if (rendererType === 'canvas') {
      const editor = editorRef.current;
      if (!editor) return;
      
      // 获取类型约束 store 数据
      const constraintState = useTypeConstraintStore.getState();
      const { buildableTypes, constraintDefs, getConstraintElements, isShapedConstraint, getAllowedContainers } = constraintState;
      
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
      // 需要调用编辑器的 showTypeSelector 方法
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const canvasEditor = editor as any;
      if (typeof canvasEditor.showTypeSelector === 'function') {
        canvasEditor.showTypeSelector(nodeId, handleId, screenX, screenY, options, typeInfo.currentType);
        
        // 设置类型选择回调
        if (typeof canvasEditor.setTypeSelectCallback === 'function') {
          canvasEditor.setTypeSelectCallback((nId: string, hId: string, type: string) => {
            callbacksRef.current.onTypeSelect?.(nId, hId, type);
          });
        }
      }
      return;
    }
    
    // ReactFlow/VueFlow 使用 DOM overlay
    dispatchOverlay({
      type: 'show-type-selector',
      payload: {
        nodeId,
        handleId,
        canvasX,
        canvasY,
        currentType: typeInfo.currentType,
        constraint: typeInfo.constraint,
      },
    });
  }, [rendererType]);
  
  // 处理类型选择
  const handleTypeSelect = useCallback((nodeId: string, handleId: string, type: string) => {
    callbacksRef.current.onTypeSelect?.(nodeId, handleId, type);
  }, []);

  // 参数名称点击处理（Canvas 渲染器）
  const handleParamNameClick = useCallback((nodeId: string, paramIndex: number, currentName: string, canvasX: number, canvasY: number) => {
    // 仅 Canvas 渲染器需要处理（ReactFlow/VueFlow 通过 DOMRenderer 直接处理）
    if (rendererType !== 'canvas') {
      return;
    }
    
    const editor = editorRef.current;
    if (!editor) return;
    
    // 转换画布坐标到屏幕坐标
    const viewport = editor.getViewport();
    const screenX = canvasX * viewport.zoom + viewport.x;
    const screenY = canvasY * viewport.zoom + viewport.y;
    
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const canvasEditor = editor as any;
    if (typeof canvasEditor.showEditableName === 'function') {
      // fieldId 格式: param-{index}
      const fieldId = `param-${paramIndex}`;
      canvasEditor.showEditableName(nodeId, fieldId, screenX, screenY, 80, currentName, 'Parameter name');
      
      // 设置名称提交回调
      if (typeof canvasEditor.setNameSubmitCallback === 'function') {
        canvasEditor.setNameSubmitCallback((nId: string, fId: string, newName: string) => {
          // 解析 fieldId 获取参数索引
          if (fId.startsWith('param-')) {
            // 从节点数据获取旧名称
            const state = useEditorStore.getState();
            const node = state.nodes.find(n => n.id === nId);
            if (node && node.data) {
              const outputs = (node.data as { outputs?: Array<{ name: string }> }).outputs;
              const idx = parseInt(fId.replace('param-', ''), 10);
              const oldName = outputs?.[idx]?.name;
              if (oldName && oldName !== newName) {
                // 获取 functionId（Entry 节点的 functionId）
                const functionId = (node.data as { functionId?: string }).functionId;
                if (functionId) {
                  callbacksRef.current.onParameterRename?.(functionId, oldName, newName);
                }
              }
            }
          }
        });
      }
    }
  }, [rendererType]);

  // 返回值名称点击处理（Canvas 渲染器）
  const handleReturnNameClick = useCallback((nodeId: string, returnIndex: number, currentName: string, canvasX: number, canvasY: number) => {
    // 仅 Canvas 渲染器需要处理
    if (rendererType !== 'canvas') {
      return;
    }
    
    const editor = editorRef.current;
    if (!editor) return;
    
    // 转换画布坐标到屏幕坐标
    const viewport = editor.getViewport();
    const screenX = canvasX * viewport.zoom + viewport.x;
    const screenY = canvasY * viewport.zoom + viewport.y;
    
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const canvasEditor = editor as any;
    if (typeof canvasEditor.showEditableName === 'function') {
      // fieldId 格式: return-{index}
      const fieldId = `return-${returnIndex}`;
      canvasEditor.showEditableName(nodeId, fieldId, screenX, screenY, 80, currentName, 'Return name');
      
      // 设置名称提交回调
      if (typeof canvasEditor.setNameSubmitCallback === 'function') {
        canvasEditor.setNameSubmitCallback((nId: string, fId: string, newName: string) => {
          // 解析 fieldId 获取返回值索引
          if (fId.startsWith('return-')) {
            // 从节点数据获取旧名称
            const state = useEditorStore.getState();
            const node = state.nodes.find(n => n.id === nId);
            if (node && node.data) {
              const inputs = (node.data as { inputs?: Array<{ name: string }> }).inputs;
              const idx = parseInt(fId.replace('return-', ''), 10);
              const oldName = inputs?.[idx]?.name;
              if (oldName && oldName !== newName) {
                // 获取 functionId（Return 节点的 functionId）
                const functionId = (node.data as { functionId?: string }).functionId;
                if (functionId) {
                  callbacksRef.current.onReturnTypeRename?.(functionId, oldName, newName);
                }
              }
            }
          }
        });
      }
    }
  }, [rendererType]);

  // 关闭覆盖层
  const handleCloseOverlay = useCallback(() => {
    dispatchOverlay({ type: 'close' });
  }, [dispatchOverlay]);
  
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
  
  // 创建/切换编辑器 - 依赖 rendererType 和 canvasBackend
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
    subContainer.style.position = 'absolute';
    subContainer.style.top = '0';
    subContainer.style.left = '0';
    containerRef.current.appendChild(subContainer);
    
    // 创建新编辑器
    const editor = createEditor(rendererType, canvasBackend);
    if (!editor) {
      console.warn(`EditorContainer: failed to create editor for type "${rendererType}" with backend "${canvasBackend}"`);
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
    editor.onTypeLabelClick = handleTypeLabelClick;
    
    // Canvas 渲染器特有的回调
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const canvasEditor = editor as any;
    // 使用 'in' 操作符检查属性是否存在（包括值为 null 的情况）
    if ('onParamNameClick' in canvasEditor) {
      canvasEditor.onParamNameClick = handleParamNameClick;
    }
    if ('onReturnNameClick' in canvasEditor) {
      canvasEditor.onReturnNameClick = handleReturnNameClick;
    }
    
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
  }, [rendererType, canvasBackend]); // 在 rendererType 或 canvasBackend 变化时重建编辑器
  
  // 从 store 订阅状态变化并同步到编辑器
  const nodes = useEditorStore(state => state.nodes);
  const edges = useEditorStore(state => state.edges);
  const viewport = useEditorStore(state => state.viewport);
  
  // 获取渲染器名称和是否支持 FPS
  const getRendererInfo = useCallback(() => {
    if (rendererType === 'reactflow') return { name: 'ReactFlow', supportsFps: false };
    if (rendererType === 'vueflow') return { name: 'VueFlow', supportsFps: false };
    if (rendererType === 'canvas') {
      switch (canvasBackend) {
        case 'webgl': return { name: 'Canvas (WebGL)', supportsFps: true };
        case 'webgpu': return { name: 'Canvas (WebGPU)', supportsFps: true };
        default: return { name: 'Canvas (2D)', supportsFps: true };
      }
    }
    return { name: 'Unknown', supportsFps: false };
  }, [rendererType, canvasBackend]);
  
  // 跟踪上次同步的节点 data，用于检测是否需要同步
  // 对于 ReactFlow/VueFlow，只在 data 变化时同步，避免 position 变化导致的循环更新
  const lastNodesDataRef = useRef<Map<string, unknown>>(new Map());
  
  // 同步节点到编辑器，并更新性能监控统计
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    
    // 更新性能监控统计
    const info = getRendererInfo();
    performanceMonitor.updateStats(nodes.length, edges.length, info.name, info.supportsFps);
    
    // Canvas 渲染器：直接同步所有数据（包括 position）
    // ReactFlow/VueFlow：只在 data 变化或节点增删时同步
    if (rendererType === 'canvas') {
      editor.setNodes(nodes);
      return;
    }
    
    // ReactFlow/VueFlow：检测是否有实质性变化
    let hasDataChanges = false;
    const currentDataMap = new Map<string, unknown>();
    
    for (const node of nodes) {
      currentDataMap.set(node.id, node.data);
    }
    
    // 检查节点数量变化
    if (currentDataMap.size !== lastNodesDataRef.current.size) {
      hasDataChanges = true;
    } else {
      // 检查 data 引用变化（类型传播会创建新的 data 对象）
      for (const [id, data] of currentDataMap) {
        if (lastNodesDataRef.current.get(id) !== data) {
          hasDataChanges = true;
          break;
        }
      }
    }
    
    if (hasDataChanges) {
      editor.setNodes(nodes);
      lastNodesDataRef.current = currentDataMap;
    }
  }, [nodes, edges, getRendererInfo, rendererType]);
  
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
  
  // FitView 处理
  const handleFitView = useCallback(() => {
    editorRef.current?.fitView();
  }, []);
  
  // 容器尺寸（用于 Canvas MiniMap）
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });
  
  // 监听容器尺寸变化
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    const updateSize = () => {
      setContainerSize({
        width: container.clientWidth,
        height: container.clientHeight,
      });
    };
    
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);
  
  // MiniMap 视口变更处理
  const handleMiniMapViewportChange = useCallback((newViewport: EditorViewport) => {
    editorRef.current?.setViewport(newViewport);
    useEditorStore.getState().setViewport(newViewport);
  }, []);
  
  // Canvas 渲染器的节点尺寸计算（使用 LayoutBox 系统）
  const getCanvasNodeSize = useCallback((node: EditorNode): NodeSize => {
    // 使用 LayoutBox 系统计算节点尺寸
    const layoutBox = computeNodeLayoutBox(node as Parameters<typeof computeNodeLayoutBox>[0], 0, 0);
    return {
      width: layoutBox.width,
      height: layoutBox.height,
    };
  }, []);
  
  // 转换节点格式用于 MiniMap
  const miniMapNodes = nodes.map(n => ({
    id: n.id,
    type: n.type,
    position: n.position,
    selected: n.selected,
    data: n.data,
  }));
  
  // Zoom 处理
  const handleZoomIn = useCallback(() => {
    const current = editorRef.current?.getViewport();
    if (current) {
      const newZoom = Math.min(current.zoom * 1.2, 4);
      editorRef.current?.setViewport({ ...current, zoom: newZoom });
    }
  }, []);
  
  const handleZoomOut = useCallback(() => {
    const current = editorRef.current?.getViewport();
    if (current) {
      const newZoom = Math.max(current.zoom / 1.2, 0.1);
      editorRef.current?.setViewport({ ...current, zoom: newZoom });
    }
  }, []);
  
  return (
    <div className="w-full h-full relative overflow-hidden">
      <div 
        ref={containerRef} 
        className="w-full h-full relative"
      />
      
      {/* 性能监控覆盖层（右上角） */}
      <PerformanceOverlay />
      
      {/* Canvas 渲染器的工具区域 */}
      {rendererType === 'canvas' && (
        <>
          {/* 工具按钮（左下角） */}
          <div className="absolute bottom-[10px] left-[10px] flex flex-col bg-[#1f2937] rounded border border-gray-600 pointer-events-auto z-10">
            <button
              onClick={handleZoomIn}
              className="w-7 h-7 flex items-center justify-center text-gray-400 hover:bg-gray-600 border-b border-gray-600 text-lg"
              title="Zoom in"
            >
              +
            </button>
            <button
              onClick={handleZoomOut}
              className="w-7 h-7 flex items-center justify-center text-gray-400 hover:bg-gray-600 border-b border-gray-600 text-lg"
              title="Zoom out"
            >
              −
            </button>
            <button
              onClick={handleFitView}
              className="w-7 h-7 flex items-center justify-center text-gray-400 hover:bg-gray-600 text-sm"
              title="Fit view"
            >
              ⊞
            </button>
          </div>
          
          {/* MiniMap（右下角） */}
          <div className="absolute bottom-[10px] right-[10px] bg-[#1f2937] rounded border border-gray-600 overflow-hidden pointer-events-auto z-10">
            <MiniMap
              nodes={miniMapNodes}
              viewport={viewport}
              containerWidth={containerSize.width}
              containerHeight={containerSize.height}
              width={200}
              height={150}
              onViewportChange={handleMiniMapViewportChange}
              getNodeSize={getCanvasNodeSize}
            />
          </div>
        </>
      )}
      
      {/* 统一覆盖层 */}
      <EditorOverlay
        state={overlayState}
        viewport={viewportRef.current}
        onClose={handleCloseOverlay}
        onTypeSelect={handleTypeSelect}
        renderTypeSelector={({ state, onSelect }) => (
          <div className="bg-gray-800 border border-gray-600 rounded shadow-xl p-1">
            <UnifiedTypeSelector
              selectedType={state.currentType}
              onTypeSelect={onSelect}
              constraint={state.constraint}
              allowedTypes={state.allowedTypes}
            />
          </div>
        )}
      />
    </div>
  );
}

export default EditorContainer;
