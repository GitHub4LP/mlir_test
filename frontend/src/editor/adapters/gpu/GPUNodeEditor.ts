/**
 * GPU 节点编辑器
 * 
 * 实现 INodeEditor 接口，使用 GPURenderer 进行渲染。
 * 集成 GraphController 处理业务逻辑。
 * 
 * 使用 UIManager 管理原生 Canvas UI 组件（TypeSelector 等）。
 */

import type { INodeEditor } from '../../INodeEditor';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
  NodeChange,
  EdgeChange,
} from '../../types';
import type { FunctionTrait } from '../../../types';
import type { TextRenderMode, EdgeRenderMode } from './GPURenderer';
import { GPURenderer } from './GPURenderer';
import { GraphController } from '../canvas/GraphController';
import { UIManager, type TypeSelectorState } from '../canvas/UIManager';
import type { TypeOption, ConstraintData } from '../canvas/ui/TypeSelector';
import { extendRenderData, type RenderExtensionConfig } from '../canvas/RenderExtensions';
import type { GraphState, GraphNode, GraphEdge, FunctionEntryData, FunctionReturnData } from '../../../types';
import { hitTestLayoutBox, parseInteractiveId } from '../../core/layout';

/**
 * 将 EditorNode 转换为 GraphNode
 */
function toGraphNode(node: EditorNode): GraphNode {
  return {
    id: node.id,
    type: node.type,
    position: { x: node.position.x, y: node.position.y },
    data: node.data as GraphNode['data'],
  };
}

/**
 * 将 EditorEdge 转换为 GraphEdge
 */
function toGraphEdge(edge: EditorEdge): GraphEdge {
  return {
    source: edge.source,
    sourceHandle: edge.sourceHandle,
    target: edge.target,
    targetHandle: edge.targetHandle,
  };
}

/**
 * GPU 节点编辑器
 */
export class GPUNodeEditor implements INodeEditor {
  private renderer: GPURenderer;
  private controller: GraphController;
  private uiManager: UIManager;
  private container: HTMLElement | null = null;
  
  // 当前数据
  private nodes: EditorNode[] = [];
  private edges: EditorEdge[] = [];
  private selection: EditorSelection = { nodeIds: [], edgeIds: [] };
  
  // 回调
  onNodesChange: ((changes: NodeChange[]) => void) | null = null;
  onEdgesChange: ((changes: EdgeChange[]) => void) | null = null;
  onSelectionChange: ((selection: EditorSelection) => void) | null = null;
  onViewportChange: ((viewport: EditorViewport) => void) | null = null;
  onConnect: ((request: ConnectionRequest) => void) | null = null;
  onNodeDoubleClick: ((nodeId: string) => void) | null = null;
  onEdgeDoubleClick: ((edgeId: string) => void) | null = null;
  onDrop: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null = null;
  onDeleteRequest: ((nodeIds: string[], edgeIds: string[]) => void) | null = null;
  
  // 业务事件回调
  onAttributeChange: ((nodeId: string, attributeName: string, value: string) => void) | null = null;
  onVariadicAdd: ((nodeId: string, groupName: string) => void) | null = null;
  onVariadicRemove: ((nodeId: string, groupName: string) => void) | null = null;
  onParameterAdd: ((functionId: string) => void) | null = null;
  onParameterRemove: ((functionId: string, parameterName: string) => void) | null = null;
  onParameterRename: ((functionId: string, oldName: string, newName: string) => void) | null = null;
  onReturnTypeAdd: ((functionId: string) => void) | null = null;
  onReturnTypeRemove: ((functionId: string, returnName: string) => void) | null = null;
  onReturnTypeRename: ((functionId: string, oldName: string, newName: string) => void) | null = null;
  onTraitsChange: ((functionId: string, traits: FunctionTrait[]) => void) | null = null;
  
  // 覆盖层回调
  onTypeSelectorChange: ((state: TypeSelectorState | null) => void) | null = null;
  onTypeLabelClick: ((nodeId: string, handleId: string, canvasX: number, canvasY: number) => void) | null = null;
  onParamNameClick: ((nodeId: string, paramIndex: number, currentName: string, canvasX: number, canvasY: number) => void) | null = null;
  onReturnNameClick: ((nodeId: string, returnIndex: number, currentName: string, canvasX: number, canvasY: number) => void) | null = null;
  onTraitsToggleClick: ((nodeId: string, canvasX: number, canvasY: number) => void) | null = null;

  // onNodeDataChange 已废弃：节点组件现在直接更新 editorStore
  onNodeDataChange: ((nodeId: string, data: Record<string, unknown>) => void) | null = null;

  // Hover 状态（用于显示删除按钮）
  private hoveredParamIndex: number | null = null;
  private hoveredReturnIndex: number | null = null;
  private hoveredNodeId: string | null = null;

  // Traits 展开状态
  private traitsExpandedMap: Map<string, boolean> = new Map();

  // Summary 展开状态
  private summaryExpandedMap: Map<string, boolean> = new Map();

  
  constructor(preferWebGPU: boolean = true) {
    this.renderer = new GPURenderer(preferWebGPU);
    this.controller = new GraphController();
    this.uiManager = new UIManager();
    
    // 设置 UIManager 回调
    this.uiManager.setCallbacks({
      onTypeSelect: () => {
        // 类型选择由外部通过 setTypeSelectCallback 处理
      },
      onTypeSelectorClose: () => {
        this.onTypeSelectorChange?.(null);
        this.requestRender();
      },
    });
    
    // 设置控制器回调
    this.setupControllerCallbacks();
    
    // 设置图数据提供者
    this.controller.setGraphDataProvider(() => this.getGraphState());
    
    // 设置渲染扩展回调
    this.controller.onExtendRenderData = (data, nodeLayouts) => {
      const nodesMap = new Map<string, GraphNode>();
      for (const node of this.nodes) {
        nodesMap.set(node.id, {
          id: node.id,
          type: node.type as GraphNode['type'],
          position: node.position,
          data: node.data as GraphNode['data'],
        });
      }
      
      const config: RenderExtensionConfig = {
        nodes: nodesMap,
        nodeList: this.nodes,
        edgeList: this.edges,
        hoveredNodeId: this.hoveredNodeId,
        hoveredParamIndex: this.hoveredParamIndex,
        hoveredReturnIndex: this.hoveredReturnIndex,
        traitsExpandedMap: this.traitsExpandedMap,
        summaryExpandedMap: this.summaryExpandedMap,
      };
      
      extendRenderData(data, nodeLayouts, config);
    };
    
    // 连接渲染器和控制器
    this.controller.setRenderer(this.renderer);
  }
  
  private setupControllerCallbacks(): void {
    // 节点位置变化
    this.controller.onNodePositionChange = (nodeId, x, y) => {
      const change: NodeChange = {
        type: 'position',
        id: nodeId,
        position: { x, y },
      };
      this.onNodesChange?.([change]);
    };
    
    // 连接尝试
    this.controller.onConnectionAttempt = (sourceNodeId, sourceHandleId, targetNodeId, targetHandleId) => {
      this.onConnect?.({
        source: sourceNodeId,
        sourceHandle: sourceHandleId,
        target: targetNodeId,
        targetHandle: targetHandleId,
      });
    };
    
    // 删除选中元素
    this.controller.onDeleteSelected = () => {
      const nodeIds = this.controller.getSelectedNodeIds();
      const edgeIds = this.controller.getSelectedEdgeIds();
      this.onDeleteRequest?.(nodeIds, edgeIds);
    };
    
    // 选择变化
    this.controller.onSelectionChange = (nodeIds, edgeIds) => {
      this.selection = { nodeIds, edgeIds };
      this.onSelectionChange?.(this.selection);
    };
    
    // 边双击
    this.controller.onEdgeDoubleClick = (edgeId) => {
      this.onEdgeDoubleClick?.(edgeId);
    };
    
    // 节点双击
    this.controller.onNodeDoubleClick = (nodeId) => {
      this.onNodeDoubleClick?.(nodeId);
    };
    
    // 视口变化
    this.controller.onViewportChange = (viewport) => {
      this.onViewportChange?.(viewport);
    };
    
    // Hover 状态更新回调
    this.controller.onHoverChange = (screenX, screenY) => {
      this.updateHoverState(screenX, screenY);
    };
    
    // 扩展命中测试回调（类型标签、按钮等）
    this.controller.onExtendedHitTest = (screenX, screenY) => {
      return this.handleExtendedHitTest(screenX, screenY);
    };
    
    // UI 事件路由（UI 组件优先）
    this.controller.onPreMouseDown = (screenX, screenY) => {
      return this.uiManager.handleMouseDown({ x: screenX, y: screenY, button: 0 });
    };
    
    this.controller.onPreMouseMove = (screenX, screenY) => {
      return this.uiManager.handleMouseMove({ x: screenX, y: screenY, button: 0 });
    };
    
    this.controller.onPreMouseUp = (screenX, screenY) => {
      return this.uiManager.handleMouseUp({ x: screenX, y: screenY, button: 0 });
    };
    
    this.controller.onPreWheel = (screenX, screenY, deltaX, deltaY) => {
      const handled = this.uiManager.handleWheel({ x: screenX, y: screenY, deltaX, deltaY });
      if (handled) {
        // UI 组件处理了滚轮事件，需要触发重新渲染
        this.controller.requestRender();
      }
      return handled;
    };
    
    this.controller.onPreKeyDown = (key, code, ctrlKey, shiftKey, altKey) => {
      return this.uiManager.handleKeyDown({ key, code, ctrlKey, shiftKey, altKey });
    };
    
    // 拖放
    this.controller.onDrop = (x, y, dataTransfer) => {
      this.onDrop?.(x, y, dataTransfer);
    };
  }
  
  private getGraphState(): GraphState {
    return {
      nodes: this.nodes.map(toGraphNode),
      edges: this.edges.map(toGraphEdge),
    };
  }
  
  // ============================================================
  // INodeEditor 实现
  // ============================================================
  
  mount(container: HTMLElement): void {
    this.container = container;
    this.renderer.mount(container);
    this.uiManager.mount(container);
    
    // 重新连接渲染器和控制器（确保事件回调正确绑定）
    this.controller.setRenderer(this.renderer);
    
    // 设置 UI 渲染回调
    this.renderer.setUIRenderCallback((ctx) => {
      this.uiManager.render(ctx);
    });
    
    // 设置拖放回调
    this.renderer.setDropCallback((x, y, dataTransfer) => {
      this.onDrop?.(x, y, dataTransfer);
    });
    
    // 等待渲染器就绪后再渲染
    this.renderer.waitForReady().then(() => {
      this.requestRender();
    });
  }
  
  unmount(): void {
    this.uiManager.unmount();
    this.uiManager.dispose();
    this.renderer.unmount();
    this.container = null;
  }
  
  setNodes(nodes: EditorNode[]): void {
    this.nodes = nodes;
    this.controller.clearLayoutCache();
    this.requestRender();
  }
  
  setEdges(edges: EditorEdge[]): void {
    this.edges = edges;
    this.requestRender();
  }

  setSelection(selection: EditorSelection): void {
    this.selection = selection;
    this.controller.syncSelectionFromExternal(selection.nodeIds);
    this.requestRender();
  }
  
  setViewport(viewport: EditorViewport): void {
    // 直接设置视口，不触发回调（避免循环）
    // 回调只在用户交互（拖拽、缩放）时触发
    this.controller.setViewportSilent({
      x: viewport.x,
      y: viewport.y,
      zoom: viewport.zoom,
    });
  }
  
  fitView(options?: { padding?: number; maxZoom?: number }): void {
    if (this.nodes.length === 0) return;
    
    // 计算所有节点的边界
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    
    for (const node of this.nodes) {
      const nodeData = node.data as Record<string, unknown> | undefined;
      const width = (typeof nodeData?.width === 'number' ? nodeData.width : 200);
      const height = (typeof nodeData?.height === 'number' ? nodeData.height : 100);
      minX = Math.min(minX, node.position.x);
      minY = Math.min(minY, node.position.y);
      maxX = Math.max(maxX, node.position.x + width);
      maxY = Math.max(maxY, node.position.y + height);
    }
    
    const padding = options?.padding ?? 50;
    const maxZoom = options?.maxZoom ?? 1;
    
    const contentWidth = maxX - minX + padding * 2;
    const contentHeight = maxY - minY + padding * 2;
    
    const containerWidth = this.container?.clientWidth ?? 800;
    const containerHeight = this.container?.clientHeight ?? 600;
    
    const zoom = Math.min(
      maxZoom,
      containerWidth / contentWidth,
      containerHeight / contentHeight
    );
    
    const x = (containerWidth - contentWidth * zoom) / 2 - (minX - padding) * zoom;
    const y = (containerHeight - contentHeight * zoom) / 2 - (minY - padding) * zoom;
    
    this.controller.setViewport({ x, y, zoom });
  }
  
  getViewport(): EditorViewport {
    const vp = this.controller.getViewport();
    return { x: vp.x, y: vp.y, zoom: vp.zoom };
  }
  
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number } {
    return this.controller.screenToCanvas(screenX, screenY);
  }
  
  getName(): string {
    return this.renderer.getName();
  }
  
  isAvailable(): boolean {
    return this.renderer.isAvailable();
  }

  /**
   * 等待渲染器就绪
   */
  waitForReady(): Promise<void> {
    return this.renderer.waitForReady();
  }
  
  private requestRender(): void {
    if (!this.renderer.isReady()) {
      // 渲染器还没准备好，等待就绪后再渲染
      this.renderer.waitForReady().then(() => {
        const renderData = this.controller.computeRenderData();
        this.renderer.render(renderData);
      });
      return;
    }
    const renderData = this.controller.computeRenderData();
    this.renderer.render(renderData);
  }

  // ============================================================
  // 类型选择器管理（原生 Canvas UI）
  // ============================================================

  /**
   * 显示类型选择器
   */
  showTypeSelector(
    nodeId: string,
    handleId: string,
    screenX: number,
    screenY: number,
    options: TypeOption[],
    currentType?: string,
    constraintData?: ConstraintData
  ): void {
    this.uiManager.showTypeSelector(nodeId, handleId, screenX, screenY, options, currentType, constraintData);
    this.onTypeSelectorChange?.(this.uiManager.getTypeSelectorState());
    this.requestRender();
  }

  /**
   * 隐藏类型选择器
   */
  hideTypeSelector(): void {
    this.uiManager.hideTypeSelector();
    this.onTypeSelectorChange?.(null);
    this.requestRender();
  }

  /**
   * 类型选择器是否可见
   */
  isTypeSelectorVisible(): boolean {
    return this.uiManager.isTypeSelectorVisible();
  }

  /**
   * 设置类型选择回调
   */
  setTypeSelectCallback(callback: (nodeId: string, handleId: string, type: string) => void): void {
    this.uiManager.setCallbacks({
      onTypeSelect: callback,
      onTypeSelectorClose: () => {
        this.onTypeSelectorChange?.(null);
        this.requestRender();
      },
    });
  }

  /**
   * 处理类型标签点击（供外部调用）
   */
  handleTypeLabelClick(screenX: number, screenY: number): boolean {
    return this.handleExtendedHitTest(screenX, screenY);
  }

  /**
   * 处理 Variadic 按钮点击（供外部调用）
   */
  handleVariadicButtonClick(screenX: number, screenY: number): boolean {
    return this.handleExtendedHitTest(screenX, screenY);
  }

  // ============================================================
  // 扩展命中测试处理
  // ============================================================

  /**
   * 处理扩展命中测试（基于 LayoutBox）
   * 返回是否命中了交互区域
   */
  handleExtendedHitTest(screenX: number, screenY: number): boolean {
    // 获取缓存的渲染数据中的 layoutBoxes
    const renderData = this.controller.getCachedRenderData();
    if (!renderData.layoutBoxes || renderData.layoutBoxes.size === 0) {
      return false;
    }
    
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    
    // 遍历所有节点的 LayoutBox 进行命中测试
    for (const [nodeId, layoutBox] of renderData.layoutBoxes) {
      // 计算相对于节点的坐标
      const localX = canvasPos.x - layoutBox.x;
      const localY = canvasPos.y - layoutBox.y;
      
      // 检查是否在节点范围内
      if (localX < 0 || localX > layoutBox.width || localY < 0 || localY > layoutBox.height) {
        continue;
      }
      
      // 使用 LayoutBox 命中测试
      const hit = hitTestLayoutBox(layoutBox, localX, localY);
      if (!hit || !hit.box.interactive?.id) {
        continue;
      }
      
      // 解析 interactive.id
      const parsed = parseInteractiveId(hit.box.interactive.id);
      
      // 找到对应的节点
      const node = this.nodes.find(n => n.id === nodeId);
      if (!node) continue;
      
      // 处理命中结果
      if (this.handleParsedHitResult(parsed, nodeId, node, canvasPos.x, canvasPos.y)) {
        return true;
      }
    }
    
    return false;
  }

  /**
   * 处理解析后的命中结果
   */
  private handleParsedHitResult(
    parsed: ReturnType<typeof parseInteractiveId>,
    nodeId: string,
    node: EditorNode,
    canvasX: number,
    canvasY: number
  ): boolean {
    switch (parsed.type) {
      case 'type-label':
        if (parsed.handleId) {
          this.onTypeLabelClick?.(nodeId, parsed.handleId, canvasX, canvasY);
          return true;
        }
        break;
        
      case 'variadic':
        if (parsed.group && parsed.action) {
          if (parsed.action === 'add') {
            this.onVariadicAdd?.(nodeId, parsed.group);
          } else {
            this.onVariadicRemove?.(nodeId, parsed.group);
          }
          return true;
        }
        break;
        
      case 'param-add':
        this.onParameterAdd?.(nodeId);
        return true;
        
      case 'param-remove':
        if (parsed.index !== undefined) {
          const entryData = node.data as FunctionEntryData;
          const paramName = entryData.outputs?.[parsed.index]?.name;
          if (paramName) {
            this.onParameterRemove?.(nodeId, paramName);
          }
          return true;
        }
        break;
        
      case 'param-name':
        if (parsed.index !== undefined) {
          const entryData = node.data as FunctionEntryData;
          const paramName = entryData.outputs?.[parsed.index]?.name ?? '';
          this.onParamNameClick?.(nodeId, parsed.index, paramName, canvasX, canvasY);
          return true;
        }
        break;
        
      case 'return-add':
        this.onReturnTypeAdd?.(nodeId);
        return true;
        
      case 'return-remove':
        if (parsed.index !== undefined) {
          const returnData = node.data as FunctionReturnData;
          const returnName = returnData.inputs?.[parsed.index]?.name;
          if (returnName) {
            this.onReturnTypeRemove?.(nodeId, returnName);
          }
          return true;
        }
        break;
        
      case 'return-name':
        if (parsed.index !== undefined) {
          const returnData = node.data as FunctionReturnData;
          const returnName = returnData.inputs?.[parsed.index]?.name ?? '';
          this.onReturnNameClick?.(nodeId, parsed.index, returnName, canvasX, canvasY);
          return true;
        }
        break;
        
      case 'traits-toggle':
        this.onTraitsToggleClick?.(nodeId, canvasX, canvasY);
        return true;
        
      case 'summary-toggle':
        this.toggleSummaryExpanded(nodeId);
        return true;
    }
    
    return false;
  }

  /**
   * 切换 Summary 展开状态
   */
  private toggleSummaryExpanded(nodeId: string): void {
    const current = this.summaryExpandedMap.get(nodeId) ?? false;
    this.summaryExpandedMap.set(nodeId, !current);
    this.requestRender();
  }

  /**
   * 更新 hover 状态（用于显示删除按钮）
   * 基于 LayoutBox 命中测试
   */
  updateHoverState(screenX: number, screenY: number): void {
    const renderData = this.controller.getCachedRenderData();
    if (!renderData.layoutBoxes || renderData.layoutBoxes.size === 0) {
      return;
    }
    
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    let newHoveredNodeId: string | null = null;
    let newHoveredParamIndex: number | null = null;
    let newHoveredReturnIndex: number | null = null;
    
    for (const [nodeId, layoutBox] of renderData.layoutBoxes) {
      const localX = canvasPos.x - layoutBox.x;
      const localY = canvasPos.y - layoutBox.y;
      
      if (localX < 0 || localX > layoutBox.width || localY < 0 || localY > layoutBox.height) {
        continue;
      }
      
      const hit = hitTestLayoutBox(layoutBox, localX, localY);
      if (!hit || !hit.box.interactive?.id) {
        continue;
      }
      
      const parsed = parseInteractiveId(hit.box.interactive.id);
      
      // 检查是否命中参数或返回值区域
      if (parsed.type === 'param-name' || parsed.type === 'param-remove') {
        newHoveredNodeId = nodeId;
        newHoveredParamIndex = parsed.index ?? null;
        break;
      } else if (parsed.type === 'return-name' || parsed.type === 'return-remove') {
        newHoveredNodeId = nodeId;
        newHoveredReturnIndex = parsed.index ?? null;
        break;
      }
    }
    
    // 检查是否有变化
    if (newHoveredNodeId !== this.hoveredNodeId ||
        newHoveredParamIndex !== this.hoveredParamIndex ||
        newHoveredReturnIndex !== this.hoveredReturnIndex) {
      this.hoveredNodeId = newHoveredNodeId;
      this.hoveredParamIndex = newHoveredParamIndex;
      this.hoveredReturnIndex = newHoveredReturnIndex;
      this.requestRender();
    }
  }

  /**
   * 设置 Traits 展开状态
   */
  setTraitsExpanded(nodeId: string, expanded: boolean): void {
    this.traitsExpandedMap.set(nodeId, expanded);
    this.requestRender();
  }

  /**
   * 获取当前文字渲染模式
   */
  getTextRenderMode(): TextRenderMode {
    return this.renderer.getTextRenderMode();
  }

  /**
   * 设置文字渲染模式
   * @param mode 'gpu' - GPU 纹理渲染, 'canvas' - Canvas 2D 渲染
   */
  setTextRenderMode(mode: TextRenderMode): void {
    this.renderer.setTextRenderMode(mode);
  }

  /**
   * 获取当前边渲染模式
   */
  getEdgeRenderMode(): EdgeRenderMode {
    return this.renderer.getEdgeRenderMode();
  }

  /**
   * 设置边渲染模式
   * @param mode 'gpu' - GPU 渲染, 'canvas' - Canvas 2D 渲染
   */
  setEdgeRenderMode(mode: EdgeRenderMode): void {
    this.renderer.setEdgeRenderMode(mode);
  }
}
