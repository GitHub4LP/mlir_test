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
import type { TypeOption } from '../canvas/ui/TypeSelector';
import { 
  hitTestTypeLabel, 
  hitTestVariadicButton, 
  hitTestNodeExtended,
  computeHoveredIndex,
  type HitResult,
  type EntryNodeHitData,
  type ReturnNodeHitData,
  type NodeType,
} from '../canvas/HitTest';
import { computeNodeLayout } from '../canvas/layout';
import { extendRenderData, type RenderExtensionConfig } from '../canvas/RenderExtensions';
import type { GraphState, GraphNode, GraphEdge, BlueprintNodeData, FunctionEntryData, FunctionReturnData } from '../../../types';

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
        hoveredNodeId: this.hoveredNodeId,
        hoveredParamIndex: this.hoveredParamIndex,
        hoveredReturnIndex: this.hoveredReturnIndex,
        traitsExpandedMap: this.traitsExpandedMap,
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
    
    // 视口变化
    this.controller.onViewportChange = (viewport) => {
      this.onViewportChange?.(viewport);
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
      return this.uiManager.handleWheel({ x: screenX, y: screenY, deltaX, deltaY });
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
      console.log('GPUNodeEditor.requestRender: renderer not ready, waiting...');
      this.renderer.waitForReady().then(() => {
        console.log('GPUNodeEditor.requestRender: renderer ready, rendering with', this.nodes.length, 'nodes');
        const renderData = this.controller.computeRenderData();
        console.log('GPUNodeEditor.requestRender: renderData has', renderData.rects.length, 'rects');
        this.renderer.render(renderData);
      });
      return;
    }
    console.log('GPUNodeEditor.requestRender: rendering with', this.nodes.length, 'nodes');
    const renderData = this.controller.computeRenderData();
    console.log('GPUNodeEditor.requestRender: renderData has', renderData.rects.length, 'rects');
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
    currentType?: string
  ): void {
    this.uiManager.showTypeSelector(nodeId, handleId, screenX, screenY, options, currentType);
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
    // 转换为画布坐标
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    
    // 遍历节点进行命中测试
    for (const node of this.nodes) {
      const graphNode: GraphNode = {
        id: node.id,
        type: node.type as GraphNode['type'],
        position: node.position,
        data: node.data as GraphNode['data'],
      };
      const layout = computeNodeLayout(graphNode, false);
      const hit = hitTestTypeLabel(canvasPos.x, canvasPos.y, layout);
      
      if (hit) {
        this.onTypeLabelClick?.(hit.nodeId, hit.handleId, hit.canvasX, hit.canvasY);
        return true;
      }
    }
    
    return false;
  }

  /**
   * 处理 Variadic 按钮点击（供外部调用）
   */
  handleVariadicButtonClick(screenX: number, screenY: number): boolean {
    // 转换为画布坐标
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    
    // 遍历节点进行命中测试
    for (const node of this.nodes) {
      // 只有 operation 节点有 Variadic 端口
      if (node.type !== 'operation') continue;
      
      const data = node.data as BlueprintNodeData;
      const variadicGroups = this.getVariadicGroups(data);
      if (variadicGroups.length === 0) continue;
      
      const graphNode: GraphNode = {
        id: node.id,
        type: node.type as GraphNode['type'],
        position: node.position,
        data: node.data as GraphNode['data'],
      };
      const layout = computeNodeLayout(graphNode, false);
      const hit = hitTestVariadicButton(canvasPos.x, canvasPos.y, layout, variadicGroups);
      
      if (hit) {
        if (hit.action === 'add') {
          this.onVariadicAdd?.(hit.nodeId, hit.groupName);
        } else {
          this.onVariadicRemove?.(hit.nodeId, hit.groupName);
        }
        return true;
      }
    }
    
    return false;
  }

  /**
   * 获取节点的 Variadic 组名列表
   */
  private getVariadicGroups(data: BlueprintNodeData): string[] {
    const groups: string[] = [];
    const op = data.operation;
    
    // 检查 operands
    for (const arg of op.arguments) {
      if (arg.kind === 'operand' && arg.isVariadic && !groups.includes(arg.name)) {
        groups.push(arg.name);
      }
    }
    
    // 检查 results
    for (const result of op.results) {
      if (result.isVariadic && !groups.includes(result.name)) {
        groups.push(result.name);
      }
    }
    
    return groups;
  }

  // ============================================================
  // 扩展命中测试处理
  // ============================================================

  /**
   * 处理扩展命中测试（支持所有交互区域）
   * 返回是否命中了交互区域
   */
  handleExtendedHitTest(screenX: number, screenY: number): boolean {
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    
    for (const node of this.nodes) {
      const graphNode: GraphNode = {
        id: node.id,
        type: node.type as GraphNode['type'],
        position: node.position,
        data: node.data as GraphNode['data'],
      };
      const layout = computeNodeLayout(graphNode, false);
      
      // 构建命中测试选项
      const options = this.buildHitTestOptions(node);
      const hit = hitTestNodeExtended(canvasPos.x, canvasPos.y, layout, options);
      
      if (this.handleHitResult(hit, node)) {
        return true;
      }
    }
    
    return false;
  }

  /**
   * 构建命中测试选项
   */
  private buildHitTestOptions(node: EditorNode): {
    nodeType: NodeType;
    variadicGroups?: string[];
    entryData?: EntryNodeHitData;
    returnData?: ReturnNodeHitData;
    hoveredParamIndex?: number | null;
    hoveredReturnIndex?: number | null;
  } {
    const nodeType = node.type as NodeType;
    
    switch (nodeType) {
      case 'operation': {
        const data = node.data as BlueprintNodeData;
        return {
          nodeType,
          variadicGroups: this.getVariadicGroups(data),
        };
      }
      case 'function-entry': {
        const data = node.data as FunctionEntryData;
        return {
          nodeType,
          entryData: {
            isMain: data.isMain,
            parameters: data.outputs?.map(o => ({ name: o.name })) ?? [],
            traitsExpanded: this.traitsExpandedMap.get(node.id) ?? false,
          },
          hoveredParamIndex: this.hoveredNodeId === node.id ? this.hoveredParamIndex : null,
        };
      }
      case 'function-return': {
        const data = node.data as FunctionReturnData;
        return {
          nodeType,
          returnData: {
            isMain: data.isMain,
            returnTypes: data.inputs?.map(i => ({ name: i.name })) ?? [],
          },
          hoveredReturnIndex: this.hoveredNodeId === node.id ? this.hoveredReturnIndex : null,
        };
      }
      case 'function-call':
        return { nodeType };
      default:
        return { nodeType: 'operation' };
    }
  }

  /**
   * 处理命中结果
   */
  private handleHitResult(hit: HitResult, node: EditorNode): boolean {
    switch (hit.kind) {
      case 'type-label':
        this.onTypeLabelClick?.(hit.nodeId, hit.handleId, hit.canvasX, hit.canvasY);
        return true;
        
      case 'variadic-button':
        if (hit.action === 'add') {
          this.onVariadicAdd?.(hit.nodeId, hit.groupName);
        } else {
          this.onVariadicRemove?.(hit.nodeId, hit.groupName);
        }
        return true;
        
      case 'param-add':
        this.onParameterAdd?.(hit.nodeId);
        return true;
        
      case 'param-remove': {
        const entryData = node.data as FunctionEntryData;
        const paramName = entryData.outputs?.[hit.paramIndex]?.name;
        if (paramName) {
          this.onParameterRemove?.(hit.nodeId, paramName);
        }
        return true;
      }
        
      case 'param-name':
        this.onParamNameClick?.(hit.nodeId, hit.paramIndex, hit.currentName, hit.canvasX, hit.canvasY);
        return true;
        
      case 'return-add':
        this.onReturnTypeAdd?.(hit.nodeId);
        return true;
        
      case 'return-remove': {
        const returnData = node.data as FunctionReturnData;
        const returnName = returnData.inputs?.[hit.returnIndex]?.name;
        if (returnName) {
          this.onReturnTypeRemove?.(hit.nodeId, returnName);
        }
        return true;
      }
        
      case 'return-name':
        this.onReturnNameClick?.(hit.nodeId, hit.returnIndex, hit.currentName, hit.canvasX, hit.canvasY);
        return true;
        
      case 'traits-toggle':
        this.onTraitsToggleClick?.(hit.nodeId, hit.canvasX, hit.canvasY);
        return true;
        
      default:
        return false;
    }
  }

  /**
   * 更新 hover 状态（用于显示删除按钮）
   */
  updateHoverState(screenX: number, screenY: number): void {
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    let newHoveredNodeId: string | null = null;
    let newHoveredParamIndex: number | null = null;
    let newHoveredReturnIndex: number | null = null;
    
    for (const node of this.nodes) {
      const graphNode: GraphNode = {
        id: node.id,
        type: node.type as GraphNode['type'],
        position: node.position,
        data: node.data as GraphNode['data'],
      };
      const layout = computeNodeLayout(graphNode, false);
      
      if (node.type === 'function-entry') {
        const data = node.data as FunctionEntryData;
        if (!data.isMain) {
          const paramCount = data.outputs?.length ?? 0;
          const index = computeHoveredIndex(canvasPos.x, canvasPos.y, layout, 'function-entry', paramCount);
          if (index !== null) {
            newHoveredNodeId = node.id;
            newHoveredParamIndex = index;
            break;
          }
        }
      } else if (node.type === 'function-return') {
        const data = node.data as FunctionReturnData;
        if (!data.isMain) {
          const returnCount = data.inputs?.length ?? 0;
          const index = computeHoveredIndex(canvasPos.x, canvasPos.y, layout, 'function-return', returnCount);
          if (index !== null) {
            newHoveredNodeId = node.id;
            newHoveredReturnIndex = index;
            break;
          }
        }
      }
    }
    
    // 检查是否有变化
    if (newHoveredNodeId !== this.hoveredNodeId ||
        newHoveredParamIndex !== this.hoveredParamIndex ||
        newHoveredReturnIndex !== this.hoveredReturnIndex) {
      this.hoveredNodeId = newHoveredNodeId;
      this.hoveredParamIndex = newHoveredParamIndex;
      this.hoveredReturnIndex = newHoveredReturnIndex;
      // 触发重渲染以显示/隐藏删除按钮
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
