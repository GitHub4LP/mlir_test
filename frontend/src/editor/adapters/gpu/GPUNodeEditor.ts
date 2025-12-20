/**
 * GPU 节点编辑器
 * 
 * 实现 INodeEditor 接口，使用 GPURenderer 进行渲染。
 * 集成 GraphController 处理业务逻辑。
 * 
 * 覆盖层支持：
 * - 使用 OverlayManager 管理 HTML 覆盖层
 * - 支持类型选择器、属性编辑器等 UI 组件
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
import { GPURenderer } from './GPURenderer';
import { GraphController } from '../canvas/GraphController';
import { OverlayManager, type OverlayConfig, type ActiveOverlay } from '../canvas/OverlayManager';
import { hitTestTypeLabel, getTypeLabelPosition } from '../canvas/HitTest';
import { computeNodeLayout } from '../canvas/layout';
import type { GraphState, GraphNode, GraphEdge } from '../../../types';

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
  private overlayManager: OverlayManager;
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
  
  // 覆盖层回调
  onOverlayChange: ((overlays: ActiveOverlay[]) => void) | null = null;
  onTypeLabelClick: ((nodeId: string, handleId: string, canvasX: number, canvasY: number) => void) | null = null;

  
  constructor(preferWebGPU: boolean = true) {
    this.renderer = new GPURenderer(preferWebGPU);
    this.controller = new GraphController();
    this.overlayManager = new OverlayManager();
    
    // 设置控制器回调
    this.setupControllerCallbacks();
    
    // 设置图数据提供者
    this.controller.setGraphDataProvider(() => this.getGraphState());
    
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
      // 更新覆盖层位置
      this.overlayManager.updateViewport(viewport);
      this.onOverlayChange?.(this.overlayManager.getActiveOverlays());
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
    this.overlayManager.mount(container);
    // 等待渲染器就绪后再渲染
    this.renderer.waitForReady().then(() => {
      this.requestRender();
    });
  }
  
  unmount(): void {
    this.overlayManager.unmount();
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
  
  private requestRender(): void {
    if (!this.renderer.isReady()) return;
    const renderData = this.controller.computeRenderData();
    this.renderer.render(renderData);
  }

  // ============================================================
  // 覆盖层管理
  // ============================================================

  /**
   * 显示类型选择器覆盖层
   */
  showTypeSelector(nodeId: string, handleId: string): string | null {
    // 查找节点
    const node = this.nodes.find(n => n.id === nodeId);
    if (!node) return null;
    
    // 计算节点布局
    const graphNode: GraphNode = {
      id: node.id,
      type: node.type as GraphNode['type'],
      position: node.position,
      data: node.data as GraphNode['data'],
    };
    const layout = computeNodeLayout(graphNode, false);
    
    // 获取类型标签位置
    const pos = getTypeLabelPosition(layout, handleId);
    if (!pos) return null;
    
    // 显示覆盖层
    const config: OverlayConfig = {
      type: 'type-selector',
      nodeId,
      portId: handleId,
      canvasX: pos.canvasX,
      canvasY: pos.canvasY,
      data: { node: node.data },
    };
    
    const id = this.overlayManager.show(config);
    this.onOverlayChange?.(this.overlayManager.getActiveOverlays());
    return id;
  }

  /**
   * 隐藏覆盖层
   */
  hideOverlay(id: string): void {
    this.overlayManager.hide(id);
    this.onOverlayChange?.(this.overlayManager.getActiveOverlays());
  }

  /**
   * 隐藏所有覆盖层
   */
  hideAllOverlays(): void {
    this.overlayManager.hideAll();
    this.onOverlayChange?.([]);
  }

  /**
   * 获取覆盖层容器
   */
  getOverlayContainer(): HTMLDivElement | null {
    return this.overlayManager.getContainer();
  }

  /**
   * 获取活动覆盖层列表
   */
  getActiveOverlays(): ActiveOverlay[] {
    return this.overlayManager.getActiveOverlays();
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
}
