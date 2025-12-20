/**
 * Canvas 节点编辑器
 * 
 * 实现 INodeEditor 接口，封装 GraphController 和 CanvasRenderer。
 * 将旧架构的图元级别 API 适配为 Node Editor 语义级别。
 */

import type { INodeEditor } from '../INodeEditor';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
  NodeChange,
  EdgeChange,
} from '../types';
import { CanvasRenderer } from './canvas/CanvasRenderer';
import { GraphController } from './canvas/GraphController';
import type { GraphState, GraphNode } from '../../types';

/**
 * 将 EditorNode/EditorEdge 转换为 GraphState
 */
function toGraphState(nodes: EditorNode[], edges: EditorEdge[]): GraphState {
  return {
    nodes: nodes.map(n => ({
      id: n.id,
      type: n.type as GraphNode['type'],
      position: { x: n.position.x, y: n.position.y },
      data: n.data,
    })) as GraphNode[],
    edges: edges.map(e => ({
      source: e.source,
      sourceHandle: e.sourceHandle,
      target: e.target,
      targetHandle: e.targetHandle,
    })),
  };
}

/**
 * Canvas 节点编辑器
 * 
 * 实现 INodeEditor 接口，使用 Canvas 2D 渲染。
 */
export class CanvasNodeEditor implements INodeEditor {
  private renderer: CanvasRenderer | null = null;
  private controller: GraphController | null = null;
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

  // ============================================================
  // 生命周期
  // ============================================================

  mount(container: HTMLElement): void {
    this.container = container;
    this.renderer = new CanvasRenderer();
    this.controller = new GraphController();
    
    this.renderer.mount(container);
    this.controller.setRenderer(this.renderer);
    
    // 设置图数据提供者
    this.controller.setGraphDataProvider(() => toGraphState(this.nodes, this.edges));
    
    // 设置拖放回调
    this.renderer.setDropCallback((x, y, dataTransfer) => {
      this.onDrop?.(x, y, dataTransfer);
    });
    
    // 适配 GraphController 回调到 INodeEditor 接口
    this.controller.onNodePositionChange = (nodeId, x, y) => {
      const change: NodeChange = {
        type: 'position',
        id: nodeId,
        position: { x, y },
        dragging: true,
      };
      this.onNodesChange?.([change]);
    };
    
    this.controller.onSelectionChange = (nodeIds, edgeIds) => {
      const newSelection: EditorSelection = { nodeIds, edgeIds };
      this.selection = newSelection;
      this.onSelectionChange?.(newSelection);
    };
    
    this.controller.onConnectionAttempt = (sourceNodeId, sourceHandleId, targetNodeId, targetHandleId) => {
      const request: ConnectionRequest = {
        source: sourceNodeId,
        sourceHandle: sourceHandleId,
        target: targetNodeId,
        targetHandle: targetHandleId,
      };
      this.onConnect?.(request);
    };
    
    this.controller.onDeleteSelected = () => {
      this.onDeleteRequest?.(this.selection.nodeIds, this.selection.edgeIds);
    };
    
    this.controller.onEdgeDoubleClick = (edgeId) => {
      this.onEdgeDoubleClick?.(edgeId);
    };
    
    // 初始渲染
    this.controller.requestRender();
    
    // 监听窗口大小变化
    window.addEventListener('resize', this.handleResize);
  }

  unmount(): void {
    window.removeEventListener('resize', this.handleResize);
    
    if (this.renderer) {
      this.renderer.unmount();
      this.renderer = null;
    }
    this.controller = null;
    this.container = null;
  }

  private handleResize = (): void => {
    this.renderer?.resize();
    this.controller?.requestRender();
  };

  // ============================================================
  // 数据设置
  // ============================================================

  setNodes(nodes: EditorNode[]): void {
    this.nodes = nodes;
    
    // 同步选择状态
    const selectedIds = nodes.filter(n => n.selected).map(n => n.id);
    this.controller?.syncSelectionFromExternal(selectedIds);
    
    // 只有在非拖拽状态时才清除缓存
    const state = this.controller?.getState();
    if (state?.kind === 'idle') {
      this.controller?.clearLayoutCache();
    }
    this.controller?.requestRender();
  }

  setEdges(edges: EditorEdge[]): void {
    this.edges = edges;
    this.controller?.requestRender();
  }

  setSelection(selection: EditorSelection): void {
    this.selection = selection;
    this.controller?.syncSelectionFromExternal(selection.nodeIds);
  }

  setViewport(viewport: EditorViewport): void {
    this.controller?.setViewport(viewport);
  }

  // ============================================================
  // 命令
  // ============================================================

  fitView(options?: { padding?: number; maxZoom?: number }): void {
    if (!this.controller || !this.container) return;
    if (this.nodes.length === 0) return;
    
    const padding = options?.padding ?? 50;
    const maxZoom = options?.maxZoom ?? 1;
    
    // 计算边界
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const node of this.nodes) {
      minX = Math.min(minX, node.position.x);
      minY = Math.min(minY, node.position.y);
      maxX = Math.max(maxX, node.position.x + 200); // 估算节点宽度
      maxY = Math.max(maxY, node.position.y + 100); // 估算节点高度
    }
    
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    const graphWidth = maxX - minX + padding * 2;
    const graphHeight = maxY - minY + padding * 2;
    
    const zoom = Math.min(width / graphWidth, height / graphHeight, maxZoom);
    const x = (width - graphWidth * zoom) / 2 - minX * zoom + padding * zoom;
    const y = (height - graphHeight * zoom) / 2 - minY * zoom + padding * zoom;
    
    this.controller.setViewport({ x, y, zoom });
  }

  getViewport(): EditorViewport {
    const vp = this.controller?.getViewport();
    return vp ?? { x: 0, y: 0, zoom: 1 };
  }

  screenToCanvas(screenX: number, screenY: number): { x: number; y: number } {
    return this.controller?.screenToCanvas(screenX, screenY) ?? { x: 0, y: 0 };
  }

  // ============================================================
  // 元信息
  // ============================================================

  getName(): string {
    return 'Canvas';
  }

  isAvailable(): boolean {
    return typeof document !== 'undefined' && 
           typeof document.createElement('canvas').getContext === 'function';
  }
}

/**
 * 创建 Canvas 节点编辑器实例
 */
export function createCanvasNodeEditor(): INodeEditor {
  return new CanvasNodeEditor();
}
