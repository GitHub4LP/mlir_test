/**
 * Vue Flow 节点编辑器
 * 
 * 实现 INodeEditor 接口，使用 createRoot 挂载 VueFlowBridge。
 * 采用与 ReactFlowNodeEditor 相同的 onReady 回调模式。
 */

import { createRoot, type Root } from 'react-dom/client';
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
import { VueFlowBridge, type VueFlowBridgeHandle } from './VueFlowBridge';

/**
 * Vue Flow 节点编辑器
 * 
 * 实现 INodeEditor 接口，使用 Vue Flow 渲染。
 */
export class VueFlowNodeEditor implements INodeEditor {
  private root: Root | null = null;
  private bridgeHandle: VueFlowBridgeHandle | null = null;
  
  // 当前数据
  private nodes: EditorNode[] = [];
  private edges: EditorEdge[] = [];
  private viewport: EditorViewport = { x: 0, y: 0, zoom: 1 };
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
    if (this.root) {
      console.warn('VueFlowNodeEditor: already mounted, unmounting first');
      this.unmount();
    }
    
    this.root = createRoot(container);
    this.render();
  }

  unmount(): void {
    if (this.root) {
      const root = this.root;
      this.root = null;
      this.bridgeHandle = null;
      
      queueMicrotask(() => {
        root.unmount();
      });
    }
  }

  // ============================================================
  // 数据设置
  // ============================================================

  setNodes(nodes: EditorNode[]): void {
    this.nodes = nodes;
    this.render();
  }

  setEdges(edges: EditorEdge[]): void {
    this.edges = edges;
    this.render();
  }

  setSelection(selection: EditorSelection): void {
    this.selection = selection;
    this.render();
  }

  setViewport(viewport: EditorViewport): void {
    this.viewport = viewport;
    // 通过 handle 调用 Vue 组件的 setViewport
    this.bridgeHandle?.setViewport(viewport);
  }

  // ============================================================
  // 命令
  // ============================================================

  fitView(): void {
    this.bridgeHandle?.fitView();
  }

  getViewport(): EditorViewport {
    return this.viewport;
  }

  screenToCanvas(): { x: number; y: number } {
    // Vue Flow 的坐标转换需要通过 bridge 实现
    return { x: 0, y: 0 };
  }

  // ============================================================
  // 元信息
  // ============================================================

  getName(): string {
    return 'Vue Flow';
  }

  isAvailable(): boolean {
    return true;
  }

  // ============================================================
  // 内部方法
  // ============================================================

  private handleBridgeReady = (handle: VueFlowBridgeHandle): void => {
    this.bridgeHandle = handle;
    // 同步初始视口状态（与 ReactFlowNodeEditor 一致）
    if (this.viewport.x !== 0 || this.viewport.y !== 0 || this.viewport.zoom !== 1) {
      handle.setViewport(this.viewport);
    }
  };

  private render(): void {
    if (!this.root) return;
    
    this.root.render(
      <VueFlowBridge
        nodes={this.nodes}
        edges={this.edges}
        viewport={this.viewport}
        selection={this.selection}
        onReady={this.handleBridgeReady}
        onNodesChange={this.onNodesChange ?? undefined}
        onSelectionChange={this.onSelectionChange ?? undefined}
        onViewportChange={this.onViewportChange ?? undefined}
        onConnect={this.onConnect ?? undefined}
        onNodeDoubleClick={this.onNodeDoubleClick ?? undefined}
        onEdgeDoubleClick={this.onEdgeDoubleClick ?? undefined}
        onDrop={this.onDrop ?? undefined}
        onDeleteRequest={this.onDeleteRequest ?? undefined}
      />
    );
  }
}

/**
 * 创建 Vue Flow 节点编辑器实例
 */
export function createVueFlowNodeEditor(): INodeEditor {
  return new VueFlowNodeEditor();
}

