/**
 * ReactFlowNodeEditor
 * 
 * INodeEditor 的 React Flow 实现。
 * 使用 createRoot 将 React Flow 组件挂载到 DOM 容器。
 * 
 * 设计：
 * - 实现 INodeEditor 接口
 * - 内部使用 ReactFlowCanvas 组件渲染
 * - 通过 props 传递状态和回调
 */

import { createRoot, type Root } from 'react-dom/client';
import type { INodeEditor } from '../../INodeEditor';
import { createEmptyCallbacks } from '../../INodeEditor';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  NodeChange,
  EdgeChange,
  ConnectionRequest,
} from '../../types';
import { ReactFlowCanvas, type ReactFlowCanvasHandle } from './ReactFlowCanvas';

import type { FunctionTrait } from '../../../types';

/**
 * React Flow 节点编辑器实现
 */
export class ReactFlowNodeEditor implements INodeEditor {
  private root: Root | null = null;
  private canvasHandle: ReactFlowCanvasHandle | null = null;
  
  // 内部状态
  private nodes: EditorNode[] = [];
  private edges: EditorEdge[] = [];
  private viewport: EditorViewport = { x: 0, y: 0, zoom: 1 };
  private selection: EditorSelection = { nodeIds: [], edgeIds: [] };
  
  // 事件回调
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
  onTypeLabelClick: ((nodeId: string, handleId: string, canvasX: number, canvasY: number) => void) | null = null;
  // onNodeDataChange 已废弃：节点组件现在直接更新 editorStore
  onNodeDataChange: ((nodeId: string, data: Record<string, unknown>) => void) | null = null;
  
  constructor() {
    // 初始化空回调
    Object.assign(this, createEmptyCallbacks());
  }
  
  // ============================================================
  // 生命周期
  // ============================================================
  
  mount(container: HTMLElement): void {
    if (this.root) {
      console.warn('ReactFlowNodeEditor: already mounted, unmounting first');
      this.unmount();
    }
    
    this.root = createRoot(container);
    this.render();
  }
  
  unmount(): void {
    if (this.root) {
      const root = this.root;
      this.root = null;
      this.canvasHandle = null;
      
      // 使用 queueMicrotask 避免 React 渲染冲突
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
    this.canvasHandle?.setNodes(nodes);
  }
  
  setEdges(edges: EditorEdge[]): void {
    this.edges = edges;
    this.canvasHandle?.setEdges(edges);
  }
  
  setSelection(selection: EditorSelection): void {
    this.selection = selection;
    this.canvasHandle?.setSelection(selection);
  }
  
  setViewport(viewport: EditorViewport): void {
    this.viewport = viewport;
    this.canvasHandle?.setViewport(viewport);
  }
  
  // ============================================================
  // 命令
  // ============================================================
  
  fitView(options?: { padding?: number; maxZoom?: number }): void {
    this.canvasHandle?.fitView(options);
  }
  
  getViewport(): EditorViewport {
    return this.canvasHandle?.getViewport() ?? this.viewport;
  }
  
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number } {
    return this.canvasHandle?.screenToCanvas(screenX, screenY) ?? { x: 0, y: 0 };
  }
  
  // ============================================================
  // 元信息
  // ============================================================
  
  getName(): string {
    return 'React Flow';
  }
  
  isAvailable(): boolean {
    return true;
  }
  
  // ============================================================
  // 内部方法
  // ============================================================
  
  private handleCanvasReady = (handle: ReactFlowCanvasHandle): void => {
    this.canvasHandle = handle;
    // 同步初始状态
    handle.setNodes(this.nodes);
    handle.setEdges(this.edges);
    handle.setSelection(this.selection);
    if (this.viewport.x !== 0 || this.viewport.y !== 0 || this.viewport.zoom !== 1) {
      handle.setViewport(this.viewport);
    }
  };
  
  private render(): void {
    if (!this.root) return;
    
    this.root.render(
      <ReactFlowCanvas
        initialNodes={this.nodes}
        initialEdges={this.edges}
        initialViewport={this.viewport}
        onReady={this.handleCanvasReady}
        onNodesChange={this.onNodesChange ?? undefined}
        onEdgesChange={this.onEdgesChange ?? undefined}
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
 * 创建 React Flow 节点编辑器实例
 */
export function createReactFlowNodeEditor(): ReactFlowNodeEditor {
  return new ReactFlowNodeEditor();
}
