/**
 * GPU 节点编辑器
 * 
 * 实现 INodeEditor 接口，封装 GPUNodeEditor (WebGL/WebGPU)。
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
import type { FunctionTrait } from '../../types';
import { GPUNodeEditor as GPUNodeEditorCore } from './gpu/GPUNodeEditor';

/**
 * GPU 节点编辑器
 * 
 * 实现 INodeEditor 接口，使用 WebGL/WebGPU 渲染。
 */
export class GPUNodeEditor implements INodeEditor {
  private editor: GPUNodeEditorCore | null = null;
  private preferWebGPU: boolean;
  
  // 当前数据（缓存，用于 mount 后同步）
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

  constructor(preferWebGPU: boolean = true) {
    this.preferWebGPU = preferWebGPU;
  }

  // ============================================================
  // 生命周期
  // ============================================================

  mount(container: HTMLElement): void {
    this.editor = new GPUNodeEditorCore(this.preferWebGPU);
    
    // 绑定回调
    this.editor.onNodesChange = this.onNodesChange;
    this.editor.onEdgesChange = this.onEdgesChange;
    this.editor.onSelectionChange = this.onSelectionChange;
    this.editor.onViewportChange = this.onViewportChange;
    this.editor.onConnect = this.onConnect;
    this.editor.onNodeDoubleClick = this.onNodeDoubleClick;
    this.editor.onEdgeDoubleClick = this.onEdgeDoubleClick;
    this.editor.onDrop = this.onDrop;
    this.editor.onDeleteRequest = this.onDeleteRequest;
    
    // 绑定业务回调
    this.editor.onAttributeChange = this.onAttributeChange;
    this.editor.onVariadicAdd = this.onVariadicAdd;
    this.editor.onVariadicRemove = this.onVariadicRemove;
    this.editor.onParameterAdd = this.onParameterAdd;
    this.editor.onParameterRemove = this.onParameterRemove;
    this.editor.onParameterRename = this.onParameterRename;
    this.editor.onReturnTypeAdd = this.onReturnTypeAdd;
    this.editor.onReturnTypeRemove = this.onReturnTypeRemove;
    this.editor.onReturnTypeRename = this.onReturnTypeRename;
    this.editor.onTraitsChange = this.onTraitsChange;
    
    // 挂载
    this.editor.mount(container);
    
    // 同步缓存的数据
    if (this.nodes.length > 0) {
      this.editor.setNodes(this.nodes);
    }
    if (this.edges.length > 0) {
      this.editor.setEdges(this.edges);
    }
    if (this.viewport.x !== 0 || this.viewport.y !== 0 || this.viewport.zoom !== 1) {
      this.editor.setViewport(this.viewport);
    }
    if (this.selection.nodeIds.length > 0 || this.selection.edgeIds.length > 0) {
      this.editor.setSelection(this.selection);
    }
  }

  unmount(): void {
    if (this.editor) {
      this.editor.unmount();
      this.editor = null;
    }
  }

  // ============================================================
  // 数据设置
  // ============================================================

  setNodes(nodes: EditorNode[]): void {
    this.nodes = nodes;
    this.editor?.setNodes(nodes);
  }

  setEdges(edges: EditorEdge[]): void {
    this.edges = edges;
    this.editor?.setEdges(edges);
  }

  setSelection(selection: EditorSelection): void {
    this.selection = selection;
    this.editor?.setSelection(selection);
  }

  setViewport(viewport: EditorViewport): void {
    this.viewport = viewport;
    this.editor?.setViewport(viewport);
  }

  // ============================================================
  // 命令
  // ============================================================

  fitView(options?: { padding?: number; maxZoom?: number }): void {
    this.editor?.fitView(options);
  }

  getViewport(): EditorViewport {
    return this.editor?.getViewport() ?? this.viewport;
  }

  screenToCanvas(screenX: number, screenY: number): { x: number; y: number } {
    return this.editor?.screenToCanvas(screenX, screenY) ?? { x: 0, y: 0 };
  }

  // ============================================================
  // 元信息
  // ============================================================

  getName(): string {
    return this.editor?.getName() ?? (this.preferWebGPU ? 'WebGPU' : 'WebGL');
  }

  isAvailable(): boolean {
    return this.editor?.isAvailable() ?? true;
  }
}

/**
 * 创建 GPU 节点编辑器实例
 */
export function createGPUNodeEditor(preferWebGPU: boolean = true): INodeEditor {
  return new GPUNodeEditor(preferWebGPU);
}
