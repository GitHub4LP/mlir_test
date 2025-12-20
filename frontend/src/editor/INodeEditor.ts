/**
 * 节点编辑器接口
 * 
 * 设计原则：
 * - 在 Node Editor 语义级别定义接口
 * - 各实现（ReactFlow、Canvas、WebGL）独立适配此接口
 * - Application 层持有状态，通过此接口与编辑器交互
 * 
 * 数据流：
 * - Application → Editor: setNodes(), setEdges(), setSelection(), setViewport()
 * - Editor → Application: onNodesChange, onEdgesChange, onSelectionChange, ...
 */

import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
  NodeChange,
  EdgeChange,
} from './types';

/** 节点编辑器接口 */
export interface INodeEditor {
  // ============================================================
  // 生命周期
  // ============================================================
  
  /** 挂载到 DOM 容器 */
  mount(container: HTMLElement): void;
  
  /** 卸载 */
  unmount(): void;
  
  // ============================================================
  // 数据设置（Application → Editor）
  // ============================================================
  
  /** 设置节点列表 */
  setNodes(nodes: EditorNode[]): void;
  
  /** 设置边列表 */
  setEdges(edges: EditorEdge[]): void;
  
  /** 设置选择状态 */
  setSelection(selection: EditorSelection): void;
  
  /** 设置视口状态 */
  setViewport(viewport: EditorViewport): void;
  
  // ============================================================
  // 命令
  // ============================================================
  
  /** 适应视口（显示所有节点） */
  fitView(options?: { padding?: number; maxZoom?: number }): void;
  
  /** 获取当前视口 */
  getViewport(): EditorViewport;
  
  /** 屏幕坐标转画布坐标 */
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number };
  
  // ============================================================
  // 事件回调（Editor → Application）
  // ============================================================
  
  /** 节点变更回调（位置、选择、删除） */
  onNodesChange: ((changes: NodeChange[]) => void) | null;
  
  /** 边变更回调（选择、删除） */
  onEdgesChange: ((changes: EdgeChange[]) => void) | null;
  
  /** 选择变更回调（节点和边的选择状态变化） */
  onSelectionChange: ((selection: EditorSelection) => void) | null;
  
  /** 视口变更回调 */
  onViewportChange: ((viewport: EditorViewport) => void) | null;
  
  /** 连接请求回调（用户尝试创建连接） */
  onConnect: ((request: ConnectionRequest) => void) | null;
  
  /** 节点双击回调 */
  onNodeDoubleClick: ((nodeId: string) => void) | null;
  
  /** 边双击回调 */
  onEdgeDoubleClick: ((edgeId: string) => void) | null;
  
  /** 拖放回调（从外部拖入元素） */
  onDrop: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null;
  
  /** 删除请求回调（用户按 Delete 键） */
  onDeleteRequest: ((nodeIds: string[], edgeIds: string[]) => void) | null;
  
  // ============================================================
  // 元信息
  // ============================================================
  
  /** 获取编辑器名称 */
  getName(): string;
  
  /** 检查是否可用 */
  isAvailable(): boolean;
}

/** 创建空的节点编辑器回调集合 */
export function createEmptyCallbacks(): Pick<
  INodeEditor,
  | 'onNodesChange'
  | 'onEdgesChange'
  | 'onSelectionChange'
  | 'onViewportChange'
  | 'onConnect'
  | 'onNodeDoubleClick'
  | 'onEdgeDoubleClick'
  | 'onDrop'
  | 'onDeleteRequest'
> {
  return {
    onNodesChange: null,
    onEdgesChange: null,
    onSelectionChange: null,
    onViewportChange: null,
    onConnect: null,
    onNodeDoubleClick: null,
    onEdgeDoubleClick: null,
    onDrop: null,
    onDeleteRequest: null,
  };
}
