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
import type { FunctionTrait } from '../types';

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
  
  /** 边双击回调 */
  onEdgeDoubleClick: ((edgeId: string) => void) | null;
  
  /** 拖放回调（从外部拖入元素） */
  onDrop: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null;
  
  /** 删除请求回调（用户按 Delete 键） */
  onDeleteRequest: ((nodeIds: string[], edgeIds: string[]) => void) | null;
  
  // ============================================================
  // 业务事件回调（节点交互）
  // ============================================================
  
  /** 属性变更回调 */
  onAttributeChange: ((nodeId: string, attributeName: string, value: string) => void) | null;
  
  /** Variadic 端口增加回调 */
  onVariadicAdd: ((nodeId: string, groupName: string) => void) | null;
  
  /** Variadic 端口减少回调 */
  onVariadicRemove: ((nodeId: string, groupName: string) => void) | null;
  
  /** 参数添加回调 */
  onParameterAdd: ((functionId: string) => void) | null;
  
  /** 参数移除回调 */
  onParameterRemove: ((functionId: string, parameterName: string) => void) | null;
  
  /** 参数重命名回调 */
  onParameterRename: ((functionId: string, oldName: string, newName: string) => void) | null;
  
  /** 返回值添加回调 */
  onReturnTypeAdd: ((functionId: string) => void) | null;
  
  /** 返回值移除回调 */
  onReturnTypeRemove: ((functionId: string, returnName: string) => void) | null;
  
  /** 返回值重命名回调 */
  onReturnTypeRename: ((functionId: string, oldName: string, newName: string) => void) | null;
  
  /** Traits 变更回调 */
  onTraitsChange: ((functionId: string, traits: FunctionTrait[]) => void) | null;
  
  /** 类型标签点击回调（用于显示类型选择器） */
  onTypeLabelClick: ((nodeId: string, handleId: string, canvasX: number, canvasY: number) => void) | null;
  
  /** 
   * 节点数据变更回调（通用）
   * 
   * 当节点的 data 发生变化时触发（类型选择、属性编辑、variadic 等）。
   * 这是一个通用回调，用于同步 ReactFlow/VueFlow 内部状态到 editorStore。
   * 
   * @param nodeId - 节点 ID
   * @param data - 变更后的完整节点数据
   */
  onNodeDataChange: ((nodeId: string, data: Record<string, unknown>) => void) | null;
  
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
  | 'onEdgeDoubleClick'
  | 'onDrop'
  | 'onDeleteRequest'
  | 'onAttributeChange'
  | 'onVariadicAdd'
  | 'onVariadicRemove'
  | 'onParameterAdd'
  | 'onParameterRemove'
  | 'onParameterRename'
  | 'onReturnTypeAdd'
  | 'onReturnTypeRemove'
  | 'onReturnTypeRename'
  | 'onTraitsChange'
  | 'onTypeLabelClick'
  | 'onNodeDataChange'
> {
  return {
    onNodesChange: null,
    onEdgesChange: null,
    onSelectionChange: null,
    onViewportChange: null,
    onConnect: null,
    onEdgeDoubleClick: null,
    onDrop: null,
    onDeleteRequest: null,
    onAttributeChange: null,
    onVariadicAdd: null,
    onVariadicRemove: null,
    onParameterAdd: null,
    onParameterRemove: null,
    onParameterRename: null,
    onReturnTypeAdd: null,
    onReturnTypeRemove: null,
    onReturnTypeRename: null,
    onTraitsChange: null,
    onTypeLabelClick: null,
    onNodeDataChange: null,
  };
}
