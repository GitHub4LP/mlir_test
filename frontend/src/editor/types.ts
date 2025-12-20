/**
 * 节点编辑器类型定义
 * 
 * 设计原则：
 * - 在 Node Editor 语义级别定义接口，而非底层图元
 * - 各实现（ReactFlow、Canvas、WebGL）独立适配此接口
 * - Application 层通过此接口与编辑器交互
 */

// ============================================================
// 基础数据类型
// ============================================================

/** 编辑器节点 */
export interface EditorNode {
  id: string;
  type: 'operation' | 'function-entry' | 'function-return' | 'function-call';
  position: { x: number; y: number };
  data: unknown;
  selected?: boolean;
}

/** 编辑器边 */
export interface EditorEdge {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
  selected?: boolean;
  /** 边类型：执行流或数据流 */
  type?: 'execution' | 'data';
  /** 边数据（如颜色） */
  data?: {
    color?: string;
  };
}

/** 视口状态 */
export interface EditorViewport {
  x: number;
  y: number;
  zoom: number;
}

/** 选择状态 */
export interface EditorSelection {
  nodeIds: string[];
  edgeIds: string[];
}

/** 连接请求（用户尝试创建连接） */
export interface ConnectionRequest {
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

// ============================================================
// 变更类型（参考 React Flow 的 NodeChange/EdgeChange）
// ============================================================

/** 节点位置变更 */
export interface NodePositionChange {
  type: 'position';
  id: string;
  position: { x: number; y: number };
  /** 是否正在拖拽中（用于区分拖拽过程和拖拽结束） */
  dragging?: boolean;
}

/** 节点选择变更 */
export interface NodeSelectChange {
  type: 'select';
  id: string;
  selected: boolean;
}

/** 节点删除变更 */
export interface NodeRemoveChange {
  type: 'remove';
  id: string;
}

/** 节点变更联合类型 */
export type NodeChange = NodePositionChange | NodeSelectChange | NodeRemoveChange;

/** 边选择变更 */
export interface EdgeSelectChange {
  type: 'select';
  id: string;
  selected: boolean;
}

/** 边删除变更 */
export interface EdgeRemoveChange {
  type: 'remove';
  id: string;
}

/** 边变更联合类型 */
export type EdgeChange = EdgeSelectChange | EdgeRemoveChange;

// ============================================================
// 辅助函数
// ============================================================

/** 应用节点变更到节点列表 */
export function applyNodeChanges(nodes: EditorNode[], changes: NodeChange[]): EditorNode[] {
  let result = [...nodes];
  
  for (const change of changes) {
    switch (change.type) {
      case 'position': {
        const index = result.findIndex(n => n.id === change.id);
        if (index !== -1) {
          result[index] = { ...result[index], position: change.position };
        }
        break;
      }
      case 'select': {
        const index = result.findIndex(n => n.id === change.id);
        if (index !== -1) {
          result[index] = { ...result[index], selected: change.selected };
        }
        break;
      }
      case 'remove': {
        result = result.filter(n => n.id !== change.id);
        break;
      }
    }
  }
  
  return result;
}

/** 应用边变更到边列表 */
export function applyEdgeChanges(edges: EditorEdge[], changes: EdgeChange[]): EditorEdge[] {
  let result = [...edges];
  
  for (const change of changes) {
    switch (change.type) {
      case 'select': {
        const index = result.findIndex(e => e.id === change.id);
        if (index !== -1) {
          result[index] = { ...result[index], selected: change.selected };
        }
        break;
      }
      case 'remove': {
        result = result.filter(e => e.id !== change.id);
        break;
      }
    }
  }
  
  return result;
}

/** 从选择变更中提取选择状态 */
export function extractSelectionFromChanges(
  nodes: EditorNode[],
  edges: EditorEdge[],
  nodeChanges: NodeChange[],
  edgeChanges: EdgeChange[]
): EditorSelection {
  // 应用变更后计算选择状态
  const updatedNodes = applyNodeChanges(nodes, nodeChanges);
  const updatedEdges = applyEdgeChanges(edges, edgeChanges);
  
  return {
    nodeIds: updatedNodes.filter(n => n.selected).map(n => n.id),
    edgeIds: updatedEdges.filter(e => e.selected).map(e => e.id),
  };
}
