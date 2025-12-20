/**
 * 编辑器状态存储
 * 
 * 统一管理编辑器的核心状态：nodes, edges, viewport, selection
 * 
 * 设计原则：
 * - 作为编辑器状态的单一数据源
 * - 与具体渲染器实现解耦
 * - 支持任意渲染器通过 INodeEditor 接口同步状态
 */

import { create } from 'zustand';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  NodeChange,
  EdgeChange,
} from '../../editor/types';
import { applyNodeChanges, applyEdgeChanges } from '../../editor/types';

/** 默认视口 */
const DEFAULT_VIEWPORT: EditorViewport = { x: 0, y: 0, zoom: 1 };

/** 默认选择 */
const DEFAULT_SELECTION: EditorSelection = { nodeIds: [], edgeIds: [] };

/**
 * 编辑器状态
 */
interface EditorState {
  /** 节点列表 */
  nodes: EditorNode[];
  /** 边列表 */
  edges: EditorEdge[];
  /** 视口状态 */
  viewport: EditorViewport;
  /** 选择状态 */
  selection: EditorSelection;
}

/**
 * 编辑器操作
 */
interface EditorActions {
  // ============================================================
  // 节点操作
  // ============================================================
  
  /** 设置节点列表（完全替换） */
  setNodes: (nodes: EditorNode[]) => void;
  
  /** 应用节点变更 */
  applyNodeChanges: (changes: NodeChange[]) => void;
  
  /** 更新单个节点 */
  updateNode: (id: string, updater: (node: EditorNode) => EditorNode) => void;
  
  /** 添加节点 */
  addNode: (node: EditorNode) => void;
  
  /** 删除节点（同时删除相关边） */
  removeNodes: (ids: string[]) => void;
  
  /** 更新节点位置 */
  updateNodePosition: (id: string, position: { x: number; y: number }) => void;
  
  // ============================================================
  // 边操作
  // ============================================================
  
  /** 设置边列表（完全替换） */
  setEdges: (edges: EditorEdge[]) => void;
  
  /** 应用边变更 */
  applyEdgeChanges: (changes: EdgeChange[]) => void;
  
  /** 添加边 */
  addEdge: (edge: EditorEdge) => void;
  
  /** 删除边 */
  removeEdges: (ids: string[]) => void;
  
  /** 删除与指定节点相关的边 */
  removeEdgesForNodes: (nodeIds: string[]) => void;
  
  // ============================================================
  // 视口操作
  // ============================================================
  
  /** 设置视口 */
  setViewport: (viewport: EditorViewport) => void;
  
  // ============================================================
  // 选择操作
  // ============================================================
  
  /** 设置选择状态 */
  setSelection: (selection: EditorSelection) => void;
  
  /** 选择节点 */
  selectNodes: (ids: string[], additive?: boolean) => void;
  
  /** 选择边 */
  selectEdges: (ids: string[], additive?: boolean) => void;
  
  /** 清除选择 */
  clearSelection: () => void;
  
  // ============================================================
  // 批量操作
  // ============================================================
  
  /** 重置编辑器状态 */
  reset: () => void;
  
  /** 批量更新（用于加载图） */
  loadGraph: (nodes: EditorNode[], edges: EditorEdge[]) => void;
}

/**
 * 编辑器存储
 */
export const useEditorStore = create<EditorState & EditorActions>()((set) => ({
  // ============================================================
  // 初始状态
  // ============================================================
  nodes: [],
  edges: [],
  viewport: DEFAULT_VIEWPORT,
  selection: DEFAULT_SELECTION,

  // ============================================================
  // 节点操作
  // ============================================================
  
  setNodes: (nodes) => set({ nodes }),
  
  applyNodeChanges: (changes) => set((state) => ({
    nodes: applyNodeChanges(state.nodes, changes),
  })),
  
  updateNode: (id, updater) => set((state) => ({
    nodes: state.nodes.map((n) => (n.id === id ? updater(n) : n)),
  })),
  
  addNode: (node) => set((state) => ({
    nodes: [...state.nodes, node],
  })),
  
  removeNodes: (ids) => {
    const idSet = new Set(ids);
    set((state) => ({
      nodes: state.nodes.filter((n) => !idSet.has(n.id)),
      edges: state.edges.filter((e) => !idSet.has(e.source) && !idSet.has(e.target)),
      selection: {
        nodeIds: state.selection.nodeIds.filter((id) => !idSet.has(id)),
        edgeIds: state.selection.edgeIds,
      },
    }));
  },
  
  updateNodePosition: (id, position) => set((state) => ({
    nodes: state.nodes.map((n) => (n.id === id ? { ...n, position } : n)),
  })),

  // ============================================================
  // 边操作
  // ============================================================
  
  setEdges: (edges) => set({ edges }),
  
  applyEdgeChanges: (changes) => set((state) => ({
    edges: applyEdgeChanges(state.edges, changes),
  })),
  
  addEdge: (edge) => set((state) => ({
    edges: [...state.edges, edge],
  })),
  
  removeEdges: (ids) => {
    const idSet = new Set(ids);
    set((state) => ({
      edges: state.edges.filter((e) => !idSet.has(e.id)),
      selection: {
        nodeIds: state.selection.nodeIds,
        edgeIds: state.selection.edgeIds.filter((id) => !idSet.has(id)),
      },
    }));
  },
  
  removeEdgesForNodes: (nodeIds) => {
    const idSet = new Set(nodeIds);
    set((state) => ({
      edges: state.edges.filter((e) => !idSet.has(e.source) && !idSet.has(e.target)),
    }));
  },

  // ============================================================
  // 视口操作
  // ============================================================
  
  setViewport: (viewport) => set({ viewport }),

  // ============================================================
  // 选择操作
  // ============================================================
  
  setSelection: (selection) => set((state) => ({
    selection,
    nodes: state.nodes.map((n) => ({
      ...n,
      selected: selection.nodeIds.includes(n.id),
    })),
    edges: state.edges.map((e) => ({
      ...e,
      selected: selection.edgeIds.includes(e.id),
    })),
  })),
  
  selectNodes: (ids, additive = false) => set((state) => {
    const newNodeIds = additive
      ? [...new Set([...state.selection.nodeIds, ...ids])]
      : ids;
    return {
      selection: { ...state.selection, nodeIds: newNodeIds },
      nodes: state.nodes.map((n) => ({
        ...n,
        selected: newNodeIds.includes(n.id),
      })),
    };
  }),
  
  selectEdges: (ids, additive = false) => set((state) => {
    const newEdgeIds = additive
      ? [...new Set([...state.selection.edgeIds, ...ids])]
      : ids;
    return {
      selection: { ...state.selection, edgeIds: newEdgeIds },
      edges: state.edges.map((e) => ({
        ...e,
        selected: newEdgeIds.includes(e.id),
      })),
    };
  }),
  
  clearSelection: () => set((state) => ({
    selection: DEFAULT_SELECTION,
    nodes: state.nodes.map((n) => ({ ...n, selected: false })),
    edges: state.edges.map((e) => ({ ...e, selected: false })),
  })),

  // ============================================================
  // 批量操作
  // ============================================================
  
  reset: () => set({
    nodes: [],
    edges: [],
    viewport: DEFAULT_VIEWPORT,
    selection: DEFAULT_SELECTION,
  }),
  
  loadGraph: (nodes, edges) => set({
    nodes,
    edges,
    selection: DEFAULT_SELECTION,
  }),
}));

/**
 * 获取选中的节点
 */
export function getSelectedNodes(state: EditorState): EditorNode[] {
  return state.nodes.filter((n) => state.selection.nodeIds.includes(n.id));
}

/**
 * 获取选中的边
 */
export function getSelectedEdges(state: EditorState): EditorEdge[] {
  return state.edges.filter((e) => state.selection.edgeIds.includes(e.id));
}
