/**
 * Port State Store
 * 
 * 存储每个端口的业务状态（displayType, canEdit 等）。
 * 由类型传播系统更新，供所有渲染器读取。
 * 
 * 设计原则：
 * - 单一数据源：所有渲染器读取同一份数据
 * - 业务层输出：canEdit 在类型传播时计算，不在渲染时计算
 * - 关注点分离：渲染器只负责读取和显示
 */

import { create } from 'zustand';

/** 端口状态 */
export interface PortState {
  /** 当前显示的类型 */
  displayType: string;
  /** 原始约束 */
  constraint: string;
  /** 是否可编辑 */
  canEdit: boolean;
}

/** 端口状态 Store */
interface PortStateStore {
  /** 端口状态映射，key = `${nodeId}:${handleId}` */
  states: Map<string, PortState>;
  
  /** 获取端口状态 */
  getPortState: (nodeId: string, handleId: string) => PortState | undefined;
  
  /** 批量更新端口状态 */
  updatePortStates: (updates: Map<string, PortState>) => void;
  
  /** 清除指定节点的端口状态 */
  clearNodePortStates: (nodeId: string) => void;
  
  /** 清除所有端口状态 */
  clearAll: () => void;
}

/** 生成端口状态的 key */
export function makePortKey(nodeId: string, handleId: string): string {
  return `${nodeId}:${handleId}`;
}

/** 解析端口状态的 key */
export function parsePortKey(key: string): { nodeId: string; handleId: string } | null {
  const idx = key.indexOf(':');
  if (idx === -1) return null;
  return {
    nodeId: key.slice(0, idx),
    handleId: key.slice(idx + 1),
  };
}

export const usePortStateStore = create<PortStateStore>((set, get) => ({
  states: new Map(),
  
  getPortState: (nodeId, handleId) => {
    const key = makePortKey(nodeId, handleId);
    return get().states.get(key);
  },
  
  updatePortStates: (updates) => {
    set((state) => {
      const newStates = new Map(state.states);
      for (const [key, portState] of updates) {
        newStates.set(key, portState);
      }
      return { states: newStates };
    });
  },
  
  clearNodePortStates: (nodeId) => {
    set((state) => {
      const newStates = new Map(state.states);
      const prefix = `${nodeId}:`;
      for (const key of newStates.keys()) {
        if (key.startsWith(prefix)) {
          newStates.delete(key);
        }
      }
      return { states: newStates };
    });
  },
  
  clearAll: () => {
    set({ states: new Map() });
  },
}));
