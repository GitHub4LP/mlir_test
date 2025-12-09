/**
 * Type State Store
 * 
 * 管理端口的具体类型，用于连接验证和边颜色。
 * 
 * 注意：类型传播现在由 typePropagation 模块处理，
 * 用户选择的类型存储在节点数据的 pinnedTypes 中。
 * 这个 store 主要用于：
 * 1. 连接验证时获取端口类型
 * 2. 边颜色计算
 * 3. FunctionCallNode 的类型管理（待迁移到传播模型）
 */

import { create } from 'zustand';

/**
 * Type store state interface
 */
interface TypeState {
  // Map of nodeId -> portId -> concrete type
  resolvedTypes: Map<string, Map<string, string>>;
}

/**
 * Type store actions interface
 */
interface TypeActions {
  // Type management
  setPortType: (nodeId: string, portId: string, type: string) => void;
  clearPortType: (nodeId: string, portId: string) => void;
  clearNodeTypes: (nodeId: string) => void;
  
  // Query operations
  getPortType: (nodeId: string, portId: string) => string | undefined;
  getNodeTypes: (nodeId: string) => Map<string, string> | undefined;
  
  // Bulk operations
  clearAll: () => void;
}

export type TypeStore = TypeState & TypeActions;

/**
 * Type state store using Zustand
 */
export const useTypeStore = create<TypeStore>((set, get) => ({
  // Initial state
  resolvedTypes: new Map(),

  // Type management
  setPortType: (nodeId, portId, type) => {
    set((state) => {
      const newResolvedTypes = new Map(state.resolvedTypes);
      
      // Get or create the node's type map
      let nodeTypes = newResolvedTypes.get(nodeId);
      if (!nodeTypes) {
        nodeTypes = new Map();
        newResolvedTypes.set(nodeId, nodeTypes);
      } else {
        nodeTypes = new Map(nodeTypes);
        newResolvedTypes.set(nodeId, nodeTypes);
      }
      
      nodeTypes.set(portId, type);
      
      return { resolvedTypes: newResolvedTypes };
    });
  },

  clearPortType: (nodeId, portId) => {
    set((state) => {
      const newResolvedTypes = new Map(state.resolvedTypes);
      const nodeTypes = newResolvedTypes.get(nodeId);
      
      if (nodeTypes) {
        const newNodeTypes = new Map(nodeTypes);
        newNodeTypes.delete(portId);
        
        if (newNodeTypes.size === 0) {
          newResolvedTypes.delete(nodeId);
        } else {
          newResolvedTypes.set(nodeId, newNodeTypes);
        }
      }
      
      return { resolvedTypes: newResolvedTypes };
    });
  },

  clearNodeTypes: (nodeId) => {
    set((state) => {
      const newResolvedTypes = new Map(state.resolvedTypes);
      newResolvedTypes.delete(nodeId);
      return { resolvedTypes: newResolvedTypes };
    });
  },

  // Query operations
  getPortType: (nodeId, portId) => {
    const nodeTypes = get().resolvedTypes.get(nodeId);
    return nodeTypes?.get(portId);
  },

  getNodeTypes: (nodeId) => {
    return get().resolvedTypes.get(nodeId);
  },

  // Bulk operations
  clearAll: () => {
    set({ resolvedTypes: new Map() });
  },
}));
