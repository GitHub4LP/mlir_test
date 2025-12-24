/**
 * useEditorStoreUpdate composable (Vue 版本)
 * 
 * 让 VueFlow 节点组件直接更新 editorStore，而不是修改 VueFlow 内部状态。
 * 
 * 数据流：
 * 用户操作 → editorStore.updateNode() → editorStore 变化 
 *          → EditorContainer useEffect → editor.setNodes() 
 *          → VueFlow 更新 UI
 * 
 * 这样实现了"数据一份，订阅更新"的模式：
 * - editorStore 是唯一的数据源
 * - VueFlow 只是视图层，不维护自己的状态副本
 * - 所有数据变更都通过 editorStore 进行
 */

import { useEditorStore } from '../../../core/stores/editorStore';

/**
 * 节点数据更新器类型
 */
export type NodeDataUpdater<T = Record<string, unknown>> = (
  currentData: T
) => T;

/**
 * useEditorStoreUpdate composable 返回值
 */
export interface UseEditorStoreUpdateResult<T = Record<string, unknown>> {
  /**
   * 更新节点数据（函数式更新）
   * 
   * @param updater - 数据更新函数，接收当前数据，返回新数据
   */
  updateNodeData: (updater: NodeDataUpdater<T>) => void;
  
  /**
   * 直接设置节点数据（部分更新）
   * 
   * @param partialData - 要合并的部分数据
   */
  setNodeData: (partialData: Partial<T>) => void;
}

/**
 * 直接更新 editorStore 的 composable
 * 
 * 节点组件使用此 composable 更新数据，数据变更会自动同步到 VueFlow。
 * 
 * @param nodeId - 节点 ID
 * @returns updateNodeData 和 setNodeData 函数
 */
export function useEditorStoreUpdate<T extends Record<string, unknown> = Record<string, unknown>>(
  nodeId: string
): UseEditorStoreUpdateResult<T> {
  // 直接从 Zustand store 获取 updateNode action
  // 注意：这里不使用 Vue 的响应式，因为我们只需要调用 action
  const { updateNode } = useEditorStore.getState();
  
  const updateNodeData = (updater: NodeDataUpdater<T>) => {
    updateNode(nodeId, (node) => {
      const currentData = node.data as T;
      const newData = updater(currentData);
      return {
        ...node,
        data: newData,
      };
    });
  };
  
  const setNodeData = (partialData: Partial<T>) => {
    updateNodeData(currentData => ({
      ...currentData,
      ...partialData,
    }));
  };
  
  return { updateNodeData, setNodeData };
}

export default useEditorStoreUpdate;
