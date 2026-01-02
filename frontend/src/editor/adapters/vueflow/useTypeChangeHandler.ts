/**
 * useTypeChangeHandler composable (Vue 版本)
 * 
 * 统一的类型变更处理逻辑，供所有 VueFlow 节点类型使用。
 * 与 React 版本的 useTypeChangeHandler hook 保持一致的行为。
 * 
 * 设计原则：
 * - 只更新 editorStore（唯一数据源）
 * - VueFlow 通过订阅 editorStore 自动更新
 * - 实现"数据一份，订阅更新"的模式
 */

import { useEditorStore } from '../../../core/stores/editorStore';
import { projectStore, typeConstraintStore } from '../../../stores';
import { handlePinnedTypeChange, type TypeChangeHandlerDeps } from '../../../services/typeChangeHandler';

/**
 * 统一的类型变更处理 composable
 * 
 * 只更新 editorStore，VueFlow 通过订阅自动更新。
 * 
 * @param nodeId - 节点 ID
 * @returns handleTypeChange 回调
 */
export function useTypeChangeHandler(nodeId: string) {
  // 类型变更处理回调
  const handleTypeChange = (portId: string, type: string, originalConstraint?: string) => {
    // 从 stores 获取依赖项
    const getCurrentFunction = projectStore.getState().getCurrentFunction;
    const getConstraintElements = typeConstraintStore.getState().getConstraintElements;
    const pickConstraintName = typeConstraintStore.getState().pickConstraintName;
    const findSubsetConstraints = typeConstraintStore.getState().findSubsetConstraints;
    
    // 从 editorStore 获取当前状态
    const currentEditorNodes = useEditorStore.getState().nodes;
    const edges = useEditorStore.getState().edges;
    
    // 构建类型变更依赖项
    const typeChangeDeps: TypeChangeHandlerDeps = {
      edges,
      getCurrentFunction,
      getConstraintElements,
      pickConstraintName,
      findSubsetConstraints,
    };
    
    // 执行类型变更处理（包含类型传播）
    const updatedEditorNodes = handlePinnedTypeChange(
      nodeId, portId, type, originalConstraint, currentEditorNodes, typeChangeDeps
    );
    
    // 只更新 editorStore，VueFlow 会自动同步
    useEditorStore.getState().setNodes(updatedEditorNodes);
  };

  return { handleTypeChange };
}

export default useTypeChangeHandler;
