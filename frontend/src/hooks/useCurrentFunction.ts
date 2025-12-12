/**
 * useCurrentFunction hook
 * 
 * 统一的当前函数订阅逻辑，供 FunctionEntryNode 和 FunctionReturnNode 使用。
 * 直接订阅函数数据（而非函数引用），确保数据变化时组件重新渲染。
 */

import { useProjectStore } from '../stores/projectStore';

/**
 * 订阅当前函数数据
 * 
 * 与 getCurrentFunction() 的区别：
 * - getCurrentFunction() 返回函数引用，数据变化不触发重渲染
 * - useCurrentFunction() 直接订阅数据，数据变化触发重渲染
 */
export function useCurrentFunction() {
  return useProjectStore(state => {
    if (!state.project || !state.currentFunctionId) return null;
    if (state.project.mainFunction.id === state.currentFunctionId) {
      return state.project.mainFunction;
    }
    return state.project.customFunctions.find(f => f.id === state.currentFunctionId) || null;
  });
}

