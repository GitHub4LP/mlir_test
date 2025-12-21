/**
 * Vue Store Adapter
 * 
 * 提供 Vue composable 访问 IStore 接口的 store
 * 
 * 特点：
 * - 返回 Vue Ref，实现响应式
 * - 支持选择器精确订阅
 * - 正确处理组件生命周期
 */

import { shallowRef, onMounted, onUnmounted, type Ref } from 'vue';
import type { IStore, Selector, EqualityFn } from '../../core/IStore';
import { defaultEqualityFn } from '../../core/IStore';

/**
 * Vue composable for accessing store state
 * 
 * @template T - Store 状态类型
 * @template S - 选择的数据类型
 * @param store - IStore 实例
 * @param selector - 状态选择器
 * @param equalityFn - 相等性比较函数（默认 Object.is）
 * @returns Vue Ref 包装的选择数据
 * 
 * @example
 * ```vue
 * <script setup>
 * const currentFunction = useStore(
 *   projectStore,
 *   (state) => state.getCurrentFunction()
 * );
 * </script>
 * 
 * <template>
 *   <div>{{ currentFunction?.name }}</div>
 * </template>
 * ```
 */
export function useStore<T, S>(
  store: IStore<T>,
  selector: Selector<T, S>,
  equalityFn: EqualityFn<S> = defaultEqualityFn as EqualityFn<S>
): Ref<S> {
  // 使用 shallowRef 避免深层响应式转换
  const state = shallowRef(selector(store.getState())) as Ref<S>;
  let unsubscribe: (() => void) | null = null;
  
  const updateState = () => {
    const nextState = selector(store.getState());
    if (!equalityFn(state.value, nextState)) {
      state.value = nextState;
    }
  };
  
  onMounted(() => {
    // 确保初始状态正确（可能在 setup 和 mounted 之间有变化）
    updateState();
    // 订阅 store 变化
    unsubscribe = store.subscribe(updateState);
  });
  
  onUnmounted(() => {
    unsubscribe?.();
  });
  
  return state;
}

/**
 * 获取 store 的直接访问（用于事件处理器中）
 * 
 * 返回的对象不是响应式的，用于命令式操作
 * 
 * @template T - Store 状态类型
 * @param store - IStore 实例
 * @returns store 访问器
 * 
 * @example
 * ```vue
 * <script setup>
 * const storeAccess = useStoreAccess(projectStore);
 * 
 * function handleClick() {
 *   const state = storeAccess.getState();
 *   storeAccess.setState({ ... });
 * }
 * </script>
 * ```
 */
export function useStoreAccess<T>(store: IStore<T>) {
  return {
    getState: () => store.getState(),
    setState: (partial: Partial<T> | ((state: T) => Partial<T>)) => store.setState(partial),
  };
}

/**
 * 立即获取 store 状态（非响应式）
 * 
 * 用于在 setup 阶段需要立即获取状态的场景
 * 
 * @template T - Store 状态类型
 * @template S - 选择的数据类型
 * @param store - IStore 实例
 * @param selector - 状态选择器
 * @returns 选择的状态数据（非响应式）
 */
export function getStoreSnapshot<T, S>(
  store: IStore<T>,
  selector: Selector<T, S>
): S {
  return selector(store.getState());
}
