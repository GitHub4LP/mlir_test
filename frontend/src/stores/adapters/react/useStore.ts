/**
 * React Store Adapter
 * 
 * 提供 React hook 访问 IStore 接口的 store
 * 
 * 特点：
 * - 使用 useSyncExternalStore 确保并发安全
 * - 支持选择器精确订阅，避免不必要的重渲染
 * - 与 Zustand 的 useStore 用法类似
 */

import { useSyncExternalStore, useCallback } from 'react';
import type { IStore, Selector, EqualityFn } from '../../core/IStore';
import { defaultEqualityFn } from '../../core/IStore';

/**
 * React hook for accessing store state
 * 
 * @template T - Store 状态类型
 * @template S - 选择的数据类型
 * @param store - IStore 实例
 * @param selector - 状态选择器
 * @param equalityFn - 相等性比较函数（默认 Object.is）
 * @returns 选择的状态数据
 * 
 * @example
 * ```tsx
 * const currentFunction = useStore(
 *   projectStore,
 *   (state) => state.getCurrentFunction()
 * );
 * ```
 */
export function useStore<T, S>(
  store: IStore<T>,
  selector: Selector<T, S>,
  equalityFn: EqualityFn<S> = defaultEqualityFn as EqualityFn<S>
): S {
  // getSnapshot：获取当前选择的状态
  const getSnapshot = useCallback(() => {
    return selector(store.getState());
  }, [store, selector]);
  
  // subscribe：订阅 store 变化，使用 selector 精确比较
  const subscribe = useCallback((onStoreChange: () => void) => {
    let prevSelected = selector(store.getState());
    
    return store.subscribe(() => {
      const nextSelected = selector(store.getState());
      if (!equalityFn(prevSelected, nextSelected)) {
        prevSelected = nextSelected;
        onStoreChange();
      }
    });
  }, [store, selector, equalityFn]);
  
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

/**
 * 获取 store 的直接访问（用于事件处理器中）
 * 
 * 返回的函数不会触发组件重渲染
 * 
 * @template T - Store 状态类型
 * @param store - IStore 实例
 * @returns store 访问器
 * 
 * @example
 * ```tsx
 * const storeAccess = useStoreAccess(projectStore);
 * 
 * const handleClick = () => {
 *   const state = storeAccess.getState();
 *   storeAccess.setState({ ... });
 * };
 * ```
 */
export function useStoreAccess<T>(store: IStore<T>) {
  return {
    getState: () => store.getState(),
    setState: (partial: Partial<T> | ((state: T) => Partial<T>)) => store.setState(partial),
  };
}

/**
 * 获取整个 store 状态（不推荐，会导致任何状态变化都触发重渲染）
 * 
 * @template T - Store 状态类型
 * @param store - IStore 实例
 * @returns 完整的 store 状态
 */
export function useStoreState<T>(store: IStore<T>): T {
  return useStore(store, (state) => state);
}
