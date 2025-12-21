/**
 * Zustand Store 包装器
 * 
 * 将 Zustand store 包装为 IStore 接口
 * 
 * Zustand 的 API 已经符合 IStore 接口，只需要类型适配
 */

import type { IStore, StoreListener } from './core/IStore';
import type { StoreApi } from 'zustand';

/**
 * 将 Zustand store 包装为 IStore 接口
 * 
 * @template T - Store 状态类型
 * @param zustandStore - Zustand store 实例
 * @returns IStore 接口实例
 * 
 * @example
 * ```ts
 * import { create } from 'zustand';
 * import { wrapZustandStore } from './wrapZustandStore';
 * 
 * const useMyStore = create<MyState>((set, get) => ({
 *   // ...
 * }));
 * 
 * export const myStore = wrapZustandStore(useMyStore);
 * ```
 */
export function wrapZustandStore<T>(zustandStore: StoreApi<T>): IStore<T> {
  return {
    getState: () => zustandStore.getState(),
    
    setState: (partial) => {
      if (typeof partial === 'function') {
        zustandStore.setState(partial as (state: T) => Partial<T>);
      } else {
        zustandStore.setState(partial);
      }
    },
    
    subscribe: (listener: StoreListener<T>) => {
      // Zustand 的 subscribe 签名与 IStore 兼容
      return zustandStore.subscribe(listener);
    },
  };
}

/**
 * 将 Zustand hook store 包装为 IStore 接口
 * 
 * Zustand 的 create() 返回的是一个 hook，但它也有 getState/setState/subscribe 方法
 * 
 * @template T - Store 状态类型
 * @param zustandHookStore - Zustand hook store（create() 的返回值）
 * @returns IStore 接口实例
 */
export function wrapZustandHookStore<T>(
  zustandHookStore: {
    getState: () => T;
    setState: (partial: Partial<T> | ((state: T) => Partial<T>)) => void;
    subscribe: (listener: (state: T, prevState: T) => void) => () => void;
  }
): IStore<T> {
  return {
    getState: () => zustandHookStore.getState(),
    setState: (partial) => zustandHookStore.setState(partial),
    subscribe: (listener) => zustandHookStore.subscribe(listener),
  };
}
