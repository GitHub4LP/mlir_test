/**
 * Vanilla JS Store Adapter
 * 
 * 提供原生 JavaScript 访问 IStore 接口的 store
 * 
 * 用于：
 * - Canvas 渲染器
 * - WebGL/WebGPU 渲染器
 * - 任何非框架代码
 */

import type { IStore, Selector, EqualityFn } from '../../core/IStore';
import { defaultEqualityFn } from '../../core/IStore';

/**
 * 订阅 store 状态变化
 * 
 * @template T - Store 状态类型
 * @template S - 选择的数据类型
 * @param store - IStore 实例
 * @param selector - 状态选择器
 * @param callback - 状态变化回调
 * @param equalityFn - 相等性比较函数（默认 Object.is）
 * @returns 取消订阅函数
 * 
 * @example
 * ```ts
 * const unsubscribe = subscribeStore(
 *   projectStore,
 *   (state) => state.getCurrentFunction()?.graph,
 *   (graph, prevGraph) => {
 *     console.log('Graph changed:', graph);
 *     this.renderGraph(graph);
 *   }
 * );
 * 
 * // 清理时调用
 * unsubscribe();
 * ```
 */
export function subscribeStore<T, S>(
  store: IStore<T>,
  selector: Selector<T, S>,
  callback: (value: S, prevValue: S) => void,
  equalityFn: EqualityFn<S> = defaultEqualityFn as EqualityFn<S>
): () => void {
  let prevValue = selector(store.getState());
  
  // 立即调用一次（初始值）
  callback(prevValue, prevValue);
  
  // 订阅变化
  return store.subscribe((state) => {
    const nextValue = selector(state);
    if (!equalityFn(prevValue, nextValue)) {
      const prev = prevValue;
      prevValue = nextValue;
      callback(nextValue, prev);
    }
  });
}

/**
 * 创建 store 访问器
 * 
 * 提供命令式的 store 访问方式
 * 
 * @template T - Store 状态类型
 * @param store - IStore 实例
 * @returns store 访问器对象
 * 
 * @example
 * ```ts
 * const accessor = createStoreAccessor(projectStore);
 * 
 * // 读取状态
 * const state = accessor.get();
 * 
 * // 更新状态
 * accessor.set({ currentFunctionId: 'main' });
 * 
 * // 订阅变化
 * const unsubscribe = accessor.subscribe((state, prevState) => {
 *   console.log('State changed');
 * });
 * ```
 */
export function createStoreAccessor<T>(store: IStore<T>) {
  return {
    /**
     * 获取当前状态
     */
    get: () => store.getState(),
    
    /**
     * 更新状态
     */
    set: (partial: Partial<T> | ((state: T) => Partial<T>)) => store.setState(partial),
    
    /**
     * 订阅状态变化
     */
    subscribe: (listener: (state: T, prevState: T) => void) => store.subscribe(listener),
    
    /**
     * 选择并订阅部分状态
     */
    select: <S>(
      selector: Selector<T, S>,
      callback: (value: S, prevValue: S) => void,
      equalityFn?: EqualityFn<S>
    ) => subscribeStore(store, selector, callback, equalityFn),
  };
}

/**
 * 批量订阅多个 store
 * 
 * @param subscriptions - 订阅配置数组
 * @returns 取消所有订阅的函数
 * 
 * @example
 * ```ts
 * const unsubscribeAll = batchSubscribe([
 *   {
 *     store: projectStore,
 *     selector: (s) => s.currentFunctionId,
 *     callback: (id) => console.log('Function changed:', id),
 *   },
 *   {
 *     store: typeConstraintStore,
 *     selector: (s) => s.buildableTypes,
 *     callback: (types) => console.log('Types loaded:', types.length),
 *   },
 * ]);
 * 
 * // 清理时调用
 * unsubscribeAll();
 * ```
 */
export function batchSubscribe<Subscriptions extends Array<{
  store: IStore<unknown>;
  selector: Selector<unknown, unknown>;
  callback: (value: unknown, prevValue: unknown) => void;
  equalityFn?: EqualityFn<unknown>;
}>>(subscriptions: Subscriptions): () => void {
  const unsubscribes = subscriptions.map(({ store, selector, callback, equalityFn }) =>
    subscribeStore(store, selector, callback, equalityFn)
  );
  
  return () => {
    unsubscribes.forEach(unsub => unsub());
  };
}
