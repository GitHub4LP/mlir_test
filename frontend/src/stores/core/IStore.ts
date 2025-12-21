/**
 * 框架无关的 Store 接口
 * 
 * 设计原则：
 * - 最小化接口，只包含必要方法
 * - 与 Zustand API 兼容，便于迁移
 * - 支持选择器模式，避免不必要的重渲染
 * - 不依赖任何 UI 框架（React、Vue、Svelte 等）
 * 
 * 使用方式：
 * - React 组件：使用 adapters/react/useStore.ts
 * - Vue 组件：使用 adapters/vue/useStore.ts
 * - Vanilla JS：使用 adapters/vanilla/subscribeStore.ts
 */

/**
 * 框架无关的 Store 接口
 * 
 * @template T - Store 状态类型
 */
export interface IStore<T> {
  /**
   * 获取当前状态快照（同步）
   * 
   * 注意：返回的是状态快照，不是响应式对象
   */
  getState(): T;
  
  /**
   * 更新状态
   * 
   * @param partial - 部分状态或更新函数
   */
  setState(partial: Partial<T> | ((state: T) => Partial<T>)): void;
  
  /**
   * 订阅状态变化
   * 
   * @param listener - 状态变化监听器
   * @returns 取消订阅函数
   */
  subscribe(listener: StoreListener<T>): () => void;
}

/**
 * Store 状态变化监听器
 * 
 * @template T - Store 状态类型
 */
export type StoreListener<T> = (state: T, prevState: T) => void;

/**
 * 状态选择器
 * 
 * 用于从完整状态中选择部分数据，避免不必要的重渲染
 * 
 * @template T - Store 状态类型
 * @template S - 选择的数据类型
 */
export type Selector<T, S> = (state: T) => S;

/**
 * 相等性比较函数
 * 
 * 用于判断选择的数据是否变化，默认使用 Object.is
 * 
 * @template S - 比较的数据类型
 */
export type EqualityFn<S> = (a: S, b: S) => boolean;

/**
 * 默认相等性比较函数
 */
export const defaultEqualityFn: EqualityFn<unknown> = Object.is;

/**
 * 浅比较函数
 * 
 * 用于对象/数组的浅层比较
 */
export function shallowEqual<T>(a: T, b: T): boolean {
  if (Object.is(a, b)) return true;
  
  if (
    typeof a !== 'object' || a === null ||
    typeof b !== 'object' || b === null
  ) {
    return false;
  }
  
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  
  if (keysA.length !== keysB.length) return false;
  
  for (const key of keysA) {
    if (
      !Object.prototype.hasOwnProperty.call(b, key) ||
      !Object.is((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key])
    ) {
      return false;
    }
  }
  
  return true;
}
