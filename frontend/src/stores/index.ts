/**
 * Store 统一导出
 * 
 * 提供框架无关的 store 访问方式
 * 
 * 使用方式：
 * - React 组件：import { useReactStore, projectStore } from '@/stores';
 * - Vue 组件：import { useVueStore, projectStore } from '@/stores';
 * - Vanilla JS：import { subscribeStore, projectStore } from '@/stores';
 */

// ============================================================
// 核心类型
// ============================================================

export type { IStore, Selector, EqualityFn, StoreListener } from './core/IStore';
export { defaultEqualityFn, shallowEqual } from './core/IStore';

// ============================================================
// 原始 Zustand Stores（向后兼容，逐步废弃）
// 这些导出必须在包装 store 之前，避免循环依赖
// ============================================================

export { useProjectStore } from './projectStore';
export { useTypeConstraintStore } from './typeConstraintStore';
export { useDialectStore } from './dialectStore';

// 重新导出类型
export type { ProjectStore } from './projectStore';
export type { TypeConstraintState, ConstraintDef, TypeDefinition } from './typeConstraintStore';
export type { DialectState } from './dialectStore';

// ============================================================
// 包装后的 Store 实例
// ============================================================

export { projectStore, typeConstraintStore, dialectStore } from './wrappedStores';

// ============================================================
// React Adapter
// ============================================================

export { useStore as useReactStore, useStoreAccess as useReactStoreAccess } from './adapters/react/useStore';

// ============================================================
// Vue Adapter
// ============================================================

export { useStore as useVueStore, useStoreAccess as useVueStoreAccess, getStoreSnapshot } from './adapters/vue/useStore';

// ============================================================
// Vanilla Adapter
// ============================================================

export { subscribeStore, createStoreAccessor, batchSubscribe } from './adapters/vanilla/subscribeStore';
