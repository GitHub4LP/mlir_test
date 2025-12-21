/**
 * 包装后的 Store 实例
 * 
 * 将 Zustand stores 包装为 IStore 接口
 * 
 * 注意：这个文件单独存在是为了避免循环依赖
 * index.ts 先导出原始 stores，再从这里导出包装后的 stores
 */

import { wrapZustandHookStore } from './wrapZustandStore';
import { useProjectStore } from './projectStore';
import { useTypeConstraintStore } from './typeConstraintStore';
import { useDialectStore } from './dialectStore';
import type { IStore } from './core/IStore';
import type { ProjectStore } from './projectStore';
import type { TypeConstraintState } from './typeConstraintStore';
import type { DialectState } from './dialectStore';

/**
 * 项目状态 Store（IStore 接口）
 */
export const projectStore: IStore<ProjectStore> = wrapZustandHookStore(useProjectStore);

/**
 * 类型约束 Store（IStore 接口）
 */
export const typeConstraintStore: IStore<TypeConstraintState> = wrapZustandHookStore(useTypeConstraintStore);

/**
 * 方言 Store（IStore 接口）
 */
export const dialectStore: IStore<DialectState> = wrapZustandHookStore(useDialectStore);
