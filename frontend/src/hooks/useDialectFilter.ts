/**
 * 方言过滤 Hook
 * 
 * 提供方言过滤配置，用于类型传播时按方言过滤 options。
 */

import { useCallback, useMemo } from 'react';
import { useProjectStore } from '../stores/projectStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { computeReachableDialects } from '../services/dialectDependency';
import type { DialectFilterConfig } from '../services/typePropagation';

/**
 * 获取方言过滤配置
 * 
 * 返回一个 DialectFilterConfig 对象，包含：
 * - getReachableDialects: 获取函数的可达方言集
 * - filterConstraintsByDialects: 按方言过滤约束名
 */
export function useDialectFilter(): DialectFilterConfig | undefined {
  const project = useProjectStore(state => state.project);
  const filterConstraintsByDialects = useTypeConstraintStore(state => state.filterConstraintsByDialects);
  
  // 获取函数的可达方言集
  const getReachableDialects = useCallback((functionId: string): string[] => {
    if (!project) return [];
    return computeReachableDialects(functionId, project);
  }, [project]);
  
  // 构建 DialectFilterConfig
  const dialectFilter = useMemo((): DialectFilterConfig | undefined => {
    if (!project) return undefined;
    
    return {
      getReachableDialects,
      filterConstraintsByDialects,
    };
  }, [project, getReachableDialects, filterConstraintsByDialects]);
  
  return dialectFilter;
}

/**
 * 创建方言过滤配置（非 hook 版本）
 * 
 * 用于不在 React 组件中的场景，如事件处理函数。
 */
export function createDialectFilterConfig(): DialectFilterConfig | undefined {
  const project = useProjectStore.getState().project;
  const filterConstraintsByDialects = useTypeConstraintStore.getState().filterConstraintsByDialects;
  
  if (!project) return undefined;
  
  return {
    getReachableDialects: (functionId: string) => computeReachableDialects(functionId, project),
    filterConstraintsByDialects,
  };
}
