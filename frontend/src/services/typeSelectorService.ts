/**
 * TypeSelector 数据服务
 * 
 * 将类型选择器的数据逻辑从渲染层分离，供所有渲染器复用：
 * - VueFlow
 * - ReactFlow  
 * - Canvas
 * - GPU
 */

import type { WrapperInfo } from './typeNodeUtils';
import { WRAPPERS } from './typeNodeUtils';

// ============ 类型定义 ============

export interface TypeGroup {
  label: string;
  items: string[];
}

export interface ConstraintAnalysis {
  /** 是否无约束或 AnyType */
  isUnconstrained: boolean;
  /** 是否是复合类型约束 */
  isCompositeConstraint: boolean;
  /** 允许的包装器名称（null 表示所有，[] 表示不允许） */
  allowedWrappers: string[] | null;
  /** 约束对应的具体标量类型（null 表示无限制） */
  scalarTypes: string[] | null;
}

export interface SearchFilter {
  searchText: string;
  showConstraints: boolean;
  showTypes: boolean;
  useRegex: boolean;
}

export interface TypeSelectorData {
  /** 约束分析结果 */
  analysis: ConstraintAnalysis;
  /** 允许的包装器列表 */
  allowedWrappers: readonly WrapperInfo[];
  /** 约束对应的标量类型（用于选择面板） */
  constraintTypes: string[] | null;
}

// ============ 搜索匹配器 ============

/**
 * 创建搜索匹配函数
 */
export function createSearchMatcher(
  searchText: string,
  useRegex: boolean
): (item: string) => boolean {
  if (!searchText) {
    return () => true;
  }
  
  if (useRegex) {
    try {
      const regex = new RegExp(searchText, 'i');
      return (item) => regex.test(item);
    } catch {
      return () => false;
    }
  }
  
  const lower = searchText.toLowerCase();
  return (item) => item.toLowerCase().includes(lower);
}

// ============ 类型分组构建 ============

/**
 * 构建类型分组（无约束时使用）
 */
export function buildTypeGroups(
  constraintDefs: Map<string, { name: string; summary: string; rule: unknown }>,
  buildableTypes: string[],
  filter: SearchFilter
): TypeGroup[] {
  const { searchText, showConstraints, showTypes, useRegex } = filter;
  const buildableSet = new Set(buildableTypes);
  const groups: TypeGroup[] = [];
  const matcher = createSearchMatcher(searchText, useRegex);

  // 1. 内置约束（排除具体类型）
  if (showConstraints) {
    const items = [...constraintDefs.keys()]
      .filter(key => !buildableSet.has(key))
      .filter(matcher)
      .sort();
    if (items.length > 0) {
      groups.push({ label: '约束', items });
    }
  }

  // 2. 内置类型
  if (showTypes) {
    const items = buildableTypes.filter(matcher).sort();
    if (items.length > 0) {
      groups.push({ label: '类型', items });
    }
  }

  return groups;
}

/**
 * 构建约束限制下的类型分组
 */
export function buildConstrainedTypeGroups(
  expandedTypes: string[],
  constraintName: string | undefined,
  searchText: string,
  useRegex: boolean
): TypeGroup[] {
  const matcher = createSearchMatcher(searchText, useRegex);
  
  const allItems = new Set<string>();
  if (constraintName) allItems.add(constraintName);
  expandedTypes.forEach(t => allItems.add(t));

  const items = [...allItems].filter(matcher).sort((a, b) => {
    // 约束名排在最前
    if (a === constraintName) return -1;
    if (b === constraintName) return 1;
    return a.localeCompare(b);
  });

  if (items.length > 0) {
    return [{ label: constraintName || '可选类型', items }];
  }
  return [];
}

// ============ 约束展开 ============

/**
 * 展开约束名到具体类型列表
 */
export function expandConstraintTypes(
  constraintTypes: string[],
  buildableTypes: string[],
  getConstraintElements: (name: string) => string[]
): string[] {
  const buildableSet = new Set(buildableTypes);
  const expanded = new Set<string>();

  for (const t of constraintTypes) {
    if (buildableSet.has(t)) {
      expanded.add(t);
      continue;
    }
    // 展开约束到集合元素
    const elements = getConstraintElements(t);
    if (elements.length > 0) {
      elements.forEach(ct => expanded.add(ct));
    } else {
      expanded.add(t);
    }
  }

  return [...expanded];
}

// ============ 约束分析 ============

/**
 * 分析约束类型，返回约束的特性
 * 
 * 约束分类：
 * 1. 无约束/AnyType：允许所有类型（标量 + 复合），显示完整选择面板
 * 2. 标量约束（如 SignlessIntegerLike）：只允许特定标量类型，不显示包装选项
 * 3. 复合类型约束（如 AnyTensor）：允许构建特定复合类型
 * 4. 元素类型约束（如 AnyTensorOf<[F32]>）：允许构建复合类型，但限制元素类型
 */
export function analyzeConstraint(
  constraint: string | undefined,
  getConstraintElements: (name: string) => string[],
  isShapedConstraint: (name: string) => boolean,
  getAllowedContainers: (name: string) => string[]
): ConstraintAnalysis {
  // 无约束或 AnyType
  if (!constraint || constraint === 'AnyType') {
    return {
      isUnconstrained: true,
      isCompositeConstraint: false,
      allowedWrappers: null,
      scalarTypes: null,
    };
  }

  // Variadic<...> 类型：解析内部类型
  const variadicMatch = constraint.match(/^Variadic<(.+)>$/);
  if (variadicMatch) {
    const innerConstraint = variadicMatch[1];
    return analyzeConstraint(innerConstraint, getConstraintElements, isShapedConstraint, getAllowedContainers);
  }

  // AnyOf<...> 类型：合成约束
  if (constraint.startsWith('AnyOf<')) {
    return {
      isUnconstrained: false,
      isCompositeConstraint: false,
      allowedWrappers: [],
      scalarTypes: null,
    };
  }

  // 使用 store 的方法判断
  const isShaped = isShapedConstraint(constraint);
  if (isShaped) {
    const containers = getAllowedContainers(constraint);
    const scalarTypes = getConstraintElements(constraint);
    return {
      isUnconstrained: false,
      isCompositeConstraint: true,
      allowedWrappers: containers.length > 0 ? containers : ['tensor', 'memref', 'vector'],
      scalarTypes: scalarTypes.length > 0 ? scalarTypes : null,
    };
  }

  // 标量约束
  const scalarTypes = getConstraintElements(constraint);
  return {
    isUnconstrained: false,
    isCompositeConstraint: false,
    allowedWrappers: [],
    scalarTypes: scalarTypes.length > 0 ? scalarTypes : [constraint],
  };
}

// ============ 计算允许的包装器 ============

/**
 * 根据约束分析结果计算允许的包装器列表
 */
export function computeAllowedWrappers(analysis: ConstraintAnalysis): readonly WrapperInfo[] {
  if (analysis.allowedWrappers === null) return WRAPPERS;
  return WRAPPERS.filter(w => analysis.allowedWrappers!.includes(w.name));
}

// ============ 完整的类型选择器数据计算 ============

export interface TypeSelectorInput {
  constraint?: string;
  allowedTypes?: string[];  // 来自后端 AnyTypeOf 解析
  buildableTypes: string[];
  constraintDefs: Map<string, { name: string; summary: string; rule: unknown }>;
  getConstraintElements: (name: string) => string[];
  isShapedConstraint: (name: string) => boolean;
  getAllowedContainers: (name: string) => string[];
}

/**
 * 计算类型选择器所需的全部数据
 */
export function computeTypeSelectorData(input: TypeSelectorInput): TypeSelectorData {
  const {
    constraint,
    allowedTypes,
    getConstraintElements,
    isShapedConstraint,
    getAllowedContainers,
  } = input;

  // 1. 分析约束
  const analysis = analyzeConstraint(
    constraint,
    getConstraintElements,
    isShapedConstraint,
    getAllowedContainers
  );

  // 2. 计算允许的包装器
  const allowedWrappers = computeAllowedWrappers(analysis);

  // 3. 约束对应的标量类型
  // 优先使用 prop 传入的 allowedTypes（来自后端 AnyTypeOf 解析）
  const constraintTypes = allowedTypes || analysis.scalarTypes;

  return {
    analysis,
    allowedWrappers,
    constraintTypes,
  };
}

/**
 * 计算类型分组（用于选择面板）
 */
export function computeTypeGroups(
  data: TypeSelectorData,
  filter: SearchFilter,
  constraintName: string | undefined,
  buildableTypes: string[],
  constraintDefs: Map<string, { name: string; summary: string; rule: unknown }>,
  getConstraintElements: (name: string) => string[]
): TypeGroup[] {
  const { constraintTypes } = data;
  const { searchText, useRegex } = filter;

  // 是否有约束限制（AnyType 视为无约束）
  const hasConstraint = constraintTypes !== null && 
    constraintName !== 'AnyType' && 
    constraintName !== undefined;

  if (hasConstraint && constraintTypes && constraintTypes.length > 0) {
    // 展开约束到具体类型
    const expandedTypes = expandConstraintTypes(
      constraintTypes,
      buildableTypes,
      getConstraintElements
    );
    
    if (expandedTypes.length > 0) {
      return buildConstrainedTypeGroups(expandedTypes, constraintName, searchText, useRegex);
    }
    return [];
  }

  // 无约束，显示所有
  return buildTypeGroups(constraintDefs, buildableTypes, filter);
}

/**
 * 判断是否有约束限制（用于 UI 显示过滤按钮）
 */
export function hasConstraintLimit(
  constraintTypes: string[] | null,
  constraintName: string | undefined
): boolean {
  return constraintTypes !== null && 
    constraintName !== 'AnyType' && 
    constraintName !== undefined;
}
