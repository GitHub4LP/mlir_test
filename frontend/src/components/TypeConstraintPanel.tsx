/**
 * TypeConstraintPanel 组件
 * 
 * 类型约束浏览器，以树形结构展示 MLIR 类型约束的层级关系。
 * 
 * 设计：
 * - AnyType 作为根节点（内置约束参与层级计算）
 * - 方言约束作为独立分组，挂在 AnyType 下（不参与层级计算）
 * - 中间层只显示约束（不显示具体类型）
 * - 展开到底（没有子约束时）才显示具体类型
 */

import { useState, useMemo, useCallback } from 'react';
import { useTypeConstraintStore, type ConstraintDef } from '../stores/typeConstraintStore';
import { getTypeColor } from '../stores/typeColorCache';

/**
 * 约束层级节点
 */
interface HierarchyNode {
  name: string;
  def: ConstraintDef | null;
  children: HierarchyNode[];
  types: string[];  // 只有叶子约束才有具体类型
  isDialectGroup?: boolean;  // 是否是方言分组节点
}

/**
 * 构建内置约束层级结构
 * 
 * 算法：
 * 1. 展开每个约束为具体类型集合
 * 2. 计算子集关系，构建 DAG
 * 3. 去除传递边，得到 Hasse 图
 * 4. 以 AnyType 为根
 */
function buildBuiltinHierarchy(
  constraintDefs: Map<string, ConstraintDef>,
  getConstraintElements: (name: string) => string[],
  builtinConstraints: string[]
): HierarchyNode {
  // 1. 展开每个内置约束为类型集合
  const constraintSets = new Map<string, Set<string>>();
  const allConstraints: string[] = [];
  
  for (const name of builtinConstraints) {
    const def = constraintDefs.get(name);
    if (!def) continue;
    
    // 跳过 shaped 类型约束（它们不是标量类型集合）
    if (def.rule?.kind === 'shaped' || def.rule?.kind === 'like') {
      continue;
    }
    
    const elements = getConstraintElements(name);
    if (elements.length > 0) {
      constraintSets.set(name, new Set(elements));
      allConstraints.push(name);
    }
  }
  
  // 2. 计算直接子约束关系（只在内置约束之间）
  const directChildren = new Map<string, string[]>();
  
  for (const parent of allConstraints) {
    const parentSet = constraintSets.get(parent)!;
    const children: string[] = [];
    
    for (const child of allConstraints) {
      if (child === parent) continue;
      
      const childSet = constraintSets.get(child)!;
      
      // 检查 child ⊂ parent（真子集）
      if (childSet.size < parentSet.size && 
          [...childSet].every(t => parentSet.has(t))) {
        // 检查是否是直接子约束（没有中间约束）
        let isDirect = true;
        for (const mid of allConstraints) {
          if (mid === parent || mid === child) continue;
          const midSet = constraintSets.get(mid)!;
          
          // 检查 child ⊂ mid ⊂ parent
          if (childSet.size < midSet.size && midSet.size < parentSet.size &&
              [...childSet].every(t => midSet.has(t)) &&
              [...midSet].every(t => parentSet.has(t))) {
            isDirect = false;
            break;
          }
        }
        
        if (isDirect) {
          children.push(child);
        }
      }
    }
    
    directChildren.set(parent, children);
  }
  
  // 3. 构建树形结构
  const buildNode = (name: string, visited: Set<string>): HierarchyNode => {
    if (visited.has(name)) {
      return { name, def: null, children: [], types: [] };
    }
    visited.add(name);
    
    const def = constraintDefs.get(name) ?? null;
    const children = directChildren.get(name) ?? [];
    const typeSet = constraintSets.get(name) ?? new Set<string>();
    
    // 递归构建子节点
    const childNodes = children
      .map(c => buildNode(c, new Set(visited)))
      .sort((a, b) => a.name.localeCompare(b.name));
    
    // 只有没有子约束时才显示具体类型
    // 如果约束名和唯一类型相同（自指），则不显示类型列表
    const typeList = [...typeSet].sort();
    const isSelfReferencing = typeList.length === 1 && typeList[0] === name;
    const types = childNodes.length === 0 && !isSelfReferencing
      ? typeList
      : [];
    
    return { name, def, children: childNodes, types };
  };
  
  // 4. 找到根节点（AnyType 或最大的约束）
  let rootName = 'AnyType';
  if (!constraintSets.has(rootName)) {
    // 找最大的约束
    let maxSize = 0;
    for (const [name, set] of constraintSets) {
      if (set.size > maxSize) {
        maxSize = set.size;
        rootName = name;
      }
    }
  }
  
  return buildNode(rootName, new Set());
}

/**
 * 构建方言约束分组节点
 */
function buildDialectGroup(
  dialectName: string,
  constraintNames: string[],
  constraintDefs: Map<string, ConstraintDef>,
  getConstraintElements: (name: string) => string[]
): HierarchyNode {
  // 方言约束平铺显示（不计算层级）
  const children: HierarchyNode[] = constraintNames
    .map(name => {
      const def = constraintDefs.get(name) ?? null;
      const elements = getConstraintElements(name);
      
      // 如果约束名和唯一类型相同（自指），则不显示类型列表
      const isSelfReferencing = elements.length === 1 && elements[0] === name;
      const types = !isSelfReferencing ? elements.sort() : [];
      
      return {
        name,
        def,
        children: [],
        types,
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
  
  return {
    name: `[${dialectName}]`,
    def: null,
    children,
    types: [],
    isDialectGroup: true,
  };
}

export function TypeConstraintPanel() {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['AnyType']));

  // 从 store 获取约束数据
  const constraintDefs = useTypeConstraintStore(state => state.constraintDefs);
  const constraintsByDialect = useTypeConstraintStore(state => state.constraintsByDialect);
  const getConstraintElements = useTypeConstraintStore(state => state.getConstraintElements);
  const isLoading = useTypeConstraintStore(state => state.isLoading);
  const error = useTypeConstraintStore(state => state.error);

  // 获取内置约束列表
  const builtinConstraints = useMemo(() => {
    return constraintsByDialect.get('__builtin__') || [];
  }, [constraintsByDialect]);

  // 获取已加载的方言列表（排除 __builtin__）
  const loadedDialects = useMemo(() => {
    const dialects: string[] = [];
    for (const key of constraintsByDialect.keys()) {
      if (key !== '__builtin__') {
        dialects.push(key);
      }
    }
    return dialects.sort();
  }, [constraintsByDialect]);

  // 构建内置约束层级结构
  const builtinHierarchy = useMemo(() => {
    if (constraintDefs.size === 0 || builtinConstraints.length === 0) return null;
    return buildBuiltinHierarchy(constraintDefs, getConstraintElements, builtinConstraints);
  }, [constraintDefs, getConstraintElements, builtinConstraints]);

  // 构建方言约束分组（增量更新）
  const dialectGroups = useMemo(() => {
    const groups: HierarchyNode[] = [];
    for (const dialect of loadedDialects) {
      const constraintNames = constraintsByDialect.get(dialect) || [];
      if (constraintNames.length > 0) {
        groups.push(buildDialectGroup(dialect, constraintNames, constraintDefs, getConstraintElements));
      }
    }
    return groups;
  }, [loadedDialects, constraintsByDialect, constraintDefs, getConstraintElements]);

  // 合并根节点：内置层级 + 方言分组
  const rootNode = useMemo(() => {
    if (!builtinHierarchy) return null;
    
    // 将方言分组添加到 AnyType 的子节点中
    return {
      ...builtinHierarchy,
      children: [...builtinHierarchy.children, ...dialectGroups],
    };
  }, [builtinHierarchy, dialectGroups]);

  // 切换节点展开状态
  const handleToggle = useCallback((path: string) => {
    setExpandedNodes(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  // 搜索输入处理
  const handleSearchChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  }, []);

  // 清空搜索
  const handleClearSearch = useCallback(() => {
    setSearchQuery('');
  }, []);

  if (isLoading) {
    return (
      <div className="p-4 text-gray-400 text-sm">
        加载类型约束中...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 text-red-400 text-sm">
        {error}
      </div>
    );
  }

  if (!rootNode) {
    return (
      <div className="p-4 text-gray-400 text-sm">
        没有可用的约束
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* 搜索框 */}
      <div className="p-3 border-b border-gray-700 flex-shrink-0">
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={handleSearchChange}
            placeholder="搜索类型约束..."
            className="w-full px-3 py-2 pr-8 bg-gray-700 border border-gray-600 rounded text-sm text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
          />
          {searchQuery && (
            <button
              onClick={handleClearSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* 约束树 */}
      <div className="flex-1 overflow-auto p-3 min-h-0">
        <HierarchyNodeView
          node={rootNode}
          path={rootNode.name}
          depth={0}
          expandedNodes={expandedNodes}
          onToggle={handleToggle}
          searchQuery={searchQuery}
        />
      </div>
    </div>
  );
}

/**
 * 层级节点视图
 */
interface HierarchyNodeViewProps {
  node: HierarchyNode;
  path: string;
  depth: number;
  expandedNodes: Set<string>;
  onToggle: (path: string) => void;
  searchQuery: string;
}

function HierarchyNodeView({
  node,
  path,
  depth,
  expandedNodes,
  onToggle,
  searchQuery,
}: HierarchyNodeViewProps) {
  const isExpanded = expandedNodes.has(path);
  const hasChildren = node.children.length > 0 || node.types.length > 0;
  const isHighlighted = searchQuery && node.name.toLowerCase().includes(searchQuery.toLowerCase());
  const isDialectGroup = node.isDialectGroup === true;
  const color = isDialectGroup ? '#9ca3af' : getTypeColor(node.name);  // 方言分组用灰色

  // 点击处理（必须在条件返回之前）
  const handleClick = useCallback(() => {
    if (hasChildren) {
      onToggle(path);
    }
  }, [hasChildren, onToggle, path]);

  // 如果有搜索词且不匹配，检查子节点是否匹配
  const childMatches = useMemo(() => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    
    const checkMatch = (n: HierarchyNode): boolean => {
      // 方言分组名称也参与匹配（去掉方括号）
      const displayName = n.isDialectGroup ? n.name.slice(1, -1) : n.name;
      if (displayName.toLowerCase().includes(query)) return true;
      if (n.types.some(t => t.toLowerCase().includes(query))) return true;
      return n.children.some(checkMatch);
    };
    
    return checkMatch(node);
  }, [node, searchQuery]);

  // 如果搜索不匹配，隐藏节点
  if (searchQuery && !childMatches) {
    return null;
  }

  return (
    <div className="select-none">
      {/* 节点行 */}
      <div
        className={`flex items-center py-1 px-2 rounded cursor-pointer transition-colors ${
          isHighlighted ? 'bg-yellow-900/30' : 'hover:bg-gray-700/50'
        }`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={handleClick}
        title={node.def?.summary || undefined}
      >
        {/* 展开/折叠箭头 */}
        {hasChildren ? (
          <span className="w-4 h-4 flex items-center justify-center text-gray-400 text-xs mr-1">
            {isExpanded ? '▼' : '▶'}
          </span>
        ) : (
          <span className="w-4 h-4 mr-1" />
        )}

        {/* 节点名称 */}
        <span className="text-sm font-medium" style={{ color }}>
          {node.name}
        </span>

        {/* 子节点数量（只显示数字） */}
        {hasChildren && !isExpanded && (
          <span className="ml-2 text-xs text-gray-500">
            ({node.children.length > 0 ? node.children.length : node.types.length})
          </span>
        )}
      </div>

      {/* 子节点 */}
      {isExpanded && (
        <div className="border-l border-gray-700 ml-4">
          {/* 子约束 */}
          {node.children.map((child) => (
            <HierarchyNodeView
              key={`${path}/${child.name}`}
              node={child}
              path={`${path}/${child.name}`}
              depth={depth + 1}
              expandedNodes={expandedNodes}
              onToggle={onToggle}
              searchQuery={searchQuery}
            />
          ))}
          
          {/* 具体类型（只有叶子约束才显示） */}
          {node.types.map(type => (
            <TypeLeafView
              key={`${path}/${type}`}
              name={type}
              depth={depth + 1}
              isHighlighted={searchQuery ? type.toLowerCase().includes(searchQuery.toLowerCase()) : false}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * 具体类型叶子节点
 */
interface TypeLeafViewProps {
  name: string;
  depth: number;
  isHighlighted: boolean;
}

function TypeLeafView({ name, depth, isHighlighted }: TypeLeafViewProps) {
  const color = getTypeColor(name);

  return (
    <div
      className={`flex items-center py-1 px-2 rounded ${
        isHighlighted ? 'bg-yellow-900/30' : ''
      }`}
      style={{ paddingLeft: `${depth * 16 + 8}px` }}
    >
      <span className="w-4 h-4 mr-1" />
      <span className="text-sm" style={{ color }}>
        {name}
      </span>
    </div>
  );
}

export default TypeConstraintPanel;
