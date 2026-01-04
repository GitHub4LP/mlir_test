/**
 * ConstraintNode 组件
 * 
 * 递归渲染单个约束/类型节点。
 * 根据 rule.kind 展示不同的子节点结构。
 */

import { useMemo, useCallback } from 'react';
import type { ConstraintRule, ConstraintDef } from '../services/constraintResolver';
import { getTypeColor } from '../stores/typeColorCache';

export interface ConstraintNodeProps {
  name: string;
  rule: ConstraintRule | null;
  summary?: string;
  path: string;  // 唯一路径，用于展开状态
  depth: number;
  constraintMap: Map<string, ConstraintDef>;
  expandedNodes: Set<string>;
  onToggle: (path: string) => void;
  isHighlighted?: boolean;
}

/**
 * 获取 rule 的子节点信息
 */
function getRuleChildren(rule: ConstraintRule | null): { label?: string; children: Array<{ name: string; rule: ConstraintRule | null }> } {
  if (!rule) {
    return { children: [] };
  }

  switch (rule.kind) {
    case 'type':
      // 具体类型，无子节点
      return { children: [] };

    case 'oneOf':
      // 类型枚举，每个类型作为子节点
      return {
        children: rule.types.map(t => ({ name: t, rule: { kind: 'type' as const, name: t } }))
      };

    case 'or':
      // 并集
      return {
        label: '∪',
        children: rule.children.map((c, i) => ({
          name: getRuleName(c, i),
          rule: c
        }))
      };

    case 'and':
      // 交集
      return {
        label: '∩',
        children: rule.children.map((c, i) => ({
          name: getRuleName(c, i),
          rule: c
        }))
      };

    case 'ref':
      // 引用，显示引用名称
      return {
        children: [{ name: rule.name, rule: null }]  // rule 为 null，需要从 constraintMap 查找
      };

    case 'like':
      // Like 类型
      return {
        label: 'Like',
        children: [{ name: getRuleName(rule.element, 0), rule: rule.element }]
      };

    case 'shaped':
      // 容器类型
      return {
        label: `${rule.container}${rule.ranked !== undefined ? (rule.ranked ? ' (ranked)' : ' (unranked)') : ''}`,
        children: rule.element ? [{ name: getRuleName(rule.element, 0), rule: rule.element }] : []
      };

    default:
      return { children: [] };
  }
}

/**
 * 从 rule 获取显示名称
 */
function getRuleName(rule: ConstraintRule, index: number): string {
  switch (rule.kind) {
    case 'type':
      return rule.name;
    case 'ref':
      return rule.name;
    case 'oneOf':
      return `[${rule.types.slice(0, 3).join(', ')}${rule.types.length > 3 ? '...' : ''}]`;
    case 'or':
      return `or_${index}`;
    case 'and':
      return `and_${index}`;
    case 'like':
      return `like_${index}`;
    case 'shaped':
      return rule.container;
    default:
      return `item_${index}`;
  }
}

/**
 * 判断是否是叶子节点（具体类型）
 */
function isLeafNode(rule: ConstraintRule | null): boolean {
  if (!rule) return true;
  return rule.kind === 'type';
}

/**
 * 获取节点显示标签
 */
function getNodeLabel(name: string, rule: ConstraintRule | null): string {
  if (!rule) return name;
  
  switch (rule.kind) {
    case 'type':
      return rule.name;
    case 'shaped':
      return `${name} (${rule.container})`;
    case 'like':
      // like 约束显示约束名
      return name || '任意类型';
    default:
      return name;
  }
}

export function ConstraintNode({
  name,
  rule,
  summary,
  path,
  depth,
  constraintMap,
  expandedNodes,
  onToggle,
  isHighlighted = false,
}: ConstraintNodeProps) {
  const isExpanded = expandedNodes.has(path);
  const isLeaf = isLeafNode(rule);

  // 如果是 ref 类型，从 constraintMap 获取实际的 rule
  const actualRule = useMemo(() => {
    if (rule?.kind === 'ref') {
      const refDef = constraintMap.get(rule.name);
      return refDef?.rule ?? null;
    }
    return rule;
  }, [rule, constraintMap]);

  // 获取子节点信息
  const { label: kindLabel, children } = useMemo(() => {
    return getRuleChildren(actualRule);
  }, [actualRule]);

  // 获取类型颜色（仅对具体类型）
  const typeColor = useMemo(() => {
    if (rule?.kind === 'type') {
      return getTypeColor(rule.name);
    }
    return null;
  }, [rule]);

  // 切换展开状态
  const handleToggle = useCallback(() => {
    onToggle(path);
  }, [onToggle, path]);

  // 节点标签
  const nodeLabel = getNodeLabel(name, rule);

  return (
    <div className="select-none">
      {/* 节点行 */}
      <div
        className={`flex items-center py-1 px-2 rounded cursor-pointer transition-colors ${
          isHighlighted ? 'bg-yellow-900/30' : 'hover:bg-gray-700/50'
        }`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={!isLeaf && children.length > 0 ? handleToggle : undefined}
        title={summary || undefined}
      >
        {/* 展开/折叠箭头 */}
        {!isLeaf && children.length > 0 ? (
          <span className="w-4 h-4 flex items-center justify-center text-gray-400 text-xs mr-1">
            {isExpanded ? '▼' : '▶'}
          </span>
        ) : (
          <span className="w-4 h-4 mr-1" />
        )}

        {/* 类型颜色指示器 */}
        {typeColor && (
          <span
            className="w-3 h-3 rounded-full mr-2 flex-shrink-0"
            style={{ backgroundColor: typeColor }}
          />
        )}

        {/* 节点名称 */}
        <span className={`text-sm ${isLeaf ? 'text-gray-300' : 'text-white font-medium'}`}>
          {nodeLabel}
        </span>

        {/* kind 标签 */}
        {kindLabel && (
          <span className="ml-2 text-xs text-gray-500">
            {kindLabel}
          </span>
        )}

        {/* 子节点数量 */}
        {!isLeaf && children.length > 0 && !isExpanded && (
          <span className="ml-2 text-xs text-gray-500">
            ({children.length})
          </span>
        )}
      </div>

      {/* 子节点 */}
      {isExpanded && children.length > 0 && (
        <div className="border-l border-gray-700 ml-4">
          {children.map((child, index) => {
            // 如果子节点是 ref，从 constraintMap 获取完整定义
            let childRule = child.rule;
            let childSummary: string | undefined;
            
            if (child.rule === null || child.rule?.kind === 'ref') {
              const refName = child.rule?.kind === 'ref' ? child.rule.name : child.name;
              const refDef = constraintMap.get(refName);
              if (refDef) {
                childRule = refDef.rule;
                childSummary = refDef.summary;
              }
            }

            return (
              <ConstraintNode
                key={`${path}/${child.name}/${index}`}
                name={child.name}
                rule={childRule}
                summary={childSummary}
                path={`${path}/${child.name}/${index}`}
                depth={depth + 1}
                constraintMap={constraintMap}
                expandedNodes={expandedNodes}
                onToggle={onToggle}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

export default ConstraintNode;
