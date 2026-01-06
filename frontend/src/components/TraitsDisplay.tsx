/**
 * TraitsDisplay Component
 * 
 * 只读显示节点或函数的 Traits。
 * 用于选中面板中显示操作节点的 traits 和函数的推断 traits。
 * 
 * 设计原则：
 * - 只读显示，不提供编辑功能
 * - 高亮类型传播相关的 traits（粗体）
 * - 显示 trait 名称（MLIR 标准名称）
 */

import { memo, useMemo } from 'react';
import type { FunctionTrait } from '../types';

/**
 * 类型传播相关的 trait 名称
 */
const TYPE_PROPAGATION_TRAITS = new Set([
  'SameOperandsAndResultType',
  'SameTypeOperands',
  'SameOperandsElementType',
  'SameOperandsAndResultElementType',
  'AllTypesMatch',
]);

interface TraitsDisplayProps {
  /** 操作的 traits（字符串数组） */
  operationTraits?: string[];
  /** 函数的 traits（FunctionTrait 数组） */
  functionTraits?: FunctionTrait[];
}

/**
 * 从 trait 字符串中提取 trait 名称
 * 
 * 处理格式：
 * - 简单名称：SameOperandsAndResultType
 * - 参数化：SameOperandsAndResultType<...>
 * - 命名空间：OpTrait::SameOperandsAndResultType
 */
function extractTraitName(trait: string): string {
  // 移除命名空间前缀
  const withoutNamespace = trait.includes('::') 
    ? trait.split('::').pop() || trait 
    : trait;
  
  // 移除参数
  const withoutParams = withoutNamespace.includes('<')
    ? withoutNamespace.split('<')[0]
    : withoutNamespace;
  
  return withoutParams.trim();
}

/**
 * 检查是否为匿名 trait（TableGen 生成的内部名称）
 */
function isAnonymousTrait(name: string): boolean {
  return /^anonymous_\d+$/.test(name);
}

/**
 * 单个 Trait 显示
 */
const TraitItem = memo(function TraitItem({
  name,
  isTypePropagation,
}: {
  name: string;
  isTypePropagation: boolean;
}) {
  return (
    <div className="rf-trait-item">
      <div className={`rf-trait-name ${isTypePropagation ? 'rf-trait-type-propagation' : ''}`}>
        {name}
      </div>
    </div>
  );
});

/**
 * Traits 显示组件
 */
export const TraitsDisplay = memo(function TraitsDisplay({
  operationTraits,
  functionTraits,
}: TraitsDisplayProps) {
  // 处理操作 traits（过滤匿名 traits）
  const processedOperationTraits = useMemo(() => {
    if (!operationTraits || operationTraits.length === 0) return [];
    
    return operationTraits
      .map(trait => {
        const name = extractTraitName(trait);
        return {
          name,
          isTypePropagation: TYPE_PROPAGATION_TRAITS.has(name),
        };
      })
      .filter(trait => !isAnonymousTrait(trait.name));
  }, [operationTraits]);

  // 处理函数 traits
  const processedFunctionTraits = useMemo(() => {
    if (!functionTraits || functionTraits.length === 0) return [];
    
    return functionTraits.map(trait => ({
      name: trait.kind,
      isTypePropagation: TYPE_PROPAGATION_TRAITS.has(trait.kind),
    }));
  }, [functionTraits]);

  const allTraits = [...processedOperationTraits, ...processedFunctionTraits];

  if (allTraits.length === 0) {
    return null;
  }

  return (
    <div className="rf-traits-display">
      <div className="rf-traits-display-list">
        {allTraits.map((trait, index) => (
          <TraitItem
            key={index}
            name={trait.name}
            isTypePropagation={trait.isTypePropagation}
          />
        ))}
      </div>
    </div>
  );
});

export default TraitsDisplay;
