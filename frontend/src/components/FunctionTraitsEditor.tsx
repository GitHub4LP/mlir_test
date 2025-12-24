/**
 * FunctionTraitsEditor Component
 * 
 * 编辑函数级别的 Traits，定义参数/返回值之间的类型关系。
 * 
 * 支持的 Traits：
 * - SameType: 指定的端口类型必须相同（用于泛型函数）
 */

import { memo, useCallback, useState, useMemo } from 'react';
import type { FunctionTrait, ParameterDef, TypeDef } from '../types';

interface FunctionTraitsEditorProps {
  parameters: ParameterDef[];
  returnTypes: TypeDef[];
  traits: FunctionTrait[];
  onChange: (traits: FunctionTrait[]) => void;
  disabled?: boolean;
}

/**
 * 单个 SameType Trait 的编辑器
 */
const SameTypeTraitEditor = memo(function SameTypeTraitEditor({
  trait,
  parameters,
  returnTypes,
  onUpdate,
  onRemove,
}: {
  trait: FunctionTrait;
  parameters: ParameterDef[];
  returnTypes: TypeDef[];
  onUpdate: (trait: FunctionTrait) => void;
  onRemove: () => void;
}) {
  // 所有可选的端口
  const allPorts = useMemo(() => {
    const ports: { id: string; label: string; group: string }[] = [];
    for (const p of parameters) {
      ports.push({ id: p.name, label: p.name, group: '参数' });
    }
    for (const r of returnTypes) {
      ports.push({ id: `return:${r.name}`, label: r.name, group: '返回值' });
    }
    return ports;
  }, [parameters, returnTypes]);

  const togglePort = useCallback((portId: string) => {
    const newPorts = trait.ports.includes(portId)
      ? trait.ports.filter(p => p !== portId)
      : [...trait.ports, portId];
    onUpdate({ ...trait, ports: newPorts });
  }, [trait, onUpdate]);

  return (
    <div className="rf-trait-editor">
      <div className="rf-trait-header">
        <span className="rf-trait-title">SameType</span>
        <button
          onClick={onRemove}
          className="rf-trait-remove-btn"
          title="删除"
        >
          <svg style={{ width: 12, height: 12 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div className="rf-trait-ports">
        {allPorts.map(port => (
          <button
            key={port.id}
            onClick={() => togglePort(port.id)}
            className={trait.ports.includes(port.id) ? 'rf-trait-port rf-trait-port-selected' : 'rf-trait-port'}
            title={`${port.group}: ${port.label}`}
          >
            {port.group === '返回值' ? `→${port.label}` : port.label}
          </button>
        ))}
      </div>
      {trait.ports.length < 2 && (
        <p className="rf-trait-warning">至少选择 2 个端口</p>
      )}
    </div>
  );
});

/**
 * 函数 Traits 编辑器
 */
export const FunctionTraitsEditor = memo(function FunctionTraitsEditor({
  parameters,
  returnTypes,
  traits,
  onChange,
  disabled = false,
}: FunctionTraitsEditorProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleAddTrait = useCallback(() => {
    // 默认选择所有参数和返回值
    const defaultPorts: string[] = [
      ...parameters.map(p => p.name),
      ...returnTypes.map(r => `return:${r.name}`),
    ];
    onChange([...traits, { kind: 'SameType', ports: defaultPorts }]);
  }, [traits, parameters, returnTypes, onChange]);

  const handleUpdateTrait = useCallback((index: number, trait: FunctionTrait) => {
    const newTraits = [...traits];
    newTraits[index] = trait;
    onChange(newTraits);
  }, [traits, onChange]);

  const handleRemoveTrait = useCallback((index: number) => {
    onChange(traits.filter((_, i) => i !== index));
  }, [traits, onChange]);

  if (disabled) {
    return null;
  }

  // 没有参数和返回值时不显示
  if (parameters.length === 0 && returnTypes.length === 0) {
    return null;
  }

  return (
    <div className="rf-traits-panel">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="rf-traits-header"
      >
        <svg
          style={{ 
            width: 12, 
            height: 12, 
            transition: 'transform 0.2s',
            transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)'
          }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="rf-traits-title">Traits ({traits.length})</span>
      </button>
      
      {isExpanded && (
        <div className="rf-traits-content">
          {traits.map((trait, index) => (
            <SameTypeTraitEditor
              key={index}
              trait={trait}
              parameters={parameters}
              returnTypes={returnTypes}
              onUpdate={(t) => handleUpdateTrait(index, t)}
              onRemove={() => handleRemoveTrait(index)}
            />
          ))}
          
          <button
            onClick={handleAddTrait}
            className="rf-trait-add-btn"
          >
            <svg style={{ width: 12, height: 12 }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            添加 SameType
          </button>
        </div>
      )}
    </div>
  );
});

export default FunctionTraitsEditor;
