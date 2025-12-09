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
    <div className="bg-gray-700/50 rounded p-2 mb-2">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-gray-300">SameType</span>
        <button
          onClick={onRemove}
          className="text-gray-500 hover:text-red-400 p-0.5"
          title="删除"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div className="flex flex-wrap gap-1">
        {allPorts.map(port => (
          <button
            key={port.id}
            onClick={() => togglePort(port.id)}
            className={`text-xs px-1.5 py-0.5 rounded border transition-colors ${
              trait.ports.includes(port.id)
                ? 'bg-blue-600/30 border-blue-500 text-blue-300'
                : 'bg-gray-600/30 border-gray-600 text-gray-400 hover:border-gray-500'
            }`}
            title={`${port.group}: ${port.label}`}
          >
            {port.group === '返回值' ? `→${port.label}` : port.label}
          </button>
        ))}
      </div>
      {trait.ports.length < 2 && (
        <p className="text-xs text-yellow-500 mt-1">至少选择 2 个端口</p>
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
    <div className="mt-2 border-t border-gray-600 pt-2">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300"
      >
        <svg
          className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        Traits ({traits.length})
      </button>
      
      {isExpanded && (
        <div className="mt-2">
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
            className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
