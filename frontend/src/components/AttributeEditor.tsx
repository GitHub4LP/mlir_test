/**
 * AttributeEditor 组件
 * 
 * MLIR 操作属性编辑控件：支持整数、浮点数、字符串、布尔值、枚举类型
 */

import { memo, useCallback, useState } from 'react';
import type { ArgumentDef } from '../types';

/**
 * Props for the AttributeEditor component
 */
export interface AttributeEditorProps {
  /** The attribute definition */
  attribute: ArgumentDef;
  /** Current value of the attribute */
  value: unknown;
  /** Callback when value changes */
  onChange: (name: string, value: unknown) => void;
  /** Whether the editor is disabled */
  disabled?: boolean;
}

/**
 * Determines the input type based on the attribute's type constraint
 */
function getInputType(typeConstraint: string): 'integer' | 'float' | 'boolean' | 'string' | 'enum' | 'array' | 'typed-attr' {
  const constraint = typeConstraint.toLowerCase();

  // TypedAttrInterface - can be integer or float, needs special handling
  // This is used by arith.constant and similar operations
  if (constraint.includes('typedattrinterface') || constraint === 'typedattr') {
    return 'typed-attr';
  }

  // Boolean types
  if (constraint.includes('bool') || constraint === 'i1' || constraint === 'unitattr') {
    return 'boolean';
  }

  // Integer types
  if (constraint.includes('int') || constraint.match(/^[su]?i\d+/) ||
    constraint.includes('index') || constraint.includes('apint')) {
    return 'integer';
  }

  // Float types
  if (constraint.includes('float') || constraint.match(/^[bt]?f\d+/) ||
    constraint.includes('apfloat')) {
    return 'float';
  }

  // Enum types (typically have "Enum" in the name or specific patterns)
  if (constraint.includes('enum') || constraint.includes('case')) {
    return 'enum';
  }

  // Array types
  if (constraint.includes('array') || constraint.includes('dense')) {
    return 'array';
  }

  // Default to string
  return 'string';
}



/**
 * Validates an integer value
 */
function validateInteger(value: string): { valid: boolean; error?: string } {
  if (value === '' || value === '-') return { valid: true };
  const num = parseInt(value, 10);
  if (isNaN(num)) {
    return { valid: false, error: 'Invalid integer' };
  }
  return { valid: true };
}

/**
 * Validates a float value
 */
function validateFloat(value: string): { valid: boolean; error?: string } {
  if (value === '' || value === '-' || value === '.') return { valid: true };
  const num = parseFloat(value);
  if (isNaN(num)) {
    return { valid: false, error: 'Invalid number' };
  }
  return { valid: true };
}

/**
 * Integer input component
 */
const IntegerInput = memo(function IntegerInput({
  value,
  onChange,
  disabled,
  name,
}: {
  value: unknown;
  onChange: (value: number) => void;
  disabled?: boolean;
  name: string;
}) {
  const [error, setError] = useState<string>();
  // 直接使用 prop 作为显示值
  const displayValue = value !== undefined && value !== null ? String(value) : '';

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;

    const validation = validateInteger(newValue);
    setError(validation.error);

    if (validation.valid && newValue !== '' && newValue !== '-') {
      onChange(parseInt(newValue, 10));
    } else if (newValue === '') {
      onChange(0);
    }
  }, [onChange]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <input
        type="number"
        className={error ? 'rf-input rf-input-error rf-input-number' : 'rf-input rf-input-number'}
        value={displayValue}
        onChange={handleChange}
        onClick={(e) => e.stopPropagation()}
        disabled={disabled}
        aria-label={name}
      />
      {error && <span className="rf-error-text">{error}</span>}
    </div>
  );
});

/**
 * Float input component
 */
const FloatInput = memo(function FloatInput({
  value,
  onChange,
  disabled,
  name,
}: {
  value: unknown;
  onChange: (value: number) => void;
  disabled?: boolean;
  name: string;
}) {
  const [error, setError] = useState<string>();
  // 直接使用 prop 作为显示值
  const displayValue = value !== undefined && value !== null ? String(value) : '';

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;

    const validation = validateFloat(newValue);
    setError(validation.error);

    if (validation.valid && newValue !== '' && newValue !== '-' && newValue !== '.') {
      onChange(parseFloat(newValue));
    } else if (newValue === '') {
      onChange(0);
    }
  }, [onChange]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <input
        type="number"
        step="any"
        className={error ? 'rf-input rf-input-error rf-input-number' : 'rf-input rf-input-number'}
        value={displayValue}
        onChange={handleChange}
        onClick={(e) => e.stopPropagation()}
        disabled={disabled}
        aria-label={name}
      />
      {error && <span className="rf-error-text">{error}</span>}
    </div>
  );
});

/**
 * Boolean input component (checkbox)
 */
const BooleanInput = memo(function BooleanInput({
  value,
  onChange,
  disabled,
  name,
}: {
  value: unknown;
  onChange: (value: boolean) => void;
  disabled?: boolean;
  name: string;
}) {
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.checked);
  }, [onChange]);

  return (
    <input
      type="checkbox"
      className="rf-checkbox"
      checked={Boolean(value)}
      onChange={handleChange}
      onClick={(e) => e.stopPropagation()}
      disabled={disabled}
      aria-label={name}
    />
  );
});

/**
 * String input component
 */
const StringInput = memo(function StringInput({
  value,
  onChange,
  disabled,
  name,
}: {
  value: unknown;
  onChange: (value: string) => void;
  disabled?: boolean;
  name: string;
}) {
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
  }, [onChange]);

  return (
    <input
      type="text"
      className="rf-input rf-input-text"
      value={String(value ?? '')}
      onChange={handleChange}
      onClick={(e) => e.stopPropagation()}
      disabled={disabled}
      aria-label={name}
    />
  );
});

/**
 * Enum option type (matches EnumOption from types/index.ts)
 */
interface EnumOptionValue {
  str: string;
  symbol: string;
  value: number;
  summary: string;
}

/**
 * Enum/dropdown input component
 * 显示 str（MLIR IR 格式），保存完整 EnumOption 对象
 * option 的 title 属性显示 summary 作为 tooltip
 */
const EnumInput = memo(function EnumInput({
  value,
  onChange,
  options,
  disabled,
  name,
}: {
  value: unknown;
  onChange: (value: EnumOptionValue) => void;
  options: EnumOptionValue[];
  disabled?: boolean;
  name: string;
}) {
  // 从 value 中提取 symbol 用于显示当前选中项
  // 兼容对象格式和字符串格式
  let currentSymbol = '';
  if (typeof value === 'object' && value !== null) {
    currentSymbol = (value as EnumOptionValue).symbol ?? '';
  } else if (typeof value === 'string') {
    // 尝试匹配 symbol 或 str
    const match = options.find(opt => opt.symbol === value || opt.str === value);
    currentSymbol = match?.symbol ?? '';
  }

  const handleChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedSymbol = e.target.value;
    const selectedOption = options.find(opt => opt.symbol === selectedSymbol);
    if (selectedOption) {
      onChange(selectedOption);  // 保存完整对象
    }
  }, [onChange, options]);

  return (
    <select
      className="rf-select"
      value={currentSymbol}
      onChange={handleChange}
      onClick={(e) => e.stopPropagation()}
      disabled={disabled}
      aria-label={name}
    >
      <option value="">Select...</option>
      {options.map(opt => (
        <option key={opt.symbol} value={opt.symbol} title={opt.summary}>
          {opt.str}
        </option>
      ))}
    </select>
  );
});

/**
 * Array input component (comma-separated values)
 */
const ArrayInput = memo(function ArrayInput({
  value,
  onChange,
  disabled,
  name,
}: {
  value: unknown;
  onChange: (value: unknown[]) => void;
  disabled?: boolean;
  name: string;
}) {
  const arrayValue = Array.isArray(value) ? value : [];
  const displayValue = arrayValue.join(', ');

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;

    // Parse comma-separated values
    const parsed = newValue
      .split(',')
      .map(s => s.trim())
      .filter(s => s !== '')
      .map(s => {
        const num = parseFloat(s);
        return isNaN(num) ? s : num;
      });

    onChange(parsed);
  }, [onChange]);

  return (
    <input
      type="text"
      className="rf-input"
      style={{ width: 128 }}
      value={displayValue}
      onChange={handleChange}
      onClick={(e) => e.stopPropagation()}
      placeholder="1, 2, 3"
      disabled={disabled}
      aria-label={name}
    />
  );
});

/**
 * TypedAttr input component - for TypedAttrInterface (integer or float values)
 * Used by arith.constant and similar operations
 * 
 * 存储为字符串以保留用户输入格式（如 "2.0" vs "2"）
 * 后端根据 outputTypes 决定最终格式
 * 
 * 需要内部状态来处理中间输入状态（如 "-"、"."）
 */
const TypedAttrInput = memo(function TypedAttrInput({
  value,
  onChange,
  disabled,
  name,
}: {
  value: unknown;
  onChange: (value: string) => void;
  disabled?: boolean;
  name: string;
}) {
  const [error, setError] = useState<string>();
  // 直接使用 prop 作为显示值
  const displayValue = value !== undefined && value !== null ? String(value) : '';

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;

    // 允许输入中间状态
    if (newValue === '' || newValue === '-' || newValue === '.' || newValue === '-.') {
      setError(undefined);
      onChange(newValue || '0');
      return;
    }

    // 验证是否为有效数字
    const num = parseFloat(newValue);
    if (isNaN(num)) {
      setError('Invalid number');
    } else {
      setError(undefined);
      // 保存原始字符串，保留 "2.0" 格式
      onChange(newValue);
    }
  }, [onChange]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <input
        type="text"
        inputMode="decimal"
        className={error ? 'rf-input rf-input-error rf-input-text' : 'rf-input rf-input-text'}
        value={displayValue}
        onChange={handleChange}
        onClick={(e) => e.stopPropagation()}
        disabled={disabled}
        placeholder="0"
        aria-label={name}
      />
      {error && <span className="rf-error-text">{error}</span>}
    </div>
  );
});

/**
 * Main AttributeEditor component
 * 
 * Renders the appropriate input control based on the attribute's type constraint.
 */
export const AttributeEditor = memo(function AttributeEditor({
  attribute,
  value,
  onChange,
  disabled = false,
}: AttributeEditorProps) {
  // 使用后端提取的 enumOptions，不再硬编码
  const enumOptions = attribute.enumOptions ?? [];
  // 如果有 enumOptions，优先使用 enum 类型
  const inputType = enumOptions.length > 0 ? 'enum' : getInputType(attribute.typeConstraint);

  const handleChange = useCallback((newValue: unknown) => {
    onChange(attribute.name, newValue);
  }, [attribute.name, onChange]);

  // Render optional indicator
  const optionalIndicator = attribute.isOptional ? (
    <span className="rf-optional-indicator">?</span>
  ) : null;

  // 构建 tooltip：显示 displayName 和 description
  const tooltip = [
    attribute.displayName !== attribute.name ? attribute.displayName : '',
    attribute.description,
  ].filter(Boolean).join('\n') || undefined;

  return (
    <div className="rf-node-attr-row">
      <span
        className="rf-node-attr-label"
        title={tooltip}
      >
        {attribute.name}
        {optionalIndicator}
      </span>

      {inputType === 'integer' && (
        <IntegerInput
          value={value}
          onChange={handleChange as (v: number) => void}
          disabled={disabled}
          name={attribute.name}
        />
      )}

      {inputType === 'float' && (
        <FloatInput
          value={value}
          onChange={handleChange as (v: number) => void}
          disabled={disabled}
          name={attribute.name}
        />
      )}

      {inputType === 'boolean' && (
        <BooleanInput
          value={value}
          onChange={handleChange as (v: boolean) => void}
          disabled={disabled}
          name={attribute.name}
        />
      )}

      {inputType === 'string' && (
        <StringInput
          value={value}
          onChange={handleChange as (v: string) => void}
          disabled={disabled}
          name={attribute.name}
        />
      )}

      {inputType === 'enum' && enumOptions.length > 0 && (
        <EnumInput
          value={value}
          onChange={handleChange as (v: EnumOptionValue) => void}
          options={enumOptions}
          disabled={disabled}
          name={attribute.name}
        />
      )}

      {inputType === 'enum' && enumOptions.length === 0 && (
        <StringInput
          value={value}
          onChange={handleChange as (v: string) => void}
          disabled={disabled}
          name={attribute.name}
        />
      )}

      {inputType === 'array' && (
        <ArrayInput
          value={value}
          onChange={handleChange as (v: unknown[]) => void}
          disabled={disabled}
          name={attribute.name}
        />
      )}

      {inputType === 'typed-attr' && (
        <TypedAttrInput
          value={value}
          onChange={handleChange}
          disabled={disabled}
          name={attribute.name}
        />
      )}
    </div>
  );
});

export default AttributeEditor;
