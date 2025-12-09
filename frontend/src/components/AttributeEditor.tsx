/**
 * AttributeEditor Component
 * 
 * Provides input controls for editing MLIR operation attributes.
 * Supports integer, float, string, boolean, and enum types.
 * 
 * Requirements: 3.4, 9.1, 9.2, 9.3, 9.4, 9.5
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
    <div className="flex flex-col">
      <input
        type="number"
        className={`w-20 text-xs bg-gray-700 text-white rounded px-2 py-1 border ${
          error ? 'border-red-500' : 'border-gray-600'
        } focus:outline-none focus:border-blue-500`}
        value={displayValue}
        onChange={handleChange}
        onClick={(e) => e.stopPropagation()}
        disabled={disabled}
        aria-label={name}
      />
      {error && <span className="text-xs text-red-400 mt-0.5">{error}</span>}
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
    <div className="flex flex-col">
      <input
        type="number"
        step="any"
        className={`w-20 text-xs bg-gray-700 text-white rounded px-2 py-1 border ${
          error ? 'border-red-500' : 'border-gray-600'
        } focus:outline-none focus:border-blue-500`}
        value={displayValue}
        onChange={handleChange}
        onClick={(e) => e.stopPropagation()}
        disabled={disabled}
        aria-label={name}
      />
      {error && <span className="text-xs text-red-400 mt-0.5">{error}</span>}
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
      className="w-4 h-4 bg-gray-700 rounded border border-gray-600 accent-blue-500"
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
      className="w-24 text-xs bg-gray-700 text-white rounded px-2 py-1 border border-gray-600 focus:outline-none focus:border-blue-500"
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
  const currentSymbol = typeof value === 'object' && value !== null 
    ? (value as EnumOptionValue).symbol 
    : '';

  const handleChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedSymbol = e.target.value;
    const selectedOption = options.find(opt => opt.symbol === selectedSymbol);
    if (selectedOption) {
      onChange(selectedOption);  // 保存完整对象
    }
  }, [onChange, options]);

  return (
    <select
      className="w-24 text-xs bg-gray-700 text-white rounded px-2 py-1 border border-gray-600 focus:outline-none focus:border-blue-500"
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
      className="w-32 text-xs bg-gray-700 text-white rounded px-2 py-1 border border-gray-600 focus:outline-none focus:border-blue-500"
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
 */
const TypedAttrInput = memo(function TypedAttrInput({
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
  const displayValue = value !== undefined && value !== null ? String(value) : '';

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    
    // Allow empty, negative sign, or decimal point during typing
    if (newValue === '' || newValue === '-' || newValue === '.') {
      setError(undefined);
      if (newValue === '') {
        onChange(0);
      }
      return;
    }
    
    // Try to parse as number (integer or float)
    const num = parseFloat(newValue);
    if (isNaN(num)) {
      setError('Invalid number');
    } else {
      setError(undefined);
      onChange(num);
    }
  }, [onChange]);

  return (
    <div className="flex flex-col">
      <input
        type="number"
        step="any"
        className={`w-24 text-xs bg-gray-700 text-white rounded px-2 py-1 border ${
          error ? 'border-red-500' : 'border-gray-600'
        } focus:outline-none focus:border-blue-500`}
        value={displayValue}
        onChange={handleChange}
        onClick={(e) => e.stopPropagation()}
        disabled={disabled}
        placeholder="0"
        aria-label={name}
      />
      {error && <span className="text-xs text-red-400 mt-0.5">{error}</span>}
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
    <span className="text-gray-500 text-xs ml-1">?</span>
  ) : null;

  // 构建 tooltip：显示 displayName 和 description
  const tooltip = [
    attribute.displayName !== attribute.name ? attribute.displayName : '',
    attribute.description,
  ].filter(Boolean).join('\n') || undefined;

  return (
    <div className="flex items-center justify-between py-1 gap-2">
      <span 
        className="text-xs text-gray-400 flex items-center cursor-help"
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
          onChange={handleChange as (v: number) => void}
          disabled={disabled}
          name={attribute.name}
        />
      )}
    </div>
  );
});

export default AttributeEditor;
