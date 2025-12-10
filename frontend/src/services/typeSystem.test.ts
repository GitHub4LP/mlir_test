/**
 * Type System Service Unit Tests
 * 
 * Tests for the type system service that manages type constraints
 * and compatibility checking.
 * 
 * 注意：类型约束数据现在从后端动态加载。
 * 测试前需要先初始化 typeConstraintStore。
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  isAbstractConstraint,
  getConcreteTypes,
  isCompatible,
  canConnect,
  findCommonType,
  hasSameOperandsAndResultTypeTrait,
  hasAllTypesMatchTrait,
  getTypeColor,
  normalizeType,
} from './typeSystem';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import type { ConstraintDef } from '../stores/typeConstraintStore';
import type { OperationDef } from '../types';

// ============================================================================
// 测试辅助：构建 mock constraintDefs
// ============================================================================

function buildMockConstraintDefs(): Map<string, ConstraintDef> {
  const buildableTypes = [
    'I1', 'I8', 'I16', 'I32', 'I64', 'I128',
    'SI1', 'SI8', 'SI16', 'SI32', 'SI64',
    'UI1', 'UI8', 'UI16', 'UI32', 'UI64',
    'F16', 'F32', 'F64', 'F80', 'F128', 'BF16', 'TF32',
    'Index', 'NoneType',
  ];
  
  const constraintMap: Record<string, string[]> = {
    'SignlessIntegerLike': ['I1', 'I8', 'I16', 'I32', 'I64', 'I128'],
    'SignlessIntegerOrIndexLike': ['I1', 'I8', 'I16', 'I32', 'I64', 'I128', 'Index'],
    'AnySignlessInteger': ['I1', 'I8', 'I16', 'I32', 'I64', 'I128'],
    'AnyFloat': ['F16', 'F32', 'F64', 'F80', 'F128', 'BF16', 'TF32'],
    'FloatLike': ['F16', 'F32', 'F64', 'F80', 'F128', 'BF16', 'TF32'],
    'SignlessIntegerOrFloatLike': [
      'I1', 'I8', 'I16', 'I32', 'I64', 'I128',
      'F16', 'F32', 'F64', 'F80', 'F128', 'BF16', 'TF32',
    ],
    'BoolLike': ['I1'],
    'AnyType': buildableTypes,
  };
  
  const defs = new Map<string, ConstraintDef>();
  
  // 添加 BuildableType
  for (const t of buildableTypes) {
    defs.set(t, { name: t, summary: '', rule: { kind: 'type', name: t } });
  }
  
  // 添加约束
  for (const [name, types] of Object.entries(constraintMap)) {
    if (!defs.has(name)) {
      defs.set(name, { name, summary: '', rule: { kind: 'oneOf', types } });
    }
  }
  
  return defs;
}

// ============================================================================
// 测试前初始化 store（模拟后端数据）
// ============================================================================

beforeAll(() => {
  const buildableTypes = [
    'I1', 'I8', 'I16', 'I32', 'I64', 'I128',
    'SI1', 'SI8', 'SI16', 'SI32', 'SI64',
    'UI1', 'UI8', 'UI16', 'UI32', 'UI64',
    'F16', 'F32', 'F64', 'F80', 'F128', 'BF16', 'TF32',
    'Index', 'NoneType',
  ];
  
  useTypeConstraintStore.setState({
    buildableTypes,
    constraintDefs: buildMockConstraintDefs(),
    typeDefinitions: [],
    isLoaded: true,
    isLoading: false,
    error: null,
  });
});

// ============================================================================
// Type Normalization Tests
// ============================================================================

describe('normalizeType', () => {
  it('should normalize lowercase integer types', () => {
    expect(normalizeType('i32')).toBe('I32');
    expect(normalizeType('i64')).toBe('I64');
  });

  it('should normalize lowercase signed integer types', () => {
    expect(normalizeType('si32')).toBe('SI32');
  });

  it('should normalize lowercase unsigned integer types', () => {
    expect(normalizeType('ui32')).toBe('UI32');
  });

  it('should normalize lowercase float types', () => {
    expect(normalizeType('f32')).toBe('F32');
    expect(normalizeType('bf16')).toBe('BF16');
    expect(normalizeType('tf32')).toBe('TF32');
  });

  it('should normalize index type', () => {
    expect(normalizeType('index')).toBe('Index');
  });

  it('should return unchanged for already normalized types', () => {
    expect(normalizeType('I32')).toBe('I32');
    expect(normalizeType('SignlessIntegerLike')).toBe('SignlessIntegerLike');
  });
});

// ============================================================================
// Type Constraint Query Tests
// ============================================================================

describe('getConcreteTypes', () => {
  it('should return concrete types for abstract constraints', () => {
    const types = getConcreteTypes('SignlessIntegerLike');
    expect(types).toContain('I32');
    expect(types).toContain('I64');
    expect(types.length).toBeGreaterThan(1);
  });

  it('should return single type for concrete constraints', () => {
    const types = getConcreteTypes('I32');
    expect(types).toEqual(['I32']);
  });

  it('should return the constraint itself for unknown constraints', () => {
    const types = getConcreteTypes('UnknownType');
    expect(types).toEqual(['UnknownType']);
  });

  it('should return a copy to prevent mutation', () => {
    const types1 = getConcreteTypes('SignlessIntegerLike');
    const types2 = getConcreteTypes('SignlessIntegerLike');
    types1.push('Modified');
    expect(types2).not.toContain('Modified');
  });
});

describe('isAbstractConstraint', () => {
  it('should return true for abstract constraints', () => {
    expect(isAbstractConstraint('SignlessIntegerLike')).toBe(true);
    expect(isAbstractConstraint('AnyFloat')).toBe(true);
    expect(isAbstractConstraint('AnyType')).toBe(true);
  });

  it('should return false for concrete types', () => {
    expect(isAbstractConstraint('I32')).toBe(false);
    expect(isAbstractConstraint('F64')).toBe(false);
    expect(isAbstractConstraint('Index')).toBe(false);
  });

  it('should return false for unknown constraints', () => {
    expect(isAbstractConstraint('UnknownType')).toBe(false);
  });
});

// ============================================================================
// Type Compatibility Tests
// ============================================================================

describe('isCompatible', () => {
  it('should return true for identical types', () => {
    expect(isCompatible('I32', 'I32')).toBe(true);
    expect(isCompatible('SignlessIntegerLike', 'SignlessIntegerLike')).toBe(true);
  });

  it('should return true when concrete type satisfies abstract constraint', () => {
    expect(isCompatible('I32', 'SignlessIntegerLike')).toBe(true);
    expect(isCompatible('I64', 'SignlessIntegerOrIndexLike')).toBe(true);
    expect(isCompatible('F32', 'AnyFloat')).toBe(true);
  });

  it('should return false when concrete type does not satisfy constraint', () => {
    expect(isCompatible('F32', 'SignlessIntegerLike')).toBe(false);
    expect(isCompatible('I32', 'AnyFloat')).toBe(false);
  });

  it('should return true when Index satisfies SignlessIntegerOrIndexLike', () => {
    expect(isCompatible('Index', 'SignlessIntegerOrIndexLike')).toBe(true);
  });

  it('should return false when Index does not satisfy SignlessIntegerLike', () => {
    expect(isCompatible('Index', 'SignlessIntegerLike')).toBe(false);
  });

  it('should handle abstract to abstract compatibility (intersection check)', () => {
    expect(isCompatible('SignlessIntegerLike', 'SignlessIntegerOrIndexLike')).toBe(true);
    expect(isCompatible('SignlessIntegerOrIndexLike', 'SignlessIntegerLike')).toBe(true);
    expect(isCompatible('AnyFloat', 'SignlessIntegerLike')).toBe(false);
  });
});

describe('canConnect', () => {
  it('should return true for compatible types in either direction', () => {
    expect(canConnect('I32', 'SignlessIntegerLike')).toBe(true);
    expect(canConnect('SignlessIntegerLike', 'I32')).toBe(true);
  });

  it('should return false for incompatible types', () => {
    expect(canConnect('F32', 'SignlessIntegerLike')).toBe(false);
  });
});

describe('findCommonType', () => {
  it('should find common type between compatible constraints', () => {
    const common = findCommonType('SignlessIntegerLike', 'SignlessIntegerOrIndexLike');
    expect(common).not.toBeNull();
    expect(['I1', 'I8', 'I16', 'I32', 'I64', 'I128']).toContain(common);
  });

  it('should return null for incompatible constraints', () => {
    const common = findCommonType('SignlessIntegerLike', 'AnyFloat');
    expect(common).toBeNull();
  });

  it('should prefer concrete type when one constraint is concrete', () => {
    const common = findCommonType('I32', 'SignlessIntegerLike');
    expect(common).toBe('I32');
  });

  it('should return the concrete type when both are the same', () => {
    const common = findCommonType('I32', 'I32');
    expect(common).toBe('I32');
  });
});

// ============================================================================
// Trait Detection Tests
// ============================================================================

function createMockOperation(traits: string[]): OperationDef {
  return {
    dialect: 'test',
    opName: 'testop',
    fullName: 'test.testop',
    summary: 'Test operation',
    description: '',
    arguments: [
      { name: 'lhs', kind: 'operand', typeConstraint: 'SignlessIntegerLike', displayName: 'SignlessIntegerLike', description: '', isOptional: false, isVariadic: false },
      { name: 'rhs', kind: 'operand', typeConstraint: 'SignlessIntegerLike', displayName: 'SignlessIntegerLike', description: '', isOptional: false, isVariadic: false },
    ],
    results: [
      { name: 'result', typeConstraint: 'SignlessIntegerLike', displayName: 'SignlessIntegerLike', description: '', isVariadic: false },
    ],
    regions: [],
    traits,
    assemblyFormat: '',
    hasRegions: false,
    isTerminator: false,
    isPure: true,
  };
}

describe('hasSameOperandsAndResultTypeTrait', () => {
  it('should detect SameOperandsAndResultType trait', () => {
    const op = createMockOperation(['SameOperandsAndResultType']);
    expect(hasSameOperandsAndResultTypeTrait(op)).toBe(true);
  });

  it('should return false when trait is not present', () => {
    const op = createMockOperation(['Commutative']);
    expect(hasSameOperandsAndResultTypeTrait(op)).toBe(false);
  });
});

describe('hasAllTypesMatchTrait', () => {
  it('should detect AllTypesMatch trait', () => {
    const op = createMockOperation(['AllTypesMatch']);
    expect(hasAllTypesMatchTrait(op)).toBe(true);
  });

  it('should return false when trait is not present', () => {
    const op = createMockOperation(['Commutative']);
    expect(hasAllTypesMatchTrait(op)).toBe(false);
  });
});

// ============================================================================
// Type Color Tests
// ============================================================================

describe('getTypeColor', () => {
  it('should return blue for integer types', () => {
    expect(getTypeColor('I32')).toBe('#4A90D9');
    expect(getTypeColor('SignlessIntegerLike')).toBe('#4A90D9');
  });

  it('should return green for float types', () => {
    expect(getTypeColor('F32')).toBe('#50C878');
    expect(getTypeColor('AnyFloat')).toBe('#50C878');
  });

  it('should return purple for Index type', () => {
    expect(getTypeColor('Index')).toBe('#9B59B6');
  });

  it('should return orange for boolean types', () => {
    expect(getTypeColor('BoolLike')).toBe('#E67E22');
    expect(getTypeColor('I1')).toBe('#E67E22');
  });

  it('should return gray for unknown types', () => {
    expect(getTypeColor('UnknownType')).toBe('#95A5A6');
  });
});
