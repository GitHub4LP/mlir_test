/**
 * Palette Utils Tests
 * 
 * Tests for the filterOperations and groupByDialect utility functions.
 * 
 * Requirements: 4.2
 */

import { describe, it, expect } from 'vitest';
import { filterOperations, groupByDialect } from './paletteUtils';
import type { OperationDef, DialectInfo } from '../types';

// Helper to create a mock operation
function createMockOperation(
  opName: string,
  dialect: string = 'arith',
  summary: string = ''
): OperationDef {
  return {
    dialect,
    opName,
    fullName: `${dialect}.${opName}`,
    summary,
    description: '',
    arguments: [],
    results: [],
    traits: [],
    assemblyFormat: '',
  };
}

describe('filterOperations', () => {
  const operations: OperationDef[] = [
    createMockOperation('addi', 'arith', 'Integer addition'),
    createMockOperation('subi', 'arith', 'Integer subtraction'),
    createMockOperation('muli', 'arith', 'Integer multiplication'),
    createMockOperation('addf', 'arith', 'Floating point addition'),
    createMockOperation('constant', 'arith', 'Constant value'),
    createMockOperation('call', 'func', 'Function call'),
    createMockOperation('return', 'func', 'Function return'),
  ];

  it('should return all operations when query is empty', () => {
    expect(filterOperations(operations, '')).toEqual(operations);
    expect(filterOperations(operations, '   ')).toEqual(operations);
  });

  it('should filter by operation name', () => {
    const result = filterOperations(operations, 'add');
    expect(result).toHaveLength(2);
    expect(result.map(op => op.opName)).toContain('addi');
    expect(result.map(op => op.opName)).toContain('addf');
  });

  it('should filter by full name (dialect.opName)', () => {
    const result = filterOperations(operations, 'func.');
    expect(result).toHaveLength(2);
    expect(result.map(op => op.opName)).toContain('call');
    expect(result.map(op => op.opName)).toContain('return');
  });

  it('should filter by summary', () => {
    const result = filterOperations(operations, 'Integer');
    expect(result).toHaveLength(3);
    expect(result.map(op => op.opName)).toContain('addi');
    expect(result.map(op => op.opName)).toContain('subi');
    expect(result.map(op => op.opName)).toContain('muli');
  });

  it('should be case-insensitive', () => {
    const result1 = filterOperations(operations, 'ADD');
    const result2 = filterOperations(operations, 'add');
    expect(result1).toEqual(result2);
  });

  it('should return empty array when no matches', () => {
    const result = filterOperations(operations, 'nonexistent');
    expect(result).toHaveLength(0);
  });

  it('should trim whitespace from query', () => {
    const result = filterOperations(operations, '  add  ');
    expect(result).toHaveLength(2);
  });
});

describe('groupByDialect', () => {
  it('should group operations by dialect name', () => {
    const dialects: DialectInfo[] = [
      {
        name: 'arith',
        operations: [
          createMockOperation('addi', 'arith'),
          createMockOperation('subi', 'arith'),
        ],
      },
      {
        name: 'func',
        operations: [
          createMockOperation('call', 'func'),
          createMockOperation('return', 'func'),
        ],
      },
    ];

    const grouped = groupByDialect(dialects);
    
    expect(grouped.size).toBe(2);
    expect(grouped.get('arith')).toHaveLength(2);
    expect(grouped.get('func')).toHaveLength(2);
  });

  it('should merge operations from same dialect', () => {
    const dialects: DialectInfo[] = [
      {
        name: 'arith',
        operations: [createMockOperation('addi', 'arith')],
      },
      {
        name: 'arith',
        operations: [createMockOperation('subi', 'arith')],
      },
    ];

    const grouped = groupByDialect(dialects);
    
    expect(grouped.size).toBe(1);
    expect(grouped.get('arith')).toHaveLength(2);
  });

  it('should return empty map for empty input', () => {
    const grouped = groupByDialect([]);
    expect(grouped.size).toBe(0);
  });
});
