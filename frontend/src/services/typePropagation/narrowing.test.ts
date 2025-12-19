/**
 * 约束收窄测试
 */

import { describe, it, expect } from 'vitest';
import { computeNarrowedConstraints } from './propagator';
import type { PropagationGraph } from './types';
import { getConstraintElements } from '../constraintResolver';
import type { ConstraintDef } from '../constraintResolver';

// 模拟后端返回的 BuildableTypes（36个）
const mockBuildableTypes = [
  'I1', 'I8', 'I16', 'I32', 'I64', 'I128', 'Index',
  'SI1', 'SI8', 'SI16', 'SI32', 'SI64',
  'UI1', 'UI8', 'UI16', 'UI32', 'UI64',
  'F16', 'F32', 'F64', 'BF16', 'F128', 'F80', 'TF32',
  'F4E2M1FN', 'F6E2M3FN', 'F6E3M2FN', 'F8E3M4', 'F8E4M3', 'F8E4M3B11FNUZ', 
  'F8E4M3FN', 'F8E4M3FNUZ', 'F8E5M2', 'F8E5M2FNUZ', 'F8E8M0FNU',
  'NoneType'
];

// 模拟约束定义
const mockConstraintDefs = new Map<string, ConstraintDef>([
  ['AnyType', { name: 'AnyType', summary: 'any type', rule: { kind: 'any' } }],
  ['SignlessIntegerLike', { 
    name: 'SignlessIntegerLike', 
    summary: 'signless integer', 
    rule: { kind: 'oneOf', types: ['I1', 'I8', 'I16', 'I32', 'I64', 'I128'] } 
  }],
  ['AnyFloat', {
    name: 'AnyFloat',
    summary: 'any float',
    rule: { kind: 'oneOf', types: ['F16', 'F32', 'F64', 'BF16'] }
  }],
  ['AnyInteger', {
    name: 'AnyInteger',
    summary: 'any integer',
    rule: { kind: 'oneOf', types: ['I1', 'I8', 'I16', 'I32', 'I64', 'I128', 'SI8', 'SI16', 'SI32', 'SI64', 'UI8', 'UI16', 'UI32', 'UI64', 'Index'] }
  }],
]);

// 模拟 constraintEquivalences（后端返回的）
const mockEquivalences = new Map<string, string[]>([
  ['I1,I128,I16,I32,I64,I8', ['AnySignlessInteger', 'SignlessIntegerLike']],
  ['BF16,F16,F32,F64', ['AnyFloat']],
]);

// 使用真实的 getConstraintElements
const realGetConstraintElements = (constraint: string): string[] => {
  return getConstraintElements(constraint, mockConstraintDefs, mockBuildableTypes);
};

// 使用真实的 pickConstraintName 逻辑
const realPickConstraintName = (types: string[]): string | null => {
  if (types.length === 0) return null;
  
  // 单一类型直接返回
  if (types.length === 1) return types[0];
  
  const key = [...types].sort().join(',');
  const equivalents = mockEquivalences.get(key);
  if (!equivalents || equivalents.length === 0) return null;
  return equivalents[0];
};

// 辅助函数：从端口对构建传播图
function buildTestGraph(connections: [string, string][]): PropagationGraph {
  const graph: PropagationGraph = new Map();
  for (const [a, b] of connections) {
    if (!graph.has(a)) graph.set(a, new Set());
    if (!graph.has(b)) graph.set(b, new Set());
    graph.get(a)!.add(b);
    graph.get(b)!.add(a);
  }
  return graph;
}

describe('Constraint Narrowing', () => {
  it('should narrow AnyInteger to SignlessIntegerLike when connected', () => {
    const portConstraints = new Map<string, string>();
    portConstraints.set('node1:data-out:result', 'AnyInteger');
    portConstraints.set('node2:data-in:lhs', 'SignlessIntegerLike');

    const graph = buildTestGraph([
      ['node1:data-out:result', 'node2:data-in:lhs']
    ]);

    const narrowed = computeNarrowedConstraints(graph, portConstraints, new Map(), new Set(), realGetConstraintElements, realPickConstraintName);

    // AnyInteger 应该收窄为 SignlessIntegerLike
    expect(narrowed.get('node1:data-out:result')).toBe('AnySignlessInteger');
    // SignlessIntegerLike 没有收窄（交集等于自身）
    expect(narrowed.has('node2:data-in:lhs')).toBe(false);
  });

  it('should narrow AnyType to AnyFloat when connected', () => {
    const portConstraints = new Map<string, string>();
    portConstraints.set('node1:data-out:result', 'AnyType');
    portConstraints.set('node2:data-in:lhs', 'AnyFloat');

    const graph = buildTestGraph([
      ['node1:data-out:result', 'node2:data-in:lhs']
    ]);

    const narrowed = computeNarrowedConstraints(graph, portConstraints, new Map(), new Set(), realGetConstraintElements, realPickConstraintName);

    // AnyType 应该收窄为 AnyFloat
    expect(narrowed.get('node1:data-out:result')).toBe('AnyFloat');
  });

  it('should narrow AnyType to SignlessIntegerLike when connected', () => {
    const portConstraints = new Map<string, string>();
    portConstraints.set('node1:data-out:result', 'AnyType');
    portConstraints.set('node2:data-in:lhs', 'SignlessIntegerLike');

    const graph = buildTestGraph([
      ['node1:data-out:result', 'node2:data-in:lhs']
    ]);

    const narrowed = computeNarrowedConstraints(graph, portConstraints, new Map(), new Set(), realGetConstraintElements, realPickConstraintName);

    // AnyType 应该收窄为 SignlessIntegerLike（或等价名）
    expect(narrowed.get('node1:data-out:result')).toBe('AnySignlessInteger');
  });

  it('should not narrow when constraints are identical', () => {
    const portConstraints = new Map<string, string>();
    portConstraints.set('node1:data-out:result', 'SignlessIntegerLike');
    portConstraints.set('node2:data-in:lhs', 'SignlessIntegerLike');

    const graph = buildTestGraph([
      ['node1:data-out:result', 'node2:data-in:lhs']
    ]);

    const narrowed = computeNarrowedConstraints(graph, portConstraints, new Map(), new Set(), realGetConstraintElements, realPickConstraintName);

    // 没有收窄
    expect(narrowed.size).toBe(0);
  });

  it('should not narrow trait neighbors when self is the only source', () => {
    // 模拟 addi: lhs, rhs, result 通过 SameOperandsAndResultType trait 连接
    // 只有 lhs 是源（用户选了类型）
    const portConstraints = new Map<string, string>();
    portConstraints.set('addi:data-in:lhs', 'SignlessIntegerLike');
    portConstraints.set('addi:data-in:rhs', 'SignlessIntegerLike');
    portConstraints.set('addi:data-out:result', 'SignlessIntegerLike');

    // trait 边（双向）
    const graph = buildTestGraph([
      ['addi:data-in:lhs', 'addi:data-in:rhs'],
      ['addi:data-in:lhs', 'addi:data-out:result'],
      ['addi:data-in:rhs', 'addi:data-out:result'],
    ]);

    // lhs 是唯一源，选了 I32
    const sourceSet = new Set(['addi:data-in:lhs']);
    const propagatedTypes = new Map<string, string>();
    propagatedTypes.set('addi:data-in:lhs', 'I32');
    propagatedTypes.set('addi:data-in:rhs', 'I32');  // 从 lhs 传播来
    propagatedTypes.set('addi:data-out:result', 'I32');  // 从 lhs 传播来

    const narrowed = computeNarrowedConstraints(graph, portConstraints, propagatedTypes, sourceSet, realGetConstraintElements, realPickConstraintName);

    // lhs 的邻居（rhs, result）能到达的源只有 lhs 自己
    // 所以 lhs 不应该被收窄（用邻居原始约束 SignlessIntegerLike）
    expect(narrowed.has('addi:data-in:lhs')).toBe(false);
  });

  it('should narrow when neighbor has other source', () => {
    // addi.lhs 连接到 external，external 是另一个源
    const portConstraints = new Map<string, string>();
    portConstraints.set('addi:data-in:lhs', 'SignlessIntegerLike');
    portConstraints.set('external:data-out:value', 'AnyInteger');

    const graph = buildTestGraph([
      ['addi:data-in:lhs', 'external:data-out:value'],
    ]);

    // external 是源，选了 I32
    const sourceSet = new Set(['external:data-out:value']);
    const propagatedTypes = new Map<string, string>();
    propagatedTypes.set('external:data-out:value', 'I32');
    propagatedTypes.set('addi:data-in:lhs', 'I32');  // 从 external 传播来

    const narrowed = computeNarrowedConstraints(graph, portConstraints, propagatedTypes, sourceSet, realGetConstraintElements, realPickConstraintName);

    // lhs 的邻居 external 是源，且不是 lhs 自己
    // 所以 lhs 应该用 external 的传播结果 I32 来收窄
    // SignlessIntegerLike ∩ {I32} = {I32}，发生收窄
    expect(narrowed.has('addi:data-in:lhs')).toBe(true);
  });
});

describe('Debug getConcreteTypes', () => {
  it('should expand AnyType to all buildable types', () => {
    const types = realGetConstraintElements('AnyType');
    console.log('AnyType expanded to:', types.length, 'types');
    expect(types.length).toBe(mockBuildableTypes.length);
  });

  it('should expand SignlessIntegerLike correctly', () => {
    const types = realGetConstraintElements('SignlessIntegerLike');
    console.log('SignlessIntegerLike expanded to:', types);
    expect(types).toEqual(['I1', 'I8', 'I16', 'I32', 'I64', 'I128']);
  });
});
