/**
 * Type Propagator Tests
 */

import { describe, it, expect, beforeAll } from 'vitest';
import type { Node } from '@xyflow/react';
import { extractTypeSources, buildPropagationGraph, propagateTypes } from './propagator';
import type { BlueprintNodeData, OperationDef } from '../../types';
import { useTypeConstraintStore } from '../../stores/typeConstraintStore';

// 初始化类型约束 store
beforeAll(() => {
  useTypeConstraintStore.setState({
    buildableTypes: ['I1', 'I8', 'I16', 'I32', 'I64', 'Index'],
    constraintMap: {
      'SignlessIntegerLike': ['I1', 'I8', 'I16', 'I32', 'I64'],
      'BoolLike': ['I1'],  // 单一类型约束
      'Index': ['Index'],   // 单一类型约束
      'I1': ['I1'],
      'I32': ['I32'],
    },
    isLoaded: true,
    isLoading: false,
    error: null,
  });
});

// Helper to create a mock operation
function createMockOperation(
  fullName: string,
  inputs: { name: string; typeConstraint: string }[],
  outputs: { name: string; typeConstraint: string }[],
  traits: string[] = []
): OperationDef {
  return {
    dialect: fullName.split('.')[0],
    opName: fullName.split('.')[1],
    fullName,
    summary: '',
    description: '',
    arguments: inputs.map(i => ({
      name: i.name,
      kind: 'operand' as const,
      typeConstraint: i.typeConstraint,
      displayName: i.typeConstraint,
      description: '',
      isOptional: false,
      isVariadic: false,
    })),
    results: outputs.map(o => ({
      name: o.name,
      typeConstraint: o.typeConstraint,
      displayName: o.typeConstraint,
      description: '',
      isVariadic: false,
    })),
    regions: [],
    traits,
    assemblyFormat: '',
    hasRegions: false,
    isTerminator: false,
    isPure: true,
  };
}

// Helper to create an operation node
function createOperationNode(
  id: string,
  operation: OperationDef,
  pinnedTypes: Record<string, string> = {}
): Node {
  const inputTypes: Record<string, string> = {};
  const outputTypes: Record<string, string> = {};

  for (const arg of operation.arguments) {
    if (arg.kind === 'operand') {
      inputTypes[arg.name] = arg.typeConstraint;
    }
  }
  for (const result of operation.results) {
    outputTypes[result.name] = result.typeConstraint;
  }

  const data: BlueprintNodeData = {
    operation,
    attributes: {},
    inputTypes,
    outputTypes,
    pinnedTypes,
    execOuts: [],
    regionPins: [],
  };

  return {
    id,
    type: 'operation',
    position: { x: 0, y: 0 },
    data,
  };
}

describe('extractTypeSources', () => {
  it('should extract pinned types from operation nodes', () => {
    const op = createMockOperation(
      'arith.addi',
      [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }],
      [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }],
      ['SameOperandsAndResultType']
    );
    const node = createOperationNode('node1', op, { 'input-lhs': 'I32' });

    const sources = extractTypeSources([node]);

    expect(sources).toContainEqual({
      nodeId: 'node1',
      portId: 'input-lhs',
      type: 'I32',
    });
  });

  it('should auto-resolve single-type constraints (BoolLike → I1)', () => {
    const op = createMockOperation(
      'arith.cmpi',
      [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }],
      [{ name: 'result', typeConstraint: 'BoolLike' }]  // 单一类型约束
    );
    const node = createOperationNode('node1', op);

    const sources = extractTypeSources([node]);

    // BoolLike 应该自动解析为 I1
    expect(sources).toContainEqual({
      nodeId: 'node1',
      portId: 'output-result',
      type: 'I1',
    });
  });

  it('should auto-resolve Index constraint', () => {
    const op = createMockOperation(
      'arith.index_cast',
      [{ name: 'in', typeConstraint: 'SignlessIntegerLike' }],
      [{ name: 'out', typeConstraint: 'Index' }]  // 单一类型约束
    );
    const node = createOperationNode('node1', op);

    const sources = extractTypeSources([node]);

    expect(sources).toContainEqual({
      nodeId: 'node1',
      portId: 'output-out',
      type: 'Index',
    });
  });

  it('should NOT auto-resolve multi-type constraints', () => {
    const op = createMockOperation(
      'arith.addi',
      [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }],
      [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }]
    );
    const node = createOperationNode('node1', op);

    const sources = extractTypeSources([node]);

    // SignlessIntegerLike 有多个类型，不应该自动解析
    expect(sources).not.toContainEqual(
      expect.objectContaining({ portId: 'input-lhs' })
    );
    expect(sources).not.toContainEqual(
      expect.objectContaining({ portId: 'output-result' })
    );
  });

  it('should prefer pinned type over auto-resolved type', () => {
    const op = createMockOperation(
      'test.op',
      [{ name: 'in', typeConstraint: 'BoolLike' }],
      []
    );
    // 用户显式选择了 I1（虽然 BoolLike 也会解析为 I1）
    const node = createOperationNode('node1', op, { 'input-in': 'I1' });

    const sources = extractTypeSources([node]);

    // 应该只有一个源（不重复）
    const inSources = sources.filter(s => s.portId === 'input-in');
    expect(inSources).toHaveLength(1);
    expect(inSources[0].type).toBe('I1');
  });
});

describe('type propagation with auto-resolved types', () => {
  it('should propagate auto-resolved BoolLike type through connections', () => {
    // arith.cmpi 输出 BoolLike，连接到另一个节点
    const cmpiOp = createMockOperation(
      'arith.cmpi',
      [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }],
      [{ name: 'result', typeConstraint: 'BoolLike' }]
    );
    const cmpiNode = createOperationNode('cmpi', cmpiOp);

    const selectOp = createMockOperation(
      'arith.select',
      [
        { name: 'condition', typeConstraint: 'BoolLike' },
        { name: 'true_value', typeConstraint: 'SignlessIntegerLike' },
      ],
      [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }]
    );
    const selectNode = createOperationNode('select', selectOp);

    const nodes = [cmpiNode, selectNode];
    const edges = [{
      id: 'e1',
      source: 'cmpi',
      sourceHandle: 'output-result',
      target: 'select',
      targetHandle: 'input-condition',
    }];

    // 构建传播图并传播
    const graph = buildPropagationGraph(nodes, edges);
    const sources = extractTypeSources(nodes);
    const result = propagateTypes(graph, sources);

    // cmpi 的输出应该是 I1（自动解析）
    expect(result.types.get('cmpi:output-result')).toBe('I1');
    // select 的 condition 输入应该通过连接传播得到 I1
    expect(result.types.get('select:input-condition')).toBe('I1');
  });
});
