/**
 * computeOptionsExcludingSelf 测试
 * 
 * 核心场景：用户重新选择类型时，可选集不应受自己上次选择的影响
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { computeOptionsExcludingSelf } from './propagator';
import type { BlueprintNodeData, OperationDef } from '../../types';
import type { EditorNode, EditorEdge } from '../../editor/types';
import { getConstraintElements } from '../constraintResolver';
import type { ConstraintDef } from '../constraintResolver';
import { useTypeConstraintStore } from '../../stores/typeConstraintStore';

// 模拟 BuildableTypes
const mockBuildableTypes = [
  'I1', 'I8', 'I16', 'I32', 'I64', 'I128', 'Index',
  'F16', 'F32', 'F64', 'BF16',
];

// 模拟约束定义
const mockConstraintDefs = new Map<string, ConstraintDef>([
  ['AnyType', { name: 'AnyType', summary: 'any type', rule: { kind: 'like', element: { kind: 'oneOf', types: ['I1', 'I8', 'I16', 'I32', 'I64', 'I128', 'F16', 'F32', 'F64', 'BF16'] } } }],
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
  ['I32', { name: 'I32', summary: 'i32', rule: { kind: 'type', name: 'I32' } }],
  ['F32', { name: 'F32', summary: 'f32', rule: { kind: 'type', name: 'F32' } }],
]);

// 初始化 store（analyzeConstraint 依赖它）
beforeAll(() => {
  useTypeConstraintStore.setState({
    buildableTypes: mockBuildableTypes,
    constraintDefs: mockConstraintDefs,
    typeDefinitions: [],
    isLoaded: true,
    isLoading: false,
    error: null,
  });
});

const realGetConstraintElements = (constraint: string): string[] => {
  return getConstraintElements(constraint, mockConstraintDefs, mockBuildableTypes);
};

// 创建 mock 操作定义
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

// 创建操作节点
function createOperationNode(
  id: string,
  operation: OperationDef,
  pinnedTypes: Record<string, string> = {}
): EditorNode {
  const inputTypes: Record<string, string[]> = {};
  const outputTypes: Record<string, string[]> = {};

  for (const arg of operation.arguments) {
    if (arg.kind === 'operand') {
      inputTypes[arg.name] = [arg.typeConstraint];
    }
  }
  for (const result of operation.results) {
    outputTypes[result.name] = [result.typeConstraint];
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

describe('computeOptionsExcludingSelf', () => {
  describe('单源场景', () => {
    it('自己是唯一源时，可选集 = 自己约束 ∩ 邻居约束', () => {
      // A(SignlessIntegerLike) --- B(SignlessIntegerLike)
      // A pin I32
      const opA = createMockOperation(
        'arith.addi',
        [
          { name: 'lhs', typeConstraint: 'SignlessIntegerLike' },
          { name: 'rhs', typeConstraint: 'SignlessIntegerLike' },
        ],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }],
        ['SameOperandsAndResultType']
      );
      const nodeA = createOperationNode('nodeA', opA, { 'data-in-lhs': 'I32' });

      const opB = createMockOperation(
        'arith.addi',
        [
          { name: 'lhs', typeConstraint: 'SignlessIntegerLike' },
          { name: 'rhs', typeConstraint: 'SignlessIntegerLike' },
        ],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }]
      );
      const nodeB = createOperationNode('nodeB', opB);

      const nodes = [nodeA, nodeB];
      const edges: EditorEdge[] = [{
        id: 'e1',
        source: 'nodeA',
        sourceHandle: 'data-out-result',
        target: 'nodeB',
        targetHandle: 'data-in-lhs',
      }];

      // 计算 A.lhs 的可选集（排除自己）
      const options = computeOptionsExcludingSelf(
        'nodeA:data-in:lhs',
        nodes,
        edges,
        undefined,
        realGetConstraintElements
      );

      // A.lhs 的邻居是 A.result（通过 trait），A.result 连接到 B.lhs
      // 排除 A.lhs 后，没有其他源，所以邻居用原始约束
      // 可选集 = SignlessIntegerLike ∩ SignlessIntegerLike = SignlessIntegerLike 全部类型
      expect(options).toEqual(['I1', 'I8', 'I16', 'I32', 'I64', 'I128']);
    });
  });

  describe('多源场景', () => {
    it('邻居有其他源时，可选集被其他源收窄', () => {
      // A(SignlessIntegerLike) --- B(SignlessIntegerLike) --- C(I32 固定)
      // A pin I64
      const opA = createMockOperation(
        'arith.addi',
        [
          { name: 'lhs', typeConstraint: 'SignlessIntegerLike' },
          { name: 'rhs', typeConstraint: 'SignlessIntegerLike' },
        ],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }],
        ['SameOperandsAndResultType']
      );
      const nodeA = createOperationNode('nodeA', opA, { 'data-in-lhs': 'I64' });

      const opB = createMockOperation(
        'arith.addi',
        [
          { name: 'lhs', typeConstraint: 'SignlessIntegerLike' },
          { name: 'rhs', typeConstraint: 'SignlessIntegerLike' },
        ],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }],
        ['SameOperandsAndResultType']
      );
      const nodeB = createOperationNode('nodeB', opB);

      // C 的输出是固定类型 I32
      const opC = createMockOperation(
        'test.const',
        [],
        [{ name: 'result', typeConstraint: 'I32' }]
      );
      const nodeC = createOperationNode('nodeC', opC);

      const nodes = [nodeA, nodeB, nodeC];
      const edges: EditorEdge[] = [
        {
          id: 'e1',
          source: 'nodeA',
          sourceHandle: 'data-out-result',
          target: 'nodeB',
          targetHandle: 'data-in-lhs',
        },
        {
          id: 'e2',
          source: 'nodeC',
          sourceHandle: 'data-out-result',
          target: 'nodeB',
          targetHandle: 'data-in-rhs',
        },
      ];

      // 计算 A.lhs 的可选集（排除自己）
      const options = computeOptionsExcludingSelf(
        'nodeA:data-in:lhs',
        nodes,
        edges,
        undefined,
        realGetConstraintElements
      );

      // A.lhs 通过 trait 连接到 A.result（和 A.rhs）
      // A.result 连接到 B.lhs
      // B.lhs 通过 trait 连接到 B.rhs 和 B.result
      // B.rhs 连接到 C.result（固定 I32）
      // 排除 A.lhs 后，C.result 是源，传播 I32 到 B 的所有端口，再传播到 A.result
      // A.lhs 的可选集 = SignlessIntegerLike ∩ I32 = [I32]
      expect(options).toEqual(['I32']);
    });
  });

  describe('无连接场景', () => {
    it('孤立端口的可选集 = 原始约束的全部类型', () => {
      const op = createMockOperation(
        'arith.addi',
        [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }]
      );
      const node = createOperationNode('node1', op);

      const options = computeOptionsExcludingSelf(
        'node1:data-in:lhs',
        [node],
        [],
        undefined,
        realGetConstraintElements
      );

      // 无连接，无 trait，可选集 = 原始约束
      expect(options).toEqual(['I1', 'I8', 'I16', 'I32', 'I64', 'I128']);
    });
  });

  describe('Trait 内传播', () => {
    it('SameOperandsAndResultType trait 内端口互相影响', () => {
      // addi: lhs, rhs, result 通过 trait 连接
      // lhs pin I32，计算 rhs 的可选集
      const op = createMockOperation(
        'arith.addi',
        [
          { name: 'lhs', typeConstraint: 'SignlessIntegerLike' },
          { name: 'rhs', typeConstraint: 'SignlessIntegerLike' },
        ],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }],
        ['SameOperandsAndResultType']
      );
      const node = createOperationNode('node1', op, { 'data-in-lhs': 'I32' });

      // 计算 rhs 的可选集
      const options = computeOptionsExcludingSelf(
        'node1:data-in:rhs',
        [node],
        [],
        undefined,
        realGetConstraintElements
      );

      // rhs 的邻居是 lhs 和 result（通过 trait）
      // lhs 是源（pin I32），排除 rhs 后仍然是源
      // 所以 rhs 的可选集 = SignlessIntegerLike ∩ I32 = [I32]
      expect(options).toEqual(['I32']);
    });
  });
});
