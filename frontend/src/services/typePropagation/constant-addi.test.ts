/**
 * 测试：constant 连接 addi 时的收窄
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { computePropagationWithNarrowing, applyPropagationResult } from './propagator';
import type { BlueprintNodeData, OperationDef } from '../../types';
import type { EditorNode, EditorEdge } from '../../editor/types';
import { useTypeConstraintStore } from '../../stores/typeConstraintStore';
import type { ConstraintDef } from '../../stores/typeConstraintStore';

function buildMockConstraintDefs(): Map<string, ConstraintDef> {
  const defs = new Map<string, ConstraintDef>();
  defs.set('I32', { name: 'I32', summary: '', rule: { kind: 'type', name: 'I32' } });
  defs.set('I64', { name: 'I64', summary: '', rule: { kind: 'type', name: 'I64' } });
  defs.set('F32', { name: 'F32', summary: '', rule: { kind: 'type', name: 'F32' } });
  defs.set('SignlessIntegerLike', { name: 'SignlessIntegerLike', summary: '', rule: { kind: 'oneOf', types: ['I32', 'I64'] } });
  defs.set('AnyType', { name: 'AnyType', summary: '', rule: { kind: 'any' } });
  return defs;
}

const mockEquivalences = new Map<string, string[]>([
  ['I32,I64', ['SignlessIntegerLike']],
]);

beforeAll(() => {
  useTypeConstraintStore.setState({
    buildableTypes: ['I32', 'I64', 'F32'],
    constraintDefs: buildMockConstraintDefs(),
    constraintEquivalences: mockEquivalences,
    typeDefinitions: [],
    isLoaded: true, isLoading: false, error: null,
  });
});

describe('constant → addi narrowing', () => {
  it('should narrow constant result when connected to addi', () => {
    // constant 节点
    const constantOp: OperationDef = {
      dialect: 'arith',
      opName: 'constant',
      fullName: 'arith.constant',
      summary: '',
      description: '',
      arguments: [],
      results: [{ name: 'result', typeConstraint: 'AnyType', isVariadic: false, displayName: '', description: '' }],
      traits: [],
      regions: [],
      hasRegions: false,
      isTerminator: false,
      isPure: true,
      assemblyFormat: '',
    };
    
    const constantNode: EditorNode = {
      id: 'const1',
      type: 'operation',
      position: { x: 0, y: 0 },
      data: {
        operation: constantOp,
        attributes: {},
        inputTypes: {},
        outputTypes: { result: 'AnyType' },
        pinnedTypes: {},
        execOuts: [],
        regionPins: [],
      } as BlueprintNodeData,
    };

    // addi 节点
    const addiOp: OperationDef = {
      dialect: 'arith',
      opName: 'addi',
      fullName: 'arith.addi',
      summary: '',
      description: '',
      arguments: [
        { name: 'lhs', kind: 'operand', typeConstraint: 'SignlessIntegerLike', isOptional: false, isVariadic: false, displayName: '', description: '' },
        { name: 'rhs', kind: 'operand', typeConstraint: 'SignlessIntegerLike', isOptional: false, isVariadic: false, displayName: '', description: '' },
      ],
      results: [{ name: 'result', typeConstraint: 'SignlessIntegerLike', isVariadic: false, displayName: '', description: '' }],
      traits: ['SameOperandsAndResultType'],
      regions: [],
      hasRegions: false,
      isTerminator: false,
      isPure: true,
      assemblyFormat: '',
    };
    
    const addiNode: EditorNode = {
      id: 'addi1',
      type: 'operation',
      position: { x: 0, y: 0 },
      data: {
        operation: addiOp,
        attributes: {},
        inputTypes: { lhs: 'SignlessIntegerLike', rhs: 'SignlessIntegerLike' },
        outputTypes: { result: 'SignlessIntegerLike' },
        pinnedTypes: {},
        execOuts: [],
        regionPins: [],
      } as BlueprintNodeData,
    };

    // 连线：constant.result → addi.lhs
    const edges: EditorEdge[] = [{
      id: 'e1',
      source: 'const1',
      sourceHandle: 'data-out-result',
      target: 'addi1',
      targetHandle: 'data-in-lhs',
    }];

    const nodes = [constantNode, addiNode];

    // 执行传播和收窄
    const { getConstraintElements, pickConstraintName } = useTypeConstraintStore.getState();
    const result = computePropagationWithNarrowing(nodes, edges, undefined, getConstraintElements, pickConstraintName);

    // constant.result 应该收窄为 SignlessIntegerLike
    // 因为 AnyType ∩ SignlessIntegerLike = SignlessIntegerLike
    const constantResultKey = 'const1:data-out:result';
    expect(result.narrowedConstraints.has(constantResultKey)).toBe(true);
    expect(result.narrowedConstraints.get(constantResultKey)).toBe('SignlessIntegerLike');

    // 应用结果到节点
    const updatedNodes = applyPropagationResult(nodes, result);
    const updatedConstant = updatedNodes.find(n => n.id === 'const1')!;
    const constantData = updatedConstant.data as BlueprintNodeData;
    
    // narrowedConstraints 应该包含 result
    expect(constantData.narrowedConstraints).toBeDefined();
    expect(constantData.narrowedConstraints!['result']).toBe('SignlessIntegerLike');
  });
});
