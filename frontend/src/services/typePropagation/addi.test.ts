/**
 * 测试：Return I32 → addi 的传播
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { buildPropagationGraph, propagateTypes, extractTypeSources } from './propagator';
import type { BlueprintNodeData, FunctionReturnData, OperationDef } from '../../types';
import type { EditorNode, EditorEdge } from '../../editor/types';
import { useTypeConstraintStore } from '../../stores/typeConstraintStore';
import type { ConstraintDef } from '../../stores/typeConstraintStore';

function buildMockConstraintDefs(): Map<string, ConstraintDef> {
  const defs = new Map<string, ConstraintDef>();
  defs.set('I32', { name: 'I32', summary: '', rule: { kind: 'type', name: 'I32' } });
  defs.set('I64', { name: 'I64', summary: '', rule: { kind: 'type', name: 'I64' } });
  defs.set('SignlessIntegerLike', { name: 'SignlessIntegerLike', summary: '', rule: { kind: 'oneOf', types: ['I32', 'I64'] } });
  return defs;
}

beforeAll(() => {
  useTypeConstraintStore.setState({
    buildableTypes: ['I32', 'I64'],
    constraintDefs: buildMockConstraintDefs(),
    typeDefinitions: [],
    isLoaded: true, isLoading: false, error: null,
  });
});

describe('Return I32 → addi propagation', () => {
  it('should propagate I32 from Return to connected addi', () => {
    // 1. Return 节点
    const returnNode: EditorNode = {
      id: 'main-return',
      type: 'function-return',
      position: { x: 0, y: 0 },
      data: {
        functionId: 'main',
        functionName: 'main',
        branchName: '',
        inputs: [{
          id: 'data-in-result',
          name: 'result',
          kind: 'input',
          typeConstraint: 'I32',
          color: '#fff',
        }],
        execIn: { id: 'exec-in', label: '' },
        isMain: true,
      } as FunctionReturnData,
    };

    // 2. addi 节点
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

    // 3. 连线：addi.result → return.result
    const edges: EditorEdge[] = [{
      id: 'e1',
      source: 'addi1',
      sourceHandle: 'data-out-result',
      target: 'main-return',
      targetHandle: 'data-in-result',
    }];

    const nodes = [returnNode, addiNode];

    // 4. 执行传播
    const graph = buildPropagationGraph(nodes, edges);
    const sources = extractTypeSources(nodes);
    const result = propagateTypes(graph, sources);

    console.log('=== addi test ===');
    console.log('Graph:', [...graph.entries()].map(([k, v]) => `${k} -> [${[...v].join(', ')}]`));
    console.log('Sources:', sources.map(s => `${s.portRef.key} = ${s.type}`));
    console.log('Result:', [...result.types.entries()]);

    // 5. 验证
    // Return 的 I32 应该传播到 addi 的 result
    expect(result.types.get('main-return:data-in:result')).toBe('I32');
    expect(result.types.get('addi1:data-out:result')).toBe('I32');
    
    // SameOperandsAndResultType trait 应该让 lhs 和 rhs 也变成 I32
    expect(result.types.get('addi1:data-in:lhs')).toBe('I32');
    expect(result.types.get('addi1:data-in:rhs')).toBe('I32');
  });
});
