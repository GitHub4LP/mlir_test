/**
 * 端到端传播测试：验证 Return I32 → constant 的传播
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { buildPropagationGraph, propagateTypes, extractTypeSources, applyPropagationResult } from './propagator';
import type { BlueprintNodeData, FunctionReturnData, OperationDef } from '../../types';
import type { EditorNode, EditorEdge } from '../../editor/types';
import { useTypeConstraintStore } from '../../stores/typeConstraintStore';
import type { ConstraintDef } from '../../stores/typeConstraintStore';

function buildMockConstraintDefs(): Map<string, ConstraintDef> {
  const defs = new Map<string, ConstraintDef>();
  defs.set('I32', { name: 'I32', summary: '', rule: { kind: 'type', name: 'I32' } });
  defs.set('TypedAttrInterface', { name: 'TypedAttrInterface', summary: '', rule: null });
  return defs;
}

beforeAll(() => {
  useTypeConstraintStore.setState({
    buildableTypes: ['I32'],
    constraintDefs: buildMockConstraintDefs(),
    typeDefinitions: [],
    isLoaded: true, isLoading: false, error: null,
  });
});

describe('Return I32 → constant propagation', () => {
  it('should propagate I32 from Return to connected constant', () => {
    // 1. 创建 Return 节点（main 函数）
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
          concreteType: 'I32',
          color: '#fff',
        }],
        execIn: { id: 'exec-in', label: '' },
        isMain: true,
      } as FunctionReturnData,
    };

    // 2. 创建 constant 节点
    const constantOp: OperationDef = {
      dialect: 'arith',
      opName: 'constant',
      fullName: 'arith.constant',
      summary: '',
      description: '',
      arguments: [],
      results: [{ name: 'result', typeConstraint: 'TypedAttrInterface', isVariadic: false, displayName: '', description: '' }],
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
        outputTypes: { result: 'TypedAttrInterface' },
        pinnedTypes: {},
        execOuts: [],
        regionPins: [],
      } as BlueprintNodeData,
    };

    // 3. 创建连线
    const edges: EditorEdge[] = [{
      id: 'e1',
      source: 'const1',
      sourceHandle: 'data-out-result',
      target: 'main-return',
      targetHandle: 'data-in-result',
    }];

    const nodes = [returnNode, constantNode];

    // 4. 执行传播
    const graph = buildPropagationGraph(nodes, edges);
    const sources = extractTypeSources(nodes);
    const result = propagateTypes(graph, sources);

    // 5. 验证传播结果
    expect(result.types.get('main-return:data-in:result')).toBe('I32');
    expect(result.types.get('const1:data-out:result')).toBe('I32');

    // 6. 应用传播结果
    const updatedNodes = applyPropagationResult(nodes, result);
    const updatedConstant = updatedNodes.find(n => n.id === 'const1');
    const constantData = updatedConstant?.data as BlueprintNodeData;

    expect(constantData.outputTypes?.result).toBe('I32');
  });
});
