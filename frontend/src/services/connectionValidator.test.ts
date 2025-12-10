/**
 * Connection Validator Tests
 * 
 * Tests for connection validation logic.
 * Requirements: 7.1, 7.2, 7.3
 */

import { describe, it, expect, beforeAll } from 'vitest';
import type { Node, Connection } from '@xyflow/react';
import {
  validateConnection,
  getPortTypeConstraint,
  getNodeErrors,
} from './connectionValidator';
import type { BlueprintNodeData, FunctionEntryData, FunctionReturnData, OperationDef } from '../types';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import type { ConstraintDef } from '../stores/typeConstraintStore';

// 构建 mock constraintDefs
function buildMockConstraintDefs(): Map<string, ConstraintDef> {
  const buildableTypes = [
    'I1', 'I8', 'I16', 'I32', 'I64', 'I128',
    'F16', 'F32', 'F64', 'F80', 'F128', 'BF16', 'TF32',
    'Index',
  ];
  
  const constraintMap: Record<string, string[]> = {
    'SignlessIntegerLike': ['I1', 'I8', 'I16', 'I32', 'I64', 'I128'],
    'AnyFloat': ['F16', 'F32', 'F64', 'F80', 'F128', 'BF16', 'TF32'],
    'AnyType': buildableTypes,
  };
  
  const defs = new Map<string, ConstraintDef>();
  for (const t of buildableTypes) {
    defs.set(t, { name: t, summary: '', rule: { kind: 'type', name: t } });
  }
  for (const [name, types] of Object.entries(constraintMap)) {
    if (!defs.has(name)) {
      defs.set(name, { name, summary: '', rule: { kind: 'oneOf', types } });
    }
  }
  return defs;
}

// 初始化类型约束 store（模拟后端数据）
beforeAll(() => {
  const buildableTypes = [
    'I1', 'I8', 'I16', 'I32', 'I64', 'I128',
    'F16', 'F32', 'F64', 'F80', 'F128', 'BF16', 'TF32',
    'Index',
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

// Helper to create a mock operation node
function createOperationNode(
  id: string,
  inputTypes: Record<string, string>,
  outputTypes: Record<string, string>,
  args: { name: string; kind: 'operand' | 'attribute'; isOptional: boolean }[] = []
): Node {
  const operation: OperationDef = {
    dialect: 'arith',
    opName: 'test',
    fullName: 'arith.test',
    summary: 'Test operation',
    description: '',
    arguments: args.length > 0 ? args.map(a => ({
      name: a.name,
      kind: a.kind,
      typeConstraint: inputTypes[a.name] || 'AnyType',
      displayName: inputTypes[a.name] || 'AnyType',
      description: '',
      isOptional: a.isOptional,
      isVariadic: false,
    })) : Object.keys(inputTypes).map(name => ({
      name,
      kind: 'operand' as const,
      typeConstraint: inputTypes[name],
      displayName: inputTypes[name],
      description: '',
      isOptional: false,
      isVariadic: false,
    })),
    results: Object.keys(outputTypes).map(name => ({
      name,
      typeConstraint: outputTypes[name],
      displayName: outputTypes[name],
      description: '',
      isVariadic: false,
    })),
    traits: [],
    assemblyFormat: '',
    regions: [],
    hasRegions: false,
    isTerminator: false,
    isPure: true,
  };

  const data: BlueprintNodeData = {
    operation,
    attributes: {},
    inputTypes,
    outputTypes,
    execIn: { id: 'exec-in', label: '' },
    execOuts: [{ id: 'exec-out', label: '' }],
    regionPins: [],
  };

  return {
    id,
    type: 'operation',
    position: { x: 0, y: 0 },
    data,
  };
}

// Helper to create a function entry node
function createFunctionEntryNode(
  id: string,
  outputs: { id: string; name: string; typeConstraint: string; concreteType?: string }[]
): Node {
  const data: FunctionEntryData = {
    functionId: 'test-func',
    functionName: 'test-func',
    outputs: outputs.map(o => ({
      id: o.id,
      name: o.name,
      kind: 'output' as const,
      typeConstraint: o.typeConstraint,
      concreteType: o.concreteType,
      color: '#fff',
    })),
    execOut: { id: 'exec-out', label: '' },
    isMain: false,
  };

  return {
    id,
    type: 'function-entry',
    position: { x: 0, y: 0 },
    data,
  };
}

// Helper to create a function return node
function createFunctionReturnNode(
  id: string,
  inputs: { id: string; name: string; typeConstraint: string; concreteType?: string }[]
): Node {
  const data: FunctionReturnData = {
    functionId: 'test-func',
    functionName: 'test-func',
    inputs: inputs.map(i => ({
      id: i.id,
      name: i.name,
      kind: 'input' as const,
      typeConstraint: i.typeConstraint,
      concreteType: i.concreteType,
      color: '#fff',
    })),
    branchName: '',
    execIn: { id: 'exec-in', label: '' },
    isMain: false,
  };

  return {
    id,
    type: 'function-return',
    position: { x: 0, y: 0 },
    data,
  };
}

describe('connectionValidator', () => {
  describe('getPortTypeConstraint', () => {
    it('should get output type from operation node', () => {
      const node = createOperationNode('node1', {}, { result: 'I32' });
      const type = getPortTypeConstraint(node, 'data-out-result', true);
      expect(type).toBe('I32');
    });

    it('should get input type from operation node', () => {
      const node = createOperationNode('node1', { lhs: 'SignlessIntegerLike' }, {});
      const type = getPortTypeConstraint(node, 'data-in-lhs', false);
      expect(type).toBe('SignlessIntegerLike');
    });

    it('should get output type from function entry node', () => {
      const node = createFunctionEntryNode('entry', [
        { id: 'data-out-x', name: 'x', typeConstraint: 'I32' },
      ]);
      const type = getPortTypeConstraint(node, 'data-out-x', true);
      expect(type).toBe('I32');
    });

    it('should get input type from function return node', () => {
      const node = createFunctionReturnNode('return', [
        { id: 'data-in-result', name: 'result', typeConstraint: 'F32' },
      ]);
      const type = getPortTypeConstraint(node, 'data-in-result', false);
      expect(type).toBe('F32');
    });

    it('should prefer resolved type over constraint', () => {
      const node = createOperationNode('node1', { lhs: 'SignlessIntegerLike' }, {});
      const resolvedTypes = new Map([
        ['node1', new Map([['data-in-lhs', 'I32']])],
      ]);
      const type = getPortTypeConstraint(node, 'data-in-lhs', false, resolvedTypes);
      expect(type).toBe('I32');
    });
  });

  describe('validateConnection', () => {
    it('should allow connection between compatible concrete types', () => {
      const sourceNode = createOperationNode('source', {}, { result: 'I32' });
      const targetNode = createOperationNode('target', { input: 'I32' }, {});

      const connection: Connection = {
        source: 'source',
        sourceHandle: 'data-out-result',
        target: 'target',
        targetHandle: 'data-in-input',
      };

      const result = validateConnection(connection, [sourceNode, targetNode]);
      expect(result.isValid).toBe(true);
    });

    it('should allow connection when concrete type satisfies abstract constraint', () => {
      const sourceNode = createOperationNode('source', {}, { result: 'I32' });
      const targetNode = createOperationNode('target', { input: 'SignlessIntegerLike' }, {});

      const connection: Connection = {
        source: 'source',
        sourceHandle: 'data-out-result',
        target: 'target',
        targetHandle: 'data-in-input',
      };

      const result = validateConnection(connection, [sourceNode, targetNode]);
      expect(result.isValid).toBe(true);
    });

    it('should reject connection between incompatible types', () => {
      const sourceNode = createOperationNode('source', {}, { result: 'F32' });
      const targetNode = createOperationNode('target', { input: 'SignlessIntegerLike' }, {});

      const connection: Connection = {
        source: 'source',
        sourceHandle: 'data-out-result',
        target: 'target',
        targetHandle: 'data-in-input',
      };

      const result = validateConnection(connection, [sourceNode, targetNode]);
      expect(result.isValid).toBe(false);
      expect(result.errorMessage).toContain('Type mismatch');
    });

    it('should reject self-connections', () => {
      const node = createOperationNode('node1', { input: 'I32' }, { result: 'I32' });

      const connection: Connection = {
        source: 'node1',
        sourceHandle: 'data-out-result',
        target: 'node1',
        targetHandle: 'data-in-input',
      };

      const result = validateConnection(connection, [node]);
      expect(result.isValid).toBe(false);
      expect(result.errorMessage).toContain('Cannot connect a node to itself');
    });

    it('should reject connection with missing source node', () => {
      const targetNode = createOperationNode('target', { input: 'I32' }, {});

      const connection: Connection = {
        source: 'nonexistent',
        sourceHandle: 'data-out-result',
        target: 'target',
        targetHandle: 'data-in-input',
      };

      const result = validateConnection(connection, [targetNode]);
      expect(result.isValid).toBe(false);
      expect(result.errorMessage).toContain('Source node not found');
    });

    it('should reject connection with missing target node', () => {
      const sourceNode = createOperationNode('source', {}, { result: 'I32' });

      const connection: Connection = {
        source: 'source',
        sourceHandle: 'data-out-result',
        target: 'nonexistent',
        targetHandle: 'data-in-input',
      };

      const result = validateConnection(connection, [sourceNode]);
      expect(result.isValid).toBe(false);
      expect(result.errorMessage).toContain('Target node not found');
    });

    it('should use resolved types when available', () => {
      const sourceNode = createOperationNode('source', {}, { result: 'SignlessIntegerLike' });
      const targetNode = createOperationNode('target', { input: 'I32' }, {});

      // Without resolved types, abstract -> concrete might not work
      // With resolved types showing I32, it should work
      const resolvedTypes = new Map([
        ['source', new Map([['data-out-result', 'I32']])],
      ]);

      const connection: Connection = {
        source: 'source',
        sourceHandle: 'data-out-result',
        target: 'target',
        targetHandle: 'data-in-input',
      };

      const result = validateConnection(connection, [sourceNode, targetNode], resolvedTypes);
      expect(result.isValid).toBe(true);
    });

    it('should allow connection from function entry to operation', () => {
      const entryNode = createFunctionEntryNode('entry', [
        { id: 'data-out-x', name: 'x', typeConstraint: 'I32', concreteType: 'I32' },
      ]);
      const opNode = createOperationNode('op', { lhs: 'SignlessIntegerLike' }, {});

      const connection: Connection = {
        source: 'entry',
        sourceHandle: 'data-out-x',
        target: 'op',
        targetHandle: 'data-in-lhs',
      };

      const result = validateConnection(connection, [entryNode, opNode]);
      expect(result.isValid).toBe(true);
    });

    it('should allow connection from operation to function return', () => {
      const opNode = createOperationNode('op', {}, { result: 'I32' });
      const returnNode = createFunctionReturnNode('return', [
        { id: 'data-in-result', name: 'result', typeConstraint: 'SignlessIntegerLike' },
      ]);

      const connection: Connection = {
        source: 'op',
        sourceHandle: 'data-out-result',
        target: 'return',
        targetHandle: 'data-in-result',
      };

      const result = validateConnection(connection, [opNode, returnNode]);
      expect(result.isValid).toBe(true);
    });
  });

  describe('getNodeErrors', () => {
    it('should return error for unconnected required input', () => {
      const node = createOperationNode(
        'node1',
        { lhs: 'I32', rhs: 'I32' },
        { result: 'I32' },
        [
          { name: 'lhs', kind: 'operand', isOptional: false },
          { name: 'rhs', kind: 'operand', isOptional: false },
        ]
      );

      const edges = [
        { source: 'other', sourceHandle: 'out', target: 'node1', targetHandle: 'data-in-lhs' },
      ];

      const errors = getNodeErrors(node, edges);
      expect(errors).toHaveLength(1);
      expect(errors[0]).toContain('rhs');
      expect(errors[0]).toContain('not connected');
    });

    it('should not return error for connected required inputs', () => {
      const node = createOperationNode(
        'node1',
        { lhs: 'I32', rhs: 'I32' },
        { result: 'I32' },
        [
          { name: 'lhs', kind: 'operand', isOptional: false },
          { name: 'rhs', kind: 'operand', isOptional: false },
        ]
      );

      const edges = [
        { source: 'other1', sourceHandle: 'out', target: 'node1', targetHandle: 'data-in-lhs' },
        { source: 'other2', sourceHandle: 'out', target: 'node1', targetHandle: 'data-in-rhs' },
      ];

      const errors = getNodeErrors(node, edges);
      expect(errors).toHaveLength(0);
    });

    it('should not return error for unconnected optional input', () => {
      const node = createOperationNode(
        'node1',
        { lhs: 'I32', optional: 'I32' },
        { result: 'I32' },
        [
          { name: 'lhs', kind: 'operand', isOptional: false },
          { name: 'optional', kind: 'operand', isOptional: true },
        ]
      );

      const edges = [
        { source: 'other', sourceHandle: 'out', target: 'node1', targetHandle: 'data-in-lhs' },
      ];

      const errors = getNodeErrors(node, edges);
      expect(errors).toHaveLength(0);
    });

    it('should return empty array for non-operation nodes', () => {
      const entryNode = createFunctionEntryNode('entry', []);
      const errors = getNodeErrors(entryNode, []);
      expect(errors).toHaveLength(0);
    });
  });
});
