/**
 * Graph Converter Tests
 * 
 * 测试前端图到后端 API 格式的转换逻辑。
 */

import { describe, it, expect } from 'vitest';
import { convertToBackendGraph } from './graphConverter';
import type { GraphNode, GraphEdge, BlueprintNodeData, FunctionEntryData, FunctionReturnData, OperationDef } from '../types';

// 创建测试用的操作定义
function createOperationDef(
  fullName: string,
  operands: { name: string; typeConstraint: string }[],
  results: { name: string; typeConstraint: string }[],
  attributes: { name: string; typeConstraint: string }[] = []
): OperationDef {
  return {
    opName: fullName.split('.')[1],
    fullName,
    dialect: fullName.split('.')[0],
    summary: '',
    description: '',
    arguments: [
      ...operands.map(o => ({
        name: o.name,
        kind: 'operand' as const,
        typeConstraint: o.typeConstraint,
        displayName: o.typeConstraint,
        description: '',
        isOptional: false,
        isVariadic: false,
      })),
      ...attributes.map(a => ({
        name: a.name,
        kind: 'attribute' as const,
        typeConstraint: a.typeConstraint,
        displayName: a.typeConstraint,
        description: '',
        isOptional: false,
        isVariadic: false,
      })),
    ],
    results: results.map(r => ({
      name: r.name,
      typeConstraint: r.typeConstraint,
      displayName: r.typeConstraint,
      description: '',
      isVariadic: false,
    })),
    traits: [],
    regions: [],
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
  outputTypes: Record<string, string>,
  attributes: Record<string, string> = {}
): GraphNode {
  const data: BlueprintNodeData = {
    operation,
    attributes,
    inputTypes: {},
    outputTypes,
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

// 创建函数入口节点
function createEntryNode(id: string, outputs: { id: string; name: string; concreteType: string }[]): GraphNode {
  const data: FunctionEntryData = {
    functionId: 'test-func',
    functionName: 'test',
    outputs: outputs.map(o => ({
      id: o.id,
      name: o.name,
      kind: 'output' as const,
      typeConstraint: o.concreteType,
      concreteType: o.concreteType,
      color: '#fff',
    })),
    execOut: { id: 'exec-out', label: '' },
    isMain: true,
  };
  return {
    id,
    type: 'function-entry',
    position: { x: 0, y: 0 },
    data,
  };
}

// 创建函数返回节点
function createReturnNode(id: string, inputs: { id: string; name: string; concreteType: string }[]): GraphNode {
  const data: FunctionReturnData = {
    functionId: 'test-func',
    functionName: 'test',
    branchName: '',
    inputs: inputs.map(i => ({
      id: i.id,
      name: i.name,
      kind: 'input' as const,
      typeConstraint: i.concreteType,
      concreteType: i.concreteType,
      color: '#fff',
    })),
    execIn: { id: 'exec-in', label: '' },
    isMain: true,
  };
  return {
    id,
    type: 'function-return',
    position: { x: 0, y: 0 },
    data,
  };
}

describe('convertToBackendGraph', () => {
  describe('节点转换', () => {
    it('应该转换函数入口节点', () => {
      const entryNode = createEntryNode('entry', [
        { id: 'param-a', name: 'a', concreteType: 'I32' },
        { id: 'param-b', name: 'b', concreteType: 'I32' },
      ]);
      
      const result = convertToBackendGraph([entryNode], []);
      
      expect(result.nodes).toHaveLength(1);
      expect(result.nodes[0]).toEqual({
        id: 'entry',
        op_name: 'function-entry',
        result_types: ['I32', 'I32'],
        attributes: {},
        region_graphs: [],
      });
    });

    it('应该转换函数返回节点', () => {
      const returnNode = createReturnNode('return', [
        { id: 'return-result', name: 'result', concreteType: 'I32' },
      ]);
      
      const result = convertToBackendGraph([returnNode], []);
      
      expect(result.nodes).toHaveLength(1);
      expect(result.nodes[0]).toEqual({
        id: 'return',
        op_name: 'function-return',
        result_types: ['I32'],
        attributes: {},
        region_graphs: [],
      });
    });

    it('应该转换操作节点', () => {
      const operation = createOperationDef(
        'arith.addi',
        [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }, { name: 'rhs', typeConstraint: 'SignlessIntegerLike' }],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }]
      );
      const opNode = createOperationNode('add', operation, { result: 'I32' });
      
      const result = convertToBackendGraph([opNode], []);
      
      expect(result.nodes).toHaveLength(1);
      expect(result.nodes[0]).toEqual({
        id: 'add',
        op_name: 'arith.addi',
        result_types: ['I32'],
        attributes: {},
        region_graphs: [],
      });
    });

    it('应该转换带属性的操作节点', () => {
      const operation = createOperationDef(
        'arith.constant',
        [],
        [{ name: 'result', typeConstraint: 'AnyType' }],
        [{ name: 'value', typeConstraint: 'TypedAttrInterface' }]
      );
      const opNode = createOperationNode('const', operation, { result: 'I32' }, { value: '42' });
      
      const result = convertToBackendGraph([opNode], []);
      
      expect(result.nodes).toHaveLength(1);
      // 属性值会被格式化为 TypedAttr 格式
      expect(result.nodes[0].attributes).toEqual({ value: '42 : I32' });
    });

    it('应该保留已格式化的属性值', () => {
      const operation = createOperationDef(
        'arith.constant',
        [],
        [{ name: 'result', typeConstraint: 'AnyType' }],
        [{ name: 'value', typeConstraint: 'TypedAttrInterface' }]
      );
      const opNode = createOperationNode('const', operation, { result: 'I32' }, { value: '100 : i32' });
      
      const result = convertToBackendGraph([opNode], []);
      
      expect(result.nodes[0].attributes).toEqual({ value: '100 : i32' });
    });
  });

  describe('边转换', () => {
    it('应该转换操作之间的数据边', () => {
      const op1 = createOperationDef(
        'arith.constant',
        [],
        [{ name: 'result', typeConstraint: 'AnyType' }]
      );
      const op2 = createOperationDef(
        'arith.addi',
        [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }, { name: 'rhs', typeConstraint: 'SignlessIntegerLike' }],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }]
      );
      
      const nodes: GraphNode[] = [
        createOperationNode('const1', op1, { result: 'I32' }),
        createOperationNode('const2', op1, { result: 'I32' }),
        createOperationNode('add', op2, { result: 'I32' }),
      ];
      
      const edges: GraphEdge[] = [
        { id: 'e1', source: 'const1', target: 'add', sourceHandle: 'output-result', targetHandle: 'input-lhs' },
        { id: 'e2', source: 'const2', target: 'add', sourceHandle: 'output-result', targetHandle: 'input-rhs' },
      ];
      
      const result = convertToBackendGraph(nodes, edges);
      
      expect(result.edges).toHaveLength(2);
      expect(result.edges[0]).toEqual({
        source_node: 'const1',
        source_output: 0,
        target_node: 'add',
        target_input: 0,
      });
      expect(result.edges[1]).toEqual({
        source_node: 'const2',
        source_output: 0,
        target_node: 'add',
        target_input: 1,
      });
    });

    it('应该转换入口节点到操作的边', () => {
      const operation = createOperationDef(
        'arith.addi',
        [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }, { name: 'rhs', typeConstraint: 'SignlessIntegerLike' }],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }]
      );
      
      const nodes: GraphNode[] = [
        createEntryNode('entry', [
          { id: 'param-a', name: 'a', concreteType: 'I32' },
          { id: 'param-b', name: 'b', concreteType: 'I32' },
        ]),
        createOperationNode('add', operation, { result: 'I32' }),
      ];
      
      const edges: GraphEdge[] = [
        { id: 'e1', source: 'entry', target: 'add', sourceHandle: 'param-a', targetHandle: 'input-lhs' },
        { id: 'e2', source: 'entry', target: 'add', sourceHandle: 'param-b', targetHandle: 'input-rhs' },
      ];
      
      const result = convertToBackendGraph(nodes, edges);
      
      expect(result.edges).toHaveLength(2);
      expect(result.edges[0]).toEqual({
        source_node: 'entry',
        source_output: 0,
        target_node: 'add',
        target_input: 0,
      });
      expect(result.edges[1]).toEqual({
        source_node: 'entry',
        source_output: 1,
        target_node: 'add',
        target_input: 1,
      });
    });

    it('应该转换操作到返回节点的边', () => {
      const operation = createOperationDef(
        'arith.addi',
        [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }, { name: 'rhs', typeConstraint: 'SignlessIntegerLike' }],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }]
      );
      
      const nodes: GraphNode[] = [
        createOperationNode('add', operation, { result: 'I32' }),
        createReturnNode('return', [{ id: 'return-result', name: 'result', concreteType: 'I32' }]),
      ];
      
      const edges: GraphEdge[] = [
        { id: 'e1', source: 'add', target: 'return', sourceHandle: 'output-result', targetHandle: 'return-result' },
      ];
      
      const result = convertToBackendGraph(nodes, edges);
      
      expect(result.edges).toHaveLength(1);
      expect(result.edges[0]).toEqual({
        source_node: 'add',
        source_output: 0,
        target_node: 'return',
        target_input: 0,
      });
    });

    it('应该跳过执行边', () => {
      const operation = createOperationDef(
        'arith.constant',
        [],
        [{ name: 'result', typeConstraint: 'AnyType' }]
      );
      
      const nodes: GraphNode[] = [
        createEntryNode('entry', []),
        createOperationNode('const', operation, { result: 'I32' }),
        createReturnNode('return', []),
      ];
      
      const edges: GraphEdge[] = [
        { id: 'e1', source: 'entry', target: 'const', sourceHandle: 'exec-out', targetHandle: 'exec-in' },
        { id: 'e2', source: 'const', target: 'return', sourceHandle: 'exec-out', targetHandle: 'exec-in' },
      ];
      
      const result = convertToBackendGraph(nodes, edges);
      
      expect(result.edges).toHaveLength(0);
    });
  });

  describe('完整图转换', () => {
    it('应该转换简单的加法函数图', () => {
      const addiOp = createOperationDef(
        'arith.addi',
        [{ name: 'lhs', typeConstraint: 'SignlessIntegerLike' }, { name: 'rhs', typeConstraint: 'SignlessIntegerLike' }],
        [{ name: 'result', typeConstraint: 'SignlessIntegerLike' }]
      );
      
      const nodes: GraphNode[] = [
        createEntryNode('entry', [
          { id: 'param-a', name: 'a', concreteType: 'I32' },
          { id: 'param-b', name: 'b', concreteType: 'I32' },
        ]),
        createOperationNode('add', addiOp, { result: 'I32' }),
        createReturnNode('return', [{ id: 'return-result', name: 'result', concreteType: 'I32' }]),
      ];
      
      const edges: GraphEdge[] = [
        { id: 'e1', source: 'entry', target: 'add', sourceHandle: 'param-a', targetHandle: 'input-lhs' },
        { id: 'e2', source: 'entry', target: 'add', sourceHandle: 'param-b', targetHandle: 'input-rhs' },
        { id: 'e3', source: 'add', target: 'return', sourceHandle: 'output-result', targetHandle: 'return-result' },
      ];
      
      const result = convertToBackendGraph(nodes, edges);
      
      expect(result.nodes).toHaveLength(3);
      expect(result.edges).toHaveLength(3);
      
      // 验证节点
      const entryBackend = result.nodes.find(n => n.id === 'entry');
      expect(entryBackend?.op_name).toBe('function-entry');
      expect(entryBackend?.result_types).toEqual(['I32', 'I32']);
      
      const addBackend = result.nodes.find(n => n.id === 'add');
      expect(addBackend?.op_name).toBe('arith.addi');
      expect(addBackend?.result_types).toEqual(['I32']);
      
      const returnBackend = result.nodes.find(n => n.id === 'return');
      expect(returnBackend?.op_name).toBe('function-return');
      expect(returnBackend?.result_types).toEqual(['I32']);
    });
  });
});
