/**
 * TypeSystemService 单元测试
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import type { ConstraintDef } from '../stores/typeConstraintStore';
import { 
  computePortTypeState, 
  computeSignaturePortOptions,
  computeSignaturePortState,
  computeCallPortState,
  isPortConnected,
  getPropagatedType,
} from './typeSystemService';

// 模拟类型约束数据
const mockBuildableTypes = ['I1', 'I8', 'I16', 'I32', 'I64', 'SI8', 'SI16', 'SI32', 'SI64', 'UI8', 'UI16', 'UI32', 'UI64', 'F16', 'F32', 'F64', 'BF16', 'Index'];

function buildMockConstraintDefs(): Map<string, ConstraintDef> {
  const constraintMap: Record<string, string[]> = {
    'SignlessIntegerLike': ['I1', 'I8', 'I16', 'I32', 'I64', 'Index'],
    'AnyFloat': ['F16', 'F32', 'F64', 'BF16'],
    'AnyInteger': ['I1', 'I8', 'I16', 'I32', 'I64', 'SI8', 'SI16', 'SI32', 'SI64', 'UI8', 'UI16', 'UI32', 'UI64'],
  };
  
  const defs = new Map<string, ConstraintDef>();
  for (const t of mockBuildableTypes) {
    defs.set(t, { name: t, summary: '', rule: { kind: 'type', name: t } });
  }
  for (const [name, types] of Object.entries(constraintMap)) {
    if (!defs.has(name)) {
      defs.set(name, { name, summary: '', rule: { kind: 'oneOf', types } });
    }
  }
  return defs;
}

describe('TypeSystemService', () => {
  beforeEach(() => {
    useTypeConstraintStore.setState({
      constraintDefs: buildMockConstraintDefs(),
      buildableTypes: mockBuildableTypes,
      typeDefinitions: [],
      isLoaded: true,
      isLoading: false,
      error: null,
    });
  });

  describe('computePortTypeState', () => {
    it('should return constraint when no pinned or propagated type', () => {
      const state = computePortTypeState({
        portId: 'data-in-lhs',
        nodeId: 'node1',
        constraint: 'SignlessIntegerLike',
        pinnedTypes: {},
        propagatedType: null,
        narrowedConstraint: null,
        isConnected: false,
      });

      expect(state.displayType).toBe('SignlessIntegerLike');
      expect(state.source).toBe('constraint');
      expect(state.canEdit).toBe(true);
    });

    it('should return pinned type when user selected', () => {
      const state = computePortTypeState({
        portId: 'data-in-lhs',
        nodeId: 'node1',
        constraint: 'SignlessIntegerLike',
        pinnedTypes: { 'data-in-lhs': 'I32' },
        propagatedType: null,
        narrowedConstraint: null,
        isConnected: false,
      });

      expect(state.displayType).toBe('I32');
      expect(state.source).toBe('pinned');
      // effectiveConstraint 是原始约束 SignlessIntegerLike（多个选项），可编辑
      expect(state.canEdit).toBe(true);
    });

    it('should return propagated type when available', () => {
      const state = computePortTypeState({
        portId: 'data-in-lhs',
        nodeId: 'node1',
        constraint: 'SignlessIntegerLike',
        pinnedTypes: {},
        propagatedType: 'I64',
        narrowedConstraint: null,
        isConnected: true,
      });

      expect(state.displayType).toBe('I64');
      expect(state.source).toBe('propagated');
      // 被外部传播决定（非自己 pin），不可编辑
      expect(state.canEdit).toBe(false);
    });

    it('should be editable when pinned and connected (self-pinned is not externally determined)', () => {
      const state = computePortTypeState({
        portId: 'data-in-lhs',
        nodeId: 'node1',
        constraint: 'SignlessIntegerLike',
        pinnedTypes: { 'data-in-lhs': 'I32' },
        propagatedType: 'I32',
        narrowedConstraint: null,
        isConnected: true,
      });

      // 自己 pin 的类型不算"外部决定"，用户可以修改自己的选择
      expect(state.canEdit).toBe(true);
    });

    it('should not be editable when only one option', () => {
      const state = computePortTypeState({
        portId: 'data-in-lhs',
        nodeId: 'node1',
        constraint: 'I32', // 只有一个选项
        pinnedTypes: {},
        propagatedType: null,
        narrowedConstraint: null,
        isConnected: false,
      });

      expect(state.canEdit).toBe(false);
    });

    it('should use narrowedConstraint when provided', () => {
      const state = computePortTypeState({
        portId: 'data-in-lhs',
        nodeId: 'node1',
        constraint: 'AnyType',  // 原始约束 36 个选项
        pinnedTypes: {},
        propagatedType: null,
        narrowedConstraint: 'SignlessIntegerLike',  // 收窄到 6 个选项
        isConnected: true,
      });

      // effectiveConstraint 是收窄后的约束
      expect(state.canEdit).toBe(true);
      expect(state.options?.length).toBeLessThan(36);
    });
  });

  describe('computeSignaturePortOptions', () => {
    it('should return null when no constraints', () => {
      const options = computeSignaturePortOptions([], []);
      expect(options).toBeNull();
    });

    it('should filter by internal constraints', () => {
      const options = computeSignaturePortOptions(['SignlessIntegerLike'], []);
      
      expect(options).not.toBeNull();
      // 应该包含 SignlessIntegerLike 的所有具体类型
      expect(options).toContain('I32');
      expect(options).toContain('I64');
      expect(options).toContain('Index');
      // 不应该包含浮点类型
      expect(options).not.toContain('F32');
    });

    it('should compute intersection of multiple internal constraints', () => {
      // SignlessIntegerLike ∩ AnyInteger
      // SignlessIntegerLike = [I1, I8, I16, I32, I64, Index]
      // AnyInteger = [I1, I8, I16, I32, I64, SI8, SI16, SI32, SI64, UI8, UI16, UI32, UI64]
      // 交集 = [I1, I8, I16, I32, I64]
      const options = computeSignaturePortOptions(['SignlessIntegerLike', 'AnyInteger'], []);
      
      expect(options).not.toBeNull();
      expect(options).toContain('I32');
      expect(options).toContain('I64');
      // Index 不在 AnyInteger 中
      expect(options).not.toContain('Index');
    });

    it('should filter by external constraints', () => {
      // 外部传入 I64，内部要求 SignlessIntegerLike
      // 结果应该只有 I64（因为外部已经确定了类型）
      const options = computeSignaturePortOptions(['SignlessIntegerLike'], ['I64']);
      
      expect(options).not.toBeNull();
      expect(options).toContain('I64');
      // I32 不兼容外部的 I64
      expect(options).not.toContain('I32');
    });

    it('should include constraints when no external connections', () => {
      const options = computeSignaturePortOptions(['SignlessIntegerLike'], []);
      
      expect(options).not.toBeNull();
      // 应该包含约束本身（如果它是内部约束的子集）
      expect(options).toContain('SignlessIntegerLike');
    });
  });

  describe('computeSignaturePortState', () => {
    it('should not be editable for main function', () => {
      const state = computeSignaturePortState({
        portId: 'data-out-a',
        nodeId: 'entry',
        currentType: 'I32',
        isConnected: false,
        isMainFunction: true,
        internalConstraints: [],
        externalTypes: [],
      });

      expect(state.canEdit).toBe(false);
    });

    it('should be editable for custom function', () => {
      const state = computeSignaturePortState({
        portId: 'data-out-a',
        nodeId: 'entry',
        currentType: 'AnyType',
        isConnected: false,
        isMainFunction: false,
        internalConstraints: [],
        externalTypes: [],
      });

      expect(state.canEdit).toBe(true);
      expect(state.options).toBeNull(); // 无限制
    });
  });

  describe('computeCallPortState', () => {
    it('should return fixed state (Call nodes use same logic as Operation nodes now)', () => {
      // 注意：computeCallPortState 仍然返回 canEdit: false
      // 但实际 FunctionCallNode 组件现在使用 computePortTypeState
      const state = computeCallPortState({
        portId: 'data-in-a',
        displayType: 'I32',
      });

      expect(state.canEdit).toBe(false);
      expect(state.displayType).toBe('I32');
    });
  });

  describe('isPortConnected', () => {
    const edges = [
      { source: 'node1', target: 'node2', sourceHandle: 'data-out-result', targetHandle: 'data-in-lhs' },
      { source: 'node2', target: 'node3', sourceHandle: 'data-out-result', targetHandle: 'data-in-x' },
    ];

    it('should return true for connected source port', () => {
      expect(isPortConnected('node1', 'data-out-result', edges)).toBe(true);
    });

    it('should return true for connected target port', () => {
      expect(isPortConnected('node2', 'data-in-lhs', edges)).toBe(true);
    });

    it('should return false for unconnected port', () => {
      expect(isPortConnected('node1', 'data-in-lhs', edges)).toBe(false);
    });
  });

  describe('getPropagatedType', () => {
    it('should return type from inputTypes', () => {
      const type = getPropagatedType(
        'data-in-lhs',
        { lhs: 'I32', rhs: 'I64' },
        {}
      );
      expect(type).toBe('I32');
    });

    it('should return type from outputTypes', () => {
      const type = getPropagatedType(
        'data-out-result',
        {},
        { result: 'F32' }
      );
      expect(type).toBe('F32');
    });

    it('should return constraint name (now returns any type including constraints)', () => {
      const type = getPropagatedType(
        'data-in-lhs',
        { lhs: 'SignlessIntegerLike' }, // 约束名也会返回
        {}
      );
      // 新设计：getPropagatedType 返回任何类型，包括约束名
      expect(type).toBe('SignlessIntegerLike');
    });

    it('should handle variadic port suffix', () => {
      const type = getPropagatedType(
        'data-in-args_0',
        { args: 'I32' },
        {}
      );
      expect(type).toBe('I32');
    });
  });
});
