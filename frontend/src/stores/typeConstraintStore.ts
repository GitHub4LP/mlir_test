/**
 * 类型约束 Store
 * 
 * 从后端加载结构化约束规则，前端按需展开
 */

import { create } from 'zustand';
import type { ConstraintDef, ConstraintRule } from '../services/constraintResolver';
import { 
  getConstraintElements as resolveConcreteTypes,
  isShapedConstraint as checkIsShapedConstraint,
  getAllowedContainers as getContainers,
} from '../services/constraintResolver';

export type { ConstraintDef, ConstraintRule };

export interface TypeParameter {
  name: string;
  kind: 'type' | 'shape' | 'integer' | 'attribute';
}

export interface TypeDefinition {
  name: string;
  typeName: string;
  dialect: string;
  summary: string;
  parameters: TypeParameter[];
  isScalar: boolean;
}

interface TypeConstraintsResponse {
  buildableTypes: string[];
  constraintDefs: ConstraintDef[];
  typeDefinitions: TypeDefinition[];
  constraintEquivalences: Record<string, string[]>;  // 类型集合 → 等价约束名
}

interface TypeConstraintState {
  // 数据
  buildableTypes: string[];
  constraintDefs: Map<string, ConstraintDef>;
  typeDefinitions: TypeDefinition[];
  constraintEquivalences: Map<string, string[]>;  // 类型集合key → 等价约束名

  // 状态
  isLoading: boolean;
  isLoaded: boolean;
  error: string | null;

  // 操作
  loadTypeConstraints: () => Promise<void>;

  // 查询方法
  getConstraintElements: (constraint: string) => string[];
  isConcreteType: (type: string) => boolean;
  isShapedConstraint: (constraint: string) => boolean;
  getAllowedContainers: (constraint: string) => string[];
  getConstraintDef: (name: string) => ConstraintDef | undefined;
  getTypeDefinition: (typeName: string) => TypeDefinition | undefined;
  getAllConstraintNames: () => string[];
  getEquivalentConstraints: (types: string[]) => string[];  // 新增
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null;  // 新增
}

export const useTypeConstraintStore = create<TypeConstraintState>((set, get) => ({
  // 初始状态
  buildableTypes: [],
  constraintDefs: new Map(),
  typeDefinitions: [],
  constraintEquivalences: new Map(),
  isLoading: false,
  isLoaded: false,
  error: null,

  loadTypeConstraints: async () => {
    const state = get();
    if (state.isLoaded || state.isLoading) return;

    set({ isLoading: true, error: null });

    try {
      const response = await fetch('/api/types/');
      if (!response.ok) {
        throw new Error(`Failed to load type constraints: ${response.statusText}`);
      }

      const data: TypeConstraintsResponse = await response.json();
      
      // 构建 constraintDefs Map
      const defsMap = new Map<string, ConstraintDef>();
      for (const def of data.constraintDefs) {
        defsMap.set(def.name, def);
      }

      // 构建 constraintEquivalences Map
      const equivMap = new Map<string, string[]>();
      for (const [key, names] of Object.entries(data.constraintEquivalences || {})) {
        equivMap.set(key, names);
      }

      set({
        buildableTypes: data.buildableTypes,
        constraintDefs: defsMap,
        typeDefinitions: data.typeDefinitions || [],
        constraintEquivalences: equivMap,
        isLoading: false,
        isLoaded: true,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      console.error('Failed to load type constraints:', message);
      set({ isLoading: false, error: message });
    }
  },

  getConstraintElements: (constraint: string) => {
    const { constraintDefs, buildableTypes } = get();
    
    // 尝试规范化
    const normalized = normalizeType(constraint);
    
    // 使用 resolver 展开
    const types = resolveConcreteTypes(normalized, constraintDefs, buildableTypes);
    if (types.length > 0) return types;
    
    // 尝试原始名称
    if (normalized !== constraint) {
      const origTypes = resolveConcreteTypes(constraint, constraintDefs, buildableTypes);
      if (origTypes.length > 0) return origTypes;
    }
    
    // 未知约束，返回自身
    return [constraint];
  },

  isConcreteType: (type: string) => {
    const { buildableTypes } = get();
    const normalized = normalizeType(type);
    return buildableTypes.includes(type) || buildableTypes.includes(normalized);
  },

  isShapedConstraint: (constraint: string) => {
    const { constraintDefs } = get();
    return checkIsShapedConstraint(constraint, constraintDefs);
  },

  getAllowedContainers: (constraint: string) => {
    const { constraintDefs } = get();
    return getContainers(constraint, constraintDefs);
  },

  getConstraintDef: (name: string) => {
    const { constraintDefs } = get();
    return constraintDefs.get(name) || constraintDefs.get(normalizeType(name));
  },

  getTypeDefinition: (typeName: string) => {
    const { typeDefinitions } = get();
    return typeDefinitions.find(td => td.typeName === typeName || td.name === typeName);
  },

  getAllConstraintNames: () => {
    const { constraintDefs } = get();
    return [...constraintDefs.keys()].sort();
  },

  getEquivalentConstraints: (types: string[]) => {
    const { constraintEquivalences } = get();
    if (types.length === 0) return [];
    const key = [...types].sort().join(',');
    return constraintEquivalences.get(key) || [];
  },

  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => {
    const { buildableTypes, constraintEquivalences } = get();
    if (types.length === 0) return null;
    
    const key = [...types].sort().join(',');
    const equivalents = constraintEquivalences.get(key);
    if (!equivalents || equivalents.length === 0) return null;
    
    // 1. 用户 pin 优先
    if (pinnedName && equivalents.includes(pinnedName)) {
      return pinnedName;
    }
    
    // 2. BuildableType 优先（单一具体类型）
    for (const name of equivalents) {
      if (buildableTypes.includes(name)) {
        return name;
      }
    }
    
    // 3. 节点方言匹配
    if (nodeDialect) {
      const dialectPrefix = `${nodeDialect}_`;
      // 尝试精确前缀匹配
      const match = equivalents.find(n => n.startsWith(dialectPrefix));
      if (match) return match;
      // 尝试大写前缀匹配 (arith -> Arith_)
      const upperPrefix = nodeDialect.charAt(0).toUpperCase() + nodeDialect.slice(1) + '_';
      const upperMatch = equivalents.find(n => n.startsWith(upperPrefix));
      if (upperMatch) return upperMatch;
    }
    
    // 4. 无前缀的内置约束
    const builtin = equivalents.find(n => !n.includes('_'));
    if (builtin) return builtin;
    
    // 5. 兜底
    return equivalents[0];
  },
}));

/**
 * 规范化类型字符串（小写 → 大写形式）
 */
function normalizeType(type: string): string {
  // i32 -> I32
  const intMatch = type.match(/^i(\d+)$/);
  if (intMatch) return `I${intMatch[1]}`;

  // si32 -> SI32
  const sintMatch = type.match(/^si(\d+)$/);
  if (sintMatch) return `SI${sintMatch[1]}`;

  // ui32 -> UI32
  const uintMatch = type.match(/^ui(\d+)$/);
  if (uintMatch) return `UI${uintMatch[1]}`;

  // f32 -> F32
  const floatMatch = type.match(/^f(\d+)$/);
  if (floatMatch) return `F${floatMatch[1]}`;

  // bf16 -> BF16
  const bfloatMatch = type.match(/^bf(\d+)$/);
  if (bfloatMatch) return `BF${bfloatMatch[1]}`;

  // tf32 -> TF32
  const tfloatMatch = type.match(/^tf(\d+)$/);
  if (tfloatMatch) return `TF${tfloatMatch[1]}`;

  // index -> Index
  if (type === 'index') return 'Index';

  return type;
}
