/**
 * 类型约束 Store
 * 
 * 从后端加载结构化约束规则，前端按需展开
 */

import { create } from 'zustand';
import type { ConstraintDef, ConstraintRule, ConstraintDescriptor } from '../services/constraintResolver';
import { 
  getConstraintElements,
  isShapedConstraint as checkIsShapedConstraint,
  getAllowedContainers as getContainers,
  getConstraintDescriptor as getDescriptor,
} from '../services/constraintResolver';
import { setConstraintResolver } from './typeColorCache';
import { apiGet } from '../services/apiClient';

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

export interface ContainerType {
  name: string;
  hasShape: boolean;
  elementConstraint: string | null;
}

interface TypeConstraintsResponse {
  buildableTypes: string[];
  constraintDefs: ConstraintDef[];
  typeDefinitions: TypeDefinition[];
  constraintEquivalences: Record<string, string[]>;  // 类型集合 → 等价约束名
  containerTypes: ContainerType[];  // 容器类型及其元素约束
}

export interface TypeConstraintState {
  // 数据
  buildableTypes: string[];
  constraintDefs: Map<string, ConstraintDef>;
  typeDefinitions: TypeDefinition[];
  constraintEquivalences: Map<string, string[]>;  // 类型集合key → 等价约束名
  constraintsByDialect: Map<string, string[]>;    // 方言 → 约束名列表
  containerTypes: ContainerType[];                // 容器类型列表

  // 状态
  isLoading: boolean;
  isLoaded: boolean;
  error: string | null;

  // 操作
  loadTypeConstraints: () => Promise<void>;
  registerDialectConstraints: (dialect: string, constraints: ConstraintDef[]) => void;  // 新增

  // 查询方法
  getConstraintElements: (constraint: string) => string[];
  isShapedConstraint: (constraint: string) => boolean;
  getAllowedContainers: (constraint: string) => string[];
  getConstraintDef: (name: string) => ConstraintDef | undefined;
  getTypeDefinition: (typeName: string) => TypeDefinition | undefined;
  getAllConstraintNames: () => string[];
  getEquivalentConstraints: (types: string[]) => string[];
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null;
  findSubsetConstraints: (E: string[]) => string[];  // 找出所有元素集合是 E 子集的具名约束
  getConstraintDescriptor: (constraint: string) => ConstraintDescriptor;  // 获取约束的三元素描述符
  
  // 容器类型方法
  getContainerType: (name: string) => ContainerType | undefined;
  getElementConstraintName: (container: string) => string | null;
  
  // 方言过滤方法
  getBuiltinConstraints: () => string[];
  getConstraintsByDialect: (dialect: string) => string[];
  filterConstraintsByDialects: (constraints: string[], dialects: string[]) => string[];
}

export const useTypeConstraintStore = create<TypeConstraintState>((set, get) => ({
  // 初始状态
  buildableTypes: [],
  constraintDefs: new Map(),
  typeDefinitions: [],
  constraintEquivalences: new Map(),
  constraintsByDialect: new Map(),
  containerTypes: [],
  isLoading: false,
  isLoaded: false,
  error: null,

  loadTypeConstraints: async () => {
    const state = get();
    if (state.isLoaded || state.isLoading) return;

    set({ isLoading: true, error: null });

    try {
      const data = await apiGet<TypeConstraintsResponse>('/types/');
      
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

      // 构建 constraintsByDialect 索引
      const byDialectMap = new Map<string, string[]>();
      // 特殊键 '__builtin__' 存储内置约束（dialect 为 undefined 或 null）
      byDialectMap.set('__builtin__', []);
      for (const def of data.constraintDefs) {
        const dialectKey = def.dialect ?? '__builtin__';
        if (!byDialectMap.has(dialectKey)) {
          byDialectMap.set(dialectKey, []);
        }
        byDialectMap.get(dialectKey)!.push(def.name);
      }

      set({
        buildableTypes: data.buildableTypes,
        constraintDefs: defsMap,
        typeDefinitions: data.typeDefinitions || [],
        constraintEquivalences: equivMap,
        constraintsByDialect: byDialectMap,
        containerTypes: data.containerTypes || [],
        isLoading: false,
        isLoaded: true,
      });

      // 注入约束解析器到颜色缓存，并清空旧缓存
      setConstraintResolver(get().getConstraintElements);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      console.error('Failed to load type constraints:', message);
      set({ isLoading: false, error: message });
    }
  },

  registerDialectConstraints: (dialect: string, constraints: ConstraintDef[]) => {
    if (constraints.length === 0) return;
    
    const state = get();
    
    // 1. 添加到 constraintDefs（跳过已存在的）
    const newDefs = new Map(state.constraintDefs);
    const addedNames: string[] = [];
    for (const def of constraints) {
      if (!newDefs.has(def.name)) {
        newDefs.set(def.name, def);
        addedNames.push(def.name);
      }
    }
    
    if (addedNames.length === 0) {
      // 没有新约束，无需更新
      return;
    }
    
    // 2. 更新 constraintsByDialect 索引
    const newByDialect = new Map(state.constraintsByDialect);
    newByDialect.set(dialect, addedNames);
    
    // 3. 增量更新 constraintEquivalences
    const newEquivalences = new Map(state.constraintEquivalences);
    const buildableSet = new Set(state.buildableTypes);
    
    for (const name of addedNames) {
      const def = newDefs.get(name);
      if (!def) continue;
      
      // 展开约束为类型集合
      const types = getConstraintElements(name, newDefs, state.buildableTypes);
      
      // 只处理标量约束（所有类型都是 buildableTypes）
      if (types.length > 0 && types.every(t => buildableSet.has(t))) {
        const key = [...types].sort().join(',');
        const existing = newEquivalences.get(key) || [];
        if (!existing.includes(name)) {
          newEquivalences.set(key, [...existing, name]);
        }
      }
    }
    
    set({
      constraintDefs: newDefs,
      constraintsByDialect: newByDialect,
      constraintEquivalences: newEquivalences,
    });
    
    // 更新颜色缓存的约束解析器
    setConstraintResolver(get().getConstraintElements);
  },

  getConstraintElements: (constraint: string) => {
    const { constraintDefs, buildableTypes } = get();
    
    // 尝试规范化
    const normalized = normalizeType(constraint);
    
    // 使用 resolver 展开
    const types = getConstraintElements(normalized, constraintDefs, buildableTypes);
    if (types.length > 0) return types;
    
    // 尝试原始名称
    if (normalized !== constraint) {
      const origTypes = getConstraintElements(constraint, constraintDefs, buildableTypes);
      if (origTypes.length > 0) return origTypes;
    }
    
    // 未知约束，返回自身
    return [constraint];
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
    
    // 4. 优先选择通用约束名（AnyType 等）
    const preferredNames = ['AnyType', 'AnyInteger', 'AnyFloat', 'AnySignlessInteger', 'AnySignedInteger', 'AnyUnsignedInteger'];
    for (const name of preferredNames) {
      if (equivalents.includes(name)) {
        return name;
      }
    }
    
    // 5. 无前缀的内置约束
    const builtin = equivalents.find(n => !n.includes('_'));
    if (builtin) return builtin;
    
    // 6. 兜底
    return equivalents[0];
  },

  findSubsetConstraints: (E: string[]) => {
    const { constraintEquivalences } = get();
    
    if (E.length === 0) return [];
    
    const ESet = new Set(E);
    const result: string[] = [];
    
    // 遍历 constraintEquivalences，找出所有元素集合是 E 子集的约束名
    // key 是排序后的类型集合字符串（如 'I1,I128,I16,I32,I64,I8'）
    // value 是等价的约束名列表
    for (const [key, names] of constraintEquivalences) {
      // 解析 key 为类型数组
      const types = key.split(',');
      // 检查是否是 E 的非空子集
      if (types.length > 0 && types.every(t => ESet.has(t))) {
        result.push(...names);
      }
    }
    
    return result;
  },

  getConstraintDescriptor: (constraint: string) => {
    const { constraintDefs, buildableTypes } = get();
    return getDescriptor(constraint, constraintDefs, buildableTypes);
  },

  // ============ 容器类型方法 ============

  getContainerType: (name: string) => {
    const { containerTypes } = get();
    return containerTypes.find(c => c.name === name);
  },

  getElementConstraintName: (container: string) => {
    const { containerTypes } = get();
    const ct = containerTypes.find(c => c.name === container);
    return ct?.elementConstraint ?? null;
  },

  // ============ 方言过滤方法 ============

  getBuiltinConstraints: () => {
    const { constraintsByDialect } = get();
    return constraintsByDialect.get('__builtin__') || [];
  },

  getConstraintsByDialect: (dialect: string) => {
    const { constraintsByDialect } = get();
    return constraintsByDialect.get(dialect) || [];
  },

  filterConstraintsByDialects: (constraints: string[], dialects: string[]) => {
    const { constraintDefs } = get();
    
    // 构建允许的方言集合
    const allowedDialects = new Set(dialects);
    
    return constraints.filter(name => {
      const def = constraintDefs.get(name);
      if (!def) return false;
      // dialect 为 undefined 或 null 表示内置约束，始终允许
      if (def.dialect === undefined || def.dialect === null) return true;
      // 检查是否在允许的方言列表中
      return allowedDialects.has(def.dialect);
    });
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
