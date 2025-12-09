/**
 * Type Constraint Store
 * 
 * 从后端动态加载类型约束数据。
 * 
 * 数据来源：
 * - buildableTypes: 内置具体类型（36 个）
 * - constraintMap: 内置约束映射
 * - dialectConstraints: 各方言特有约束（剔除内置重复的）
 * - typeDefinitions: 所有 TypeDef（标量 + 参数化）
 */

import { create } from 'zustand';

export interface TypeParameter {
  name: string;
  kind: 'type' | 'shape' | 'integer' | 'attribute';
}

export interface TypeDefinition {
  name: string;       // TableGen 名称
  typeName: string;   // MLIR 类型名
  dialect: string;    // 所属方言
  summary: string;    // 描述
  parameters: TypeParameter[];
  isScalar: boolean;  // 是否标量
}

interface TypeConstraintsData {
  buildableTypes: string[];
  constraintMap: Record<string, string[]>;
  dialectConstraints: Record<string, Record<string, string[]>>;
  typeDefinitions: TypeDefinition[];
  allConstraints: string[];
}

interface TypeConstraintState {
  // 数据
  buildableTypes: string[];
  constraintMap: Record<string, string[]>;
  dialectConstraints: Record<string, Record<string, string[]>>;
  typeDefinitions: TypeDefinition[];
  allConstraints: string[];  // 所有可选约束名（包括复合类型约束）
  
  // 状态
  isLoading: boolean;
  isLoaded: boolean;
  error: string | null;
  
  // 操作
  loadTypeConstraints: () => Promise<void>;
  
  // 查询方法
  getConcreteTypes: (constraint: string) => string[];
  isConcreteType: (type: string) => boolean;
  isAbstractConstraint: (constraint: string) => boolean;
  getDialectNames: () => string[];
  getTypeDefinition: (typeName: string) => TypeDefinition | undefined;
  getParameterizedTypes: () => TypeDefinition[];
  getScalarTypes: () => TypeDefinition[];
  getAllConstraints: () => string[];
  isConstraint: (name: string) => boolean;
}

export const useTypeConstraintStore = create<TypeConstraintState>((set, get) => ({
  // 初始状态
  buildableTypes: [],
  constraintMap: {},
  dialectConstraints: {},
  typeDefinitions: [],
  allConstraints: [],
  isLoading: false,
  isLoaded: false,
  error: null,
  
  loadTypeConstraints: async () => {
    const state = get();
    if (state.isLoaded || state.isLoading) {
      return;
    }
    
    set({ isLoading: true, error: null });
    
    try {
      const response = await fetch('/api/types/');
      if (!response.ok) {
        throw new Error(`Failed to load type constraints: ${response.statusText}`);
      }
      
      const data: TypeConstraintsData = await response.json();
      
      set({
        buildableTypes: data.buildableTypes,
        constraintMap: data.constraintMap,
        dialectConstraints: data.dialectConstraints || {},
        typeDefinitions: data.typeDefinitions || [],
        allConstraints: data.allConstraints || [],
        isLoading: false,
        isLoaded: true,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      console.error('Failed to load type constraints:', message);
      set({
        isLoading: false,
        error: message,
      });
    }
  },
  
  getConcreteTypes: (constraint: string) => {
    const { constraintMap, buildableTypes } = get();
    
    // 先查约束映射
    const types = constraintMap[constraint];
    if (types && types.length > 0) {
      return [...types];
    }
    
    // 尝试规范化（处理小写形式）
    const normalized = normalizeType(constraint);
    const normalizedTypes = constraintMap[normalized];
    if (normalizedTypes && normalizedTypes.length > 0) {
      return [...normalizedTypes];
    }
    
    // 如果是具体类型，返回自身
    if (buildableTypes.includes(constraint) || buildableTypes.includes(normalized)) {
      return [normalized];
    }
    
    // 未知约束，返回自身
    return [constraint];
  },
  
  isConcreteType: (type: string) => {
    const { buildableTypes } = get();
    const normalized = normalizeType(type);
    return buildableTypes.includes(type) || buildableTypes.includes(normalized);
  },
  
  isAbstractConstraint: (constraint: string) => {
    const { constraintMap, dialectConstraints } = get();
    // 检查内置约束
    const types = constraintMap[constraint];
    if (types !== undefined && types.length > 1) {
      return true;
    }
    // 检查方言约束
    for (const dialectMap of Object.values(dialectConstraints)) {
      const dialectTypes = dialectMap[constraint];
      if (dialectTypes !== undefined && dialectTypes.length > 1) {
        return true;
      }
    }
    return false;
  },
  
  getDialectNames: () => {
    const { dialectConstraints } = get();
    return Object.keys(dialectConstraints).sort();
  },
  
  getTypeDefinition: (typeName: string) => {
    const { typeDefinitions } = get();
    return typeDefinitions.find(td => td.typeName === typeName || td.name === typeName);
  },
  
  getParameterizedTypes: () => {
    const { typeDefinitions } = get();
    return typeDefinitions.filter(td => !td.isScalar);
  },
  
  getScalarTypes: () => {
    const { typeDefinitions } = get();
    return typeDefinitions.filter(td => td.isScalar);
  },
  
  getAllConstraints: () => {
    const { allConstraints } = get();
    return allConstraints;
  },
  
  isConstraint: (name: string) => {
    const { allConstraints } = get();
    return allConstraints.includes(name);
  },
}));

/**
 * 规范化类型字符串（小写 → 大写形式）
 */
function normalizeType(type: string): string {
  // i32 -> I32
  const intMatch = type.match(/^i(\d+)$/);
  if (intMatch) {
    return `I${intMatch[1]}`;
  }
  
  // si32 -> SI32
  const sintMatch = type.match(/^si(\d+)$/);
  if (sintMatch) {
    return `SI${sintMatch[1]}`;
  }
  
  // ui32 -> UI32
  const uintMatch = type.match(/^ui(\d+)$/);
  if (uintMatch) {
    return `UI${uintMatch[1]}`;
  }
  
  // f32 -> F32
  const floatMatch = type.match(/^f(\d+)$/);
  if (floatMatch) {
    return `F${floatMatch[1]}`;
  }
  
  // bf16 -> BF16
  const bfloatMatch = type.match(/^bf(\d+)$/);
  if (bfloatMatch) {
    return `BF${bfloatMatch[1]}`;
  }
  
  // tf32 -> TF32
  const tfloatMatch = type.match(/^tf(\d+)$/);
  if (tfloatMatch) {
    return `TF${tfloatMatch[1]}`;
  }
  
  // index -> Index
  if (type === 'index') {
    return 'Index';
  }
  
  return type;
}
