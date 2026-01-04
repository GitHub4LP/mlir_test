/**
 * 类型交集计算模块
 * 
 * 支持标量类型和容器类型的交集计算。
 * 
 * 三元素统一模型：
 * - 标量：{ container: null, shape: [], element: 自身 }
 * - 容器：{ container: 'tensor'|'vector'|..., shape: [...], element: 标量 }
 * - 约束：{ containers: [...], elements: [...], shapeConstraint: 'any'|'ranked'|'unranked' }
 * 
 * 类型字符串格式：
 * - 标量：I32, F32, Index
 * - 约束：SignlessIntegerLike, AnyType, AnyTensor
 * - 容器：tensor<4x4xF32>, vector<4xI32>, memref<?x?xF32>
 * 
 * 交集规则：
 * - 标量 vs 标量：使用约束展开后计算交集
 * - 容器 vs 容器：容器相同 + 形状兼容 + 元素递归
 * - 约束 vs 容器：检查约束的三元素是否允许该容器
 * - 约束 vs 约束：三元素交集
 */

import { typeConstraintStore } from '../stores';
import type { ConstraintDef, ConstraintDescriptor } from './constraintResolver';
import { expandRule, getAllowedContainers, getConstraintDescriptor } from './constraintResolver';

// ============================================================================
// 类型节点定义
// ============================================================================

/** 标量类型节点 */
export interface ScalarNode {
  kind: 'scalar';
  name: string;  // 具体类型名或约束名
}

/** 容器类型节点 */
export interface CompositeNode {
  kind: 'composite';
  container: string;  // tensor, vector, memref, complex
  shape?: (number | null)[];  // null 表示动态维度 (?)
  element: TypeNode;
}

export type TypeNode = ScalarNode | CompositeNode;

// ============================================================================
// 类型解析
// ============================================================================

/**
 * 解析类型字符串为 TypeNode
 * 
 * 支持格式：
 * - I32, F32 (标量)
 * - SignlessIntegerLike (约束)
 * - tensor<4x4xF32> (容器)
 * - tensor<?x?xF32> (动态形状)
 * - memref<4x4xvector<4xF32>> (嵌套容器)
 */
export function parseType(typeStr: string): TypeNode {
  const trimmed = typeStr.trim();
  
  // 尝试解析容器类型
  const containerMatch = trimmed.match(/^(tensor|vector|memref|complex)<(.+)>$/);
  if (containerMatch) {
    const [, container, inner] = containerMatch;
    return parseContainerType(container, inner);
  }
  
  // 标量或约束
  return { kind: 'scalar', name: trimmed };
}

/**
 * 解析容器类型的内部结构
 */
function parseContainerType(container: string, inner: string): CompositeNode {
  // complex 没有形状，只有元素类型
  if (container === 'complex') {
    return {
      kind: 'composite',
      container,
      element: parseType(inner),
    };
  }
  
  // 解析形状和元素类型：4x4xF32 或 ?x?xF32
  const parts = splitShapeAndElement(inner);
  
  return {
    kind: 'composite',
    container,
    shape: parts.shape,
    element: parseType(parts.element),
  };
}

/**
 * 分离形状和元素类型
 * 
 * 输入：4x4xF32 或 ?x?xvector<4xF32>
 * 输出：{ shape: [4, 4], element: 'F32' }
 */
function splitShapeAndElement(inner: string): { shape: (number | null)[]; element: string } {
  const shape: (number | null)[] = [];
  let remaining = inner;
  
  // 逐个解析维度
  while (true) {
    // 检查是否是维度（数字或 ?）
    const dimMatch = remaining.match(/^(\d+|\?)x/);
    if (!dimMatch) break;
    
    const dim = dimMatch[1];
    shape.push(dim === '?' ? null : parseInt(dim, 10));
    remaining = remaining.slice(dimMatch[0].length);
  }
  
  return { shape, element: remaining };
}

// ============================================================================
// 类型序列化
// ============================================================================

/**
 * 将 TypeNode 序列化为类型字符串
 */
export function serializeType(node: TypeNode): string {
  if (node.kind === 'scalar') {
    return node.name;
  }
  
  const { container, shape, element } = node;
  const elementStr = serializeType(element);
  
  if (container === 'complex') {
    return `complex<${elementStr}>`;
  }
  
  if (shape && shape.length > 0) {
    const shapeStr = shape.map(d => d === null ? '?' : d.toString()).join('x');
    return `${container}<${shapeStr}x${elementStr}>`;
  }
  
  return `${container}<${elementStr}>`;
}

// ============================================================================
// 交集计算
// ============================================================================

/**
 * 计算两个类型的交集
 * 
 * @returns 交集类型，如果不兼容返回 null
 */
export function computeTypeIntersection(type1: string, type2: string): string | null {
  const node1 = parseType(type1);
  const node2 = parseType(type2);
  
  const result = intersectNodes(node1, node2);
  return result ? serializeType(result) : null;
}

/**
 * 检查两个类型是否兼容（交集非空）
 */
export function areTypesCompatible(type1: string, type2: string): boolean {
  return computeTypeIntersection(type1, type2) !== null;
}

/**
 * 递归计算两个 TypeNode 的交集
 */
function intersectNodes(n1: TypeNode, n2: TypeNode): TypeNode | null {
  // 都是标量/约束
  if (n1.kind === 'scalar' && n2.kind === 'scalar') {
    return intersectScalars(n1, n2);
  }
  
  // 都是容器
  if (n1.kind === 'composite' && n2.kind === 'composite') {
    return intersectComposites(n1, n2);
  }
  
  // 一个标量/约束，一个容器
  if (n1.kind === 'scalar') {
    return intersectScalarWithComposite(n1, n2 as CompositeNode);
  }
  return intersectScalarWithComposite(n2 as ScalarNode, n1 as CompositeNode);
}

/**
 * 计算两个标量/约束的交集
 */
function intersectScalars(n1: ScalarNode, n2: ScalarNode): ScalarNode | null {
  const { constraintDefs, buildableTypes } = typeConstraintStore.getState();
  
  // 快速路径：完全相同
  if (n1.name === n2.name) {
    return n1;
  }
  
  // 展开约束
  const types1 = expandScalarConstraint(n1.name, constraintDefs, buildableTypes);
  const types2 = expandScalarConstraint(n2.name, constraintDefs, buildableTypes);
  
  // 计算交集
  const set2 = new Set(types2);
  const intersection = types1.filter(t => set2.has(t));
  
  if (intersection.length === 0) {
    return null;
  }
  
  // 选择更具体的名称
  const name = pickMoreSpecificName(intersection, n1.name, n2.name, buildableTypes);
  return { kind: 'scalar', name };
}

/**
 * 展开标量约束为具体类型列表
 */
function expandScalarConstraint(
  name: string,
  defs: Map<string, ConstraintDef>,
  buildableTypes: string[]
): string[] {
  // 如果是 BuildableType，直接返回
  if (buildableTypes.includes(name)) {
    return [name];
  }
  
  // 尝试展开约束
  const def = defs.get(name);
  if (def?.rule) {
    const expanded = expandRule(def.rule, defs, buildableTypes);
    if (expanded.length > 0) {
      return expanded;
    }
  }
  
  // 无法展开，返回自身
  return [name];
}

/**
 * 选择更具体的约束名
 */
function pickMoreSpecificName(
  intersection: string[],
  name1: string,
  name2: string,
  buildableTypes: string[]
): string {
  // 如果交集只有一个类型，使用该类型
  if (intersection.length === 1) {
    return intersection[0];
  }
  
  // 优先使用 BuildableType
  if (buildableTypes.includes(name1)) return name1;
  if (buildableTypes.includes(name2)) return name2;
  
  // 使用第一个交集类型
  return intersection[0];
}

/**
 * 计算两个容器类型的交集
 */
function intersectComposites(n1: CompositeNode, n2: CompositeNode): CompositeNode | null {
  // 容器必须相同
  if (n1.container !== n2.container) {
    return null;
  }
  
  // 形状交集
  const shape = intersectShapes(n1.shape, n2.shape);
  if (shape === false) {
    return null;
  }
  
  // 元素递归
  const element = intersectNodes(n1.element, n2.element);
  if (!element) {
    return null;
  }
  
  return {
    kind: 'composite',
    container: n1.container,
    shape: shape === undefined ? undefined : shape,
    element,
  };
}

/**
 * 计算形状交集
 * 
 * @returns 交集形状，undefined 表示无形状，false 表示不兼容
 */
function intersectShapes(
  s1: (number | null)[] | undefined,
  s2: (number | null)[] | undefined
): (number | null)[] | undefined | false {
  // 都没有形状
  if (!s1 && !s2) return undefined;
  
  // 一个有形状一个没有
  if (!s1) return s2;
  if (!s2) return s1;
  
  // 维度数必须相同
  if (s1.length !== s2.length) return false;
  
  // 逐维度取交集
  const result: (number | null)[] = [];
  for (let i = 0; i < s1.length; i++) {
    const d1 = s1[i];
    const d2 = s2[i];
    
    if (d1 === null) {
      result.push(d2);  // 动态匹配具体
    } else if (d2 === null) {
      result.push(d1);  // 动态匹配具体
    } else if (d1 === d2) {
      result.push(d1);  // 相同
    } else {
      return false;  // 不兼容
    }
  }
  
  return result;
}

/**
 * 计算标量/约束与容器的交集（使用三元素模型）
 * 
 * 约束的三元素：{ containers, elements, shapeConstraint }
 * 容器的三元素：{ container, shape, element }
 * 
 * 兼容条件：
 * 1. 约束的 containers 包含容器类型
 * 2. 约束的 elements 与容器的 element 有交集
 * 3. 形状约束兼容
 */
function intersectScalarWithComposite(
  scalar: ScalarNode,
  composite: CompositeNode
): TypeNode | null {
  const { constraintDefs, buildableTypes } = typeConstraintStore.getState();
  
  // 获取约束的三元素描述符
  const descriptor = getConstraintDescriptor(scalar.name, constraintDefs, buildableTypes);
  
  // 1. 检查约束是否允许该容器类型
  if (!descriptor.containers.includes(composite.container)) {
    return null;
  }
  
  // 2. 检查形状约束（简化：ranked/unranked 检查）
  if (descriptor.shapeConstraint === 'ranked' && (!composite.shape || composite.shape.length === 0)) {
    return null;
  }
  if (descriptor.shapeConstraint === 'unranked' && composite.shape && composite.shape.length > 0) {
    return null;
  }
  
  // 3. 检查元素类型兼容性
  const elementNode = composite.element;
  if (elementNode.kind === 'scalar') {
    // 展开容器的元素类型
    const expandedElement = expandScalarConstraint(elementNode.name, constraintDefs, buildableTypes);
    
    // 与约束的 elements 求交集
    const elemSet = new Set(descriptor.elements);
    const intersection = expandedElement.filter(t => elemSet.has(t));
    
    if (intersection.length === 0) {
      return null;
    }
    
    // 返回收窄后的容器
    return {
      ...composite,
      element: {
        kind: 'scalar',
        name: pickMoreSpecificName(intersection, elementNode.name, '', buildableTypes),
      },
    };
  }
  
  // 嵌套容器：递归检查
  if (elementNode.kind === 'composite') {
    // 构造元素约束的描述符（只保留 elements 部分）
    const elementDescriptor: ConstraintDescriptor = {
      containers: [null, ...getAllowedContainers(scalar.name, constraintDefs)],
      elements: descriptor.elements,
      shapeConstraint: 'any',
    };
    
    // 检查嵌套容器是否与元素约束兼容
    if (!elementDescriptor.containers.includes(elementNode.container)) {
      return null;
    }
    
    // 递归检查嵌套容器的元素
    const nestedResult = intersectNodes(
      { kind: 'scalar', name: descriptor.elements[0] || '' },
      elementNode.element
    );
    if (!nestedResult) {
      return null;
    }
    
    return {
      ...composite,
      element: {
        ...elementNode,
        element: nestedResult,
      },
    };
  }
  
  return composite;
}

// ============================================================================
// 导出便捷函数
// ============================================================================

/**
 * 计算类型交集的元素数量（用于连接验证）
 * 
 * 兼容现有 API，返回交集大小
 */
export function getTypeIntersectionCountV2(type1: string, type2: string): number {
  const result = computeTypeIntersection(type1, type2);
  if (!result) return 0;
  
  // 解析结果，估算交集大小
  const node = parseType(result);
  if (node.kind === 'scalar') {
    const { constraintDefs, buildableTypes } = typeConstraintStore.getState();
    const types = expandScalarConstraint(node.name, constraintDefs, buildableTypes);
    return types.length > 0 ? types.length : 1;
  }
  
  // 容器类型，返回 1 表示兼容
  return 1;
}
