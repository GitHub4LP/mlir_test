/**
 * 元素类型提取模块
 * 
 * 用于支持 SameOperandsElementType 和 SameOperandsAndResultElementType traits。
 * 这些 traits 要求端口的元素类型相同，而非完整类型相同。
 * 
 * 元素类型规则：
 * - 标量类型：元素类型是自身（I32 → I32）
 * - 容器类型：元素类型是内部类型（tensor<4x4xF32> → F32）
 * - 嵌套容器：递归提取最内层元素（tensor<4xvector<4xF32>> → F32）
 */

import { parseType, serializeType, type TypeNode } from '../typeIntersection';

/**
 * 容器类型列表
 */
const CONTAINER_TYPES = ['tensor', 'vector', 'memref', 'complex'] as const;

/**
 * 判断类型字符串是否为容器类型
 * 
 * @param typeStr 类型字符串
 * @returns 是否为容器类型
 */
export function isContainerType(typeStr: string): boolean {
  const trimmed = typeStr.trim();
  return CONTAINER_TYPES.some(container => trimmed.startsWith(`${container}<`));
}

/**
 * 从类型中提取元素类型
 * 
 * - 标量类型返回自身
 * - 容器类型返回内部元素类型（递归到最内层）
 * 
 * @param typeStr 类型字符串
 * @returns 元素类型字符串
 */
export function extractElementType(typeStr: string): string {
  const node = parseType(typeStr);
  const elementNode = extractElementNode(node);
  return serializeType(elementNode);
}

/**
 * 递归提取最内层元素类型节点
 */
function extractElementNode(node: TypeNode): TypeNode {
  if (node.kind === 'scalar') {
    return node;
  }
  
  // 容器类型：递归提取元素
  return extractElementNode(node.element);
}

/**
 * 获取类型的容器结构（不含元素类型）
 * 
 * 用于元素类型传播时保持容器结构不变
 * 
 * @param typeStr 类型字符串
 * @returns 容器结构描述，标量返回 null
 */
export function getContainerStructure(typeStr: string): ContainerStructure | null {
  const node = parseType(typeStr);
  return extractContainerStructure(node);
}

/**
 * 容器结构描述
 */
export interface ContainerStructure {
  container: string;
  shape?: (number | null)[];
  nested?: ContainerStructure;
}

/**
 * 递归提取容器结构
 */
function extractContainerStructure(node: TypeNode): ContainerStructure | null {
  if (node.kind === 'scalar') {
    return null;
  }
  
  const structure: ContainerStructure = {
    container: node.container,
  };
  
  if (node.shape) {
    structure.shape = node.shape;
  }
  
  // 检查是否有嵌套容器
  if (node.element.kind === 'composite') {
    structure.nested = extractContainerStructure(node.element) ?? undefined;
  }
  
  return structure;
}

/**
 * 将元素类型应用到容器结构
 * 
 * @param structure 容器结构
 * @param elementType 元素类型字符串
 * @returns 完整类型字符串
 */
export function applyElementToStructure(
  structure: ContainerStructure | null,
  elementType: string
): string {
  if (!structure) {
    return elementType;
  }
  
  // 递归构建嵌套容器
  const innerType = structure.nested
    ? applyElementToStructure(structure.nested, elementType)
    : elementType;
  
  // 构建当前层容器
  if (structure.container === 'complex') {
    return `complex<${innerType}>`;
  }
  
  if (structure.shape && structure.shape.length > 0) {
    const shapeStr = structure.shape.map(d => d === null ? '?' : d.toString()).join('x');
    return `${structure.container}<${shapeStr}x${innerType}>`;
  }
  
  return `${structure.container}<${innerType}>`;
}
