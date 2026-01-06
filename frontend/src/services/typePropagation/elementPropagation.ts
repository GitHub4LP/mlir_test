/**
 * 元素类型传播算法
 * 
 * 扩展基础传播算法，支持元素类型边。
 * 
 * 元素类型传播规则：
 * - 通过 'element' 边传播时，只传播元素类型部分
 * - 标量类型的元素类型是自身
 * - 容器类型的元素类型是内部类型
 * 
 * 混合传播场景：
 * - tensor<4x4xF32> 通过元素边传播 → F32
 * - F32 通过元素边传播到 tensor<?x?xI32> → 检查 F32 是否与 I32 兼容
 */

import type { ExtendedPropagationGraph, PropagationResult, VariableId, TypeSource } from './types';
import { extractElementType, isContainerType, getContainerStructure, applyElementToStructure } from './elementType';
import { computeTypeIntersection } from '../typeIntersection';
import { typeConstraintStore } from '../../stores';

/**
 * 检查约束是否是容器约束（如 AnyTensor, AnyMemRef）
 */
function isContainerConstraint(constraint: string): boolean {
  if (isContainerType(constraint)) return false;
  const { isShapedConstraint } = typeConstraintStore.getState();
  return isShapedConstraint(constraint);
}

/**
 * 展开标量约束为具体类型列表
 */
function expandScalarConstraint(name: string): string[] {
  const { buildableTypes, getConstraintElements } = typeConstraintStore.getState();
  
  if (buildableTypes.includes(name)) {
    return [name];
  }
  
  const elements = getConstraintElements(name);
  return elements.length > 0 ? elements : [name];
}

/**
 * 计算两个类型集合的交集（支持容器类型）
 */
function computeSetIntersection(set1: string[], set2: string[]): string[] {
  if (set1.length === 0 || set2.length === 0) return [];
  
  const result: string[] = [];
  const seen = new Set<string>();
  
  for (const t1 of set1) {
    for (const t2 of set2) {
      if (t1 === t2) {
        if (!seen.has(t1)) {
          seen.add(t1);
          result.push(t1);
        }
        continue;
      }
      
      const t1IsContainer = isContainerType(t1);
      const t2IsContainer = isContainerType(t2);
      const t1IsContainerConstraint = !t1IsContainer && isContainerConstraint(t1);
      const t2IsContainerConstraint = !t2IsContainer && isContainerConstraint(t2);
      
      if (t1IsContainer || t2IsContainer || t1IsContainerConstraint || t2IsContainerConstraint) {
        const intersection = computeTypeIntersection(t1, t2);
        if (intersection && !seen.has(intersection)) {
          seen.add(intersection);
          result.push(intersection);
        }
      }
    }
  }
  
  return result;
}

/**
 * 提取类型集合的元素类型
 * 
 * 对于每个类型，提取其元素类型
 */
function extractElementTypes(types: string[]): string[] {
  const result: string[] = [];
  const seen = new Set<string>();
  
  for (const type of types) {
    const elementType = extractElementType(type);
    if (!seen.has(elementType)) {
      seen.add(elementType);
      result.push(elementType);
    }
  }
  
  return result;
}

/**
 * 使用扩展传播图进行类型传播（支持元素类型边）
 * 
 * 算法：
 * 1. 初始化每个端口的有效集合
 * 2. 应用类型源
 * 3. BFS 传播，区分完整类型边和元素类型边
 * 
 * @param graph - 扩展传播图（包含边类型）
 * @param sources - 类型源
 * @param portConstraints - 端口原始约束
 */
export function propagateTypesWithElementEdges(
  graph: ExtendedPropagationGraph,
  sources: TypeSource[],
  portConstraints: Map<VariableId, string>
): PropagationResult {
  const effectiveSets = new Map<VariableId, string[]>();
  const sourceMap = new Map<VariableId, VariableId | null>();
  const queue: VariableId[] = [];
  const inQueue = new Set<VariableId>();

  // 1. 初始化有效集合
  for (const [varId, constraint] of portConstraints) {
    if (isContainerConstraint(constraint)) {
      const elements = expandScalarConstraint(constraint);
      effectiveSets.set(varId, [constraint, ...elements]);
    } else {
      effectiveSets.set(varId, expandScalarConstraint(constraint));
    }
  }

  // 2. 应用类型源
  for (const source of sources) {
    const varId = source.portRef.key;
    const sourceType = source.type;
    
    if (isContainerType(sourceType)) {
      effectiveSets.set(varId, [sourceType]);
    } else {
      const sourceElements = expandScalarConstraint(sourceType);
      const originalElements = effectiveSets.get(varId) || [];
      const intersection = computeSetIntersection(sourceElements, originalElements);
      
      if (intersection.length > 0) {
        effectiveSets.set(varId, intersection);
      } else {
        effectiveSets.set(varId, sourceElements);
      }
    }
    
    sourceMap.set(varId, null);
  }

  // 3. 将所有有邻居的端口加入队列
  for (const varId of graph.keys()) {
    if (!inQueue.has(varId)) {
      queue.push(varId);
      inQueue.add(varId);
    }
  }

  // 4. BFS 传播
  while (queue.length > 0) {
    const varId = queue.shift()!;
    inQueue.delete(varId);
    
    const currentSet = effectiveSets.get(varId) || [];
    if (currentSet.length === 0) continue;

    const edges = graph.get(varId);
    if (!edges) continue;

    for (const edge of edges) {
      const neighborId = edge.target;
      const neighborSet = effectiveSets.get(neighborId) || [];
      if (neighborSet.length === 0) continue;

      let propagatedSet: string[];
      
      if (edge.kind === 'element') {
        // 元素类型传播：提取当前类型的元素类型，与邻居求交集
        const currentElementTypes = extractElementTypes(currentSet);
        const neighborElementTypes = extractElementTypes(neighborSet);
        
        // 计算元素类型交集
        const elementIntersection = computeSetIntersection(currentElementTypes, neighborElementTypes);
        
        if (elementIntersection.length === 0) continue;
        
        // 将元素类型应用回邻居的容器结构
        propagatedSet = [];
        for (const neighborType of neighborSet) {
          const structure = getContainerStructure(neighborType);
          for (const elemType of elementIntersection) {
            const newType = applyElementToStructure(structure, elemType);
            if (!propagatedSet.includes(newType)) {
              propagatedSet.push(newType);
            }
          }
        }
      } else {
        // 完整类型传播
        propagatedSet = computeSetIntersection(neighborSet, currentSet);
      }
      
      // 如果交集发生变化，更新并重新入队
      if (propagatedSet.length > 0 && propagatedSet.length < neighborSet.length) {
        effectiveSets.set(neighborId, propagatedSet);
        sourceMap.set(neighborId, varId);
        
        if (!inQueue.has(neighborId)) {
          queue.push(neighborId);
          inQueue.add(neighborId);
        }
      }
    }
  }

  return { effectiveSets, sources: sourceMap };
}
