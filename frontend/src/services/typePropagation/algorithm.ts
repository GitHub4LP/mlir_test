/**
 * 类型传播算法
 * 
 * BFS + 交集计算的传播算法。
 * 
 * 支持两种类型：
 * 1. 标量约束：展开为标量类型列表，使用集合交集
 * 2. 容器约束/类型：保留约束名或具体类型，使用 computeTypeIntersection
 * 
 * 三元素模型：
 * - 标量：{ container: null, elements: [自身] }
 * - 容器类型：tensor<4x4xF32> 作为整体
 * - 容器约束：AnyTensor 保留约束名，交集时使用三元素匹配
 */

import type { PropagationGraph, PropagationResult, VariableId, TypeSource } from './types';
import { computeTypeIntersection, parseType } from '../typeIntersection';
import { typeConstraintStore } from '../../stores';

/**
 * 检查类型字符串是否是容器类型（具体类型如 tensor<4x4xF32>）
 */
function isContainerType(type: string): boolean {
  const node = parseType(type);
  return node.kind === 'composite';
}

/**
 * 检查约束是否是容器约束（如 AnyTensor, AnyMemRef）
 * 
 * 容器约束的特征：允许容器类型，但不是具体的容器类型字符串
 */
function isContainerConstraint(constraint: string): boolean {
  // 如果是具体容器类型，不是约束
  if (isContainerType(constraint)) return false;
  
  // 检查约束是否允许容器
  const { isShapedConstraint } = typeConstraintStore.getState();
  return isShapedConstraint(constraint);
}

/**
 * 计算两个类型集合的交集
 * 
 * 支持标量、容器类型、容器约束混合的情况：
 * - 标量 vs 标量：相等则保留
 * - 容器类型 vs 容器类型：使用 computeTypeIntersection
 * - 容器约束 vs 容器类型：使用 computeTypeIntersection（约束作为标量节点处理）
 * - 容器约束 vs 容器约束：使用 computeTypeIntersection
 */
function computeSetIntersection(set1: string[], set2: string[]): string[] {
  if (set1.length === 0 || set2.length === 0) return [];
  
  const result: string[] = [];
  const seen = new Set<string>();
  
  for (const t1 of set1) {
    for (const t2 of set2) {
      // 快速路径：完全相同
      if (t1 === t2) {
        if (!seen.has(t1)) {
          seen.add(t1);
          result.push(t1);
        }
        continue;
      }
      
      // 检查是否涉及容器类型或容器约束
      const t1IsContainer = isContainerType(t1);
      const t2IsContainer = isContainerType(t2);
      const t1IsContainerConstraint = !t1IsContainer && isContainerConstraint(t1);
      const t2IsContainerConstraint = !t2IsContainer && isContainerConstraint(t2);
      
      // 如果任一是容器类型或容器约束，使用 computeTypeIntersection
      if (t1IsContainer || t2IsContainer || t1IsContainerConstraint || t2IsContainerConstraint) {
        const intersection = computeTypeIntersection(t1, t2);
        if (intersection && !seen.has(intersection)) {
          seen.add(intersection);
          result.push(intersection);
        }
      }
      // 都是标量，已经在 t1 === t2 检查过了
    }
  }
  
  return result;
}

/**
 * 从类型源传播类型（BFS + 交集计算）
 * 
 * 算法：
 * 1. 初始化：每个端口的有效集合 = 原始约束的元素集合
 * 2. 将所有源端口加入队列，设置其有效集合为源类型的元素集合
 * 3. BFS 传播：对于每个端口，与所有邻居的有效集合求交集
 * 4. 如果交集发生变化，将该端口重新加入队列
 * 
 * @param graph - 传播图
 * @param sources - 类型源（用户选择的类型）
 * @param portConstraints - 每个端口的原始约束
 * @param getConstraintElements - 获取约束映射到的类型集合元素
 * @returns 传播结果
 */
export function propagateTypes(
  graph: PropagationGraph,
  sources: TypeSource[],
  portConstraints: Map<VariableId, string>,
  getConstraintElements: (constraint: string) => string[]
): PropagationResult {
  const effectiveSets = new Map<VariableId, string[]>();
  const sourceMap = new Map<VariableId, VariableId | null>();
  const queue: VariableId[] = [];
  const inQueue = new Set<VariableId>();

  // 1. 初始化：每个端口的有效集合
  // - 标量约束：展开为标量类型列表
  // - 容器约束：保留约束名（如 AnyTensor），让 computeTypeIntersection 处理
  for (const [varId, constraint] of portConstraints) {
    if (isContainerConstraint(constraint)) {
      // 容器约束：保留约束名，同时包含展开的标量类型（用于标量传播）
      const elements = getConstraintElements(constraint);
      effectiveSets.set(varId, [constraint, ...elements]);
    } else {
      // 标量约束或具体类型：展开为元素集合
      effectiveSets.set(varId, getConstraintElements(constraint));
    }
  }

  // 2. 应用类型源（用户 pin + 单一元素约束）
  for (const source of sources) {
    const varId = source.portRef.key;
    const sourceType = source.type;
    
    // 检查源类型是否是容器类型
    if (isContainerType(sourceType)) {
      // 容器类型：直接使用该类型作为有效集合
      effectiveSets.set(varId, [sourceType]);
    } else {
      // 标量类型：展开为元素集合
      const sourceElements = getConstraintElements(sourceType);
      
      // 源的有效集合 = 源类型的元素集合 ∩ 原始约束的元素集合
      const originalElements = effectiveSets.get(varId) || [];
      const intersection = computeSetIntersection(sourceElements, originalElements);
      
      if (intersection.length > 0) {
        effectiveSets.set(varId, intersection);
      } else {
        // 如果交集为空，使用源类型的元素集合（允许用户覆盖）
        effectiveSets.set(varId, sourceElements);
      }
    }
    
    sourceMap.set(varId, null);  // 源的来源是 null（用户选择）
  }

  // 3. 将所有有邻居的端口加入队列（不只是源）
  for (const varId of graph.keys()) {
    if (!inQueue.has(varId)) {
      queue.push(varId);
      inQueue.add(varId);
    }
  }

  // 4. BFS 传播：与邻居求交集
  while (queue.length > 0) {
    const varId = queue.shift()!;
    inQueue.delete(varId);
    
    const currentSet = effectiveSets.get(varId) || [];
    if (currentSet.length === 0) continue;

    // 获取所有邻居
    const neighbors = graph.get(varId);
    if (!neighbors) continue;

    for (const neighborId of neighbors) {
      const neighborSet = effectiveSets.get(neighborId) || [];
      if (neighborSet.length === 0) continue;

      // 计算交集（支持容器类型）
      const intersection = computeSetIntersection(neighborSet, currentSet);
      
      // 如果交集发生变化（变小了），更新并重新入队
      if (intersection.length > 0 && intersection.length < neighborSet.length) {
        effectiveSets.set(neighborId, intersection);
        sourceMap.set(neighborId, varId);  // 记录传播来源
        
        // 如果不在队列中，加入队列
        if (!inQueue.has(neighborId)) {
          queue.push(neighborId);
          inQueue.add(neighborId);
        }
      }
    }
  }

  return { effectiveSets, sources: sourceMap };
}
