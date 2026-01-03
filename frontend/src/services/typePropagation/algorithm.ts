/**
 * 类型传播算法
 * 
 * BFS + 交集计算的传播算法。
 */

import type { PropagationGraph, PropagationResult, VariableId, TypeSource } from './types';

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

  // 1. 初始化：每个端口的有效集合 = 原始约束的元素集合
  for (const [varId, constraint] of portConstraints) {
    effectiveSets.set(varId, getConstraintElements(constraint));
  }

  // 2. 应用类型源（用户 pin + 单一元素约束）
  for (const source of sources) {
    const varId = source.portRef.key;
    const sourceElements = getConstraintElements(source.type);
    
    // 源的有效集合 = 源类型的元素集合 ∩ 原始约束的元素集合
    const originalElements = effectiveSets.get(varId) || [];
    const intersection = sourceElements.filter(t => originalElements.includes(t));
    
    if (intersection.length > 0) {
      effectiveSets.set(varId, intersection);
    } else {
      // 如果交集为空，使用源类型的元素集合（允许用户覆盖）
      effectiveSets.set(varId, sourceElements);
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

      // 计算交集
      const intersection = neighborSet.filter(t => currentSet.includes(t));
      
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
