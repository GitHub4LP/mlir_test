/**
 * 方言依赖计算服务
 * 
 * 计算函数的直接依赖方言和可达方言集
 */

import type { Project, FunctionDef, GraphState, BlueprintNodeData } from '../types';

/**
 * 从图中计算直接使用的方言
 * 
 * 遍历所有 Operation 节点，收集其方言名
 */
export function computeDirectDialects(graph: GraphState): string[] {
  const dialects = new Set<string>();
  
  for (const node of graph.nodes) {
    if (node.type === 'operation') {
      const data = node.data as BlueprintNodeData;
      if (data.operation?.dialect) {
        dialects.add(data.operation.dialect);
      }
    }
  }
  
  return [...dialects].sort();
}

/**
 * 查找函数定义
 */
export function findFunction(project: Project, functionId: string): FunctionDef | undefined {
  if (project.mainFunction.id === functionId) {
    return project.mainFunction;
  }
  return project.customFunctions.find(f => f.id === functionId);
}

/**
 * 计算函数的可达方言集（递归）
 * 
 * 可达方言 = 直接依赖 ∪ 所有被调用函数的可达方言
 * 
 * @param functionId 函数 ID
 * @param project 项目
 * @param visited 已访问的函数 ID（防止循环调用死循环）
 */
export function computeReachableDialects(
  functionId: string,
  project: Project,
  visited: Set<string> = new Set()
): string[] {
  // 防止循环调用
  if (visited.has(functionId)) return [];
  visited.add(functionId);
  
  const func = findFunction(project, functionId);
  if (!func) return [];
  
  // 从 directDialects 开始
  const result = new Set(func.directDialects || []);
  
  // 遍历 Call 节点，递归收集被调用函数的可达方言
  for (const node of func.graph.nodes) {
    if (node.type === 'function-call') {
      const callData = node.data as { functionId: string };
      const calledDialects = computeReachableDialects(
        callData.functionId,
        project,
        new Set(visited)  // 传递副本，允许不同路径访问同一函数
      );
      calledDialects.forEach(d => result.add(d));
    }
  }
  
  return [...result].sort();
}

/**
 * 计算项目方言列表（所有函数 directDialects 的并集）
 * 
 * 用于项目保存时更新 project.dialects
 */
export function computeProjectDialects(project: Project): string[] {
  const dialects = new Set<string>();
  
  // 主函数
  for (const d of project.mainFunction.directDialects || []) {
    dialects.add(d);
  }
  
  // 自定义函数
  for (const func of project.customFunctions) {
    for (const d of func.directDialects || []) {
      dialects.add(d);
    }
  }
  
  return [...dialects].sort();
}

/**
 * 从 fullName 提取方言名
 * 
 * @example extractDialectFromFullName('arith.addi') => 'arith'
 */
export function extractDialectFromFullName(fullName: string): string | null {
  const dotIndex = fullName.indexOf('.');
  if (dotIndex === -1) return null;
  return fullName.substring(0, dotIndex);
}
