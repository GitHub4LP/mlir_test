/**
 * 节点交互服务 - 纯函数集合
 * 
 * 所有方法都是纯函数：
 * - 接收当前状态作为参数
 * - 返回新状态
 * - 无副作用
 * 
 * 设计原则：
 * - 框架无关：使用 EditorNode 类型，不依赖任何渲染框架
 * - 单一职责：只处理数据变换，不负责状态管理或 UI 渲染
 */

import type { EditorNode } from '../editor/types';
import type { FunctionDef, FunctionTrait, BlueprintNodeData } from '../types';

// ============================================================
// 属性变更
// ============================================================

/**
 * 处理属性变更
 * 
 * @param nodes - 当前节点列表
 * @param nodeId - 目标节点 ID
 * @param attributeName - 属性名
 * @param value - 新值
 * @returns 更新后的节点列表
 */
export function handleAttributeChange(
  nodes: EditorNode[],
  nodeId: string,
  attributeName: string,
  value: string
): EditorNode[] {
  return nodes.map(node => {
    if (node.id !== nodeId) return node;
    
    const data = node.data as BlueprintNodeData;
    return {
      ...node,
      data: {
        ...data,
        attributes: {
          ...data.attributes,
          [attributeName]: value,
        },
      },
    };
  });
}

// ============================================================
// Variadic 端口操作
// ============================================================

/**
 * 增加 Variadic 端口实例
 * 
 * @param nodes - 当前节点列表
 * @param nodeId - 目标节点 ID
 * @param groupName - Variadic 组名
 * @returns 更新后的节点列表
 */
export function handleVariadicAdd(
  nodes: EditorNode[],
  nodeId: string,
  groupName: string
): EditorNode[] {
  return nodes.map(node => {
    if (node.id !== nodeId) return node;
    
    const data = node.data as BlueprintNodeData;
    const currentCount = data.variadicCounts?.[groupName] ?? 0;
    
    return {
      ...node,
      data: {
        ...data,
        variadicCounts: {
          ...data.variadicCounts,
          [groupName]: currentCount + 1,
        },
      },
    };
  });
}

/**
 * 减少 Variadic 端口实例（最小为 0）
 * 
 * @param nodes - 当前节点列表
 * @param nodeId - 目标节点 ID
 * @param groupName - Variadic 组名
 * @returns 更新后的节点列表
 */
export function handleVariadicRemove(
  nodes: EditorNode[],
  nodeId: string,
  groupName: string
): EditorNode[] {
  return nodes.map(node => {
    if (node.id !== nodeId) return node;
    
    const data = node.data as BlueprintNodeData;
    const currentCount = data.variadicCounts?.[groupName] ?? 0;
    
    return {
      ...node,
      data: {
        ...data,
        variadicCounts: {
          ...data.variadicCounts,
          [groupName]: Math.max(0, currentCount - 1),
        },
      },
    };
  });
}


// ============================================================
// 函数签名管理
// ============================================================

/**
 * 生成唯一参数名
 */
function generateUniqueParameterName(existingNames: string[], prefix: string = 'arg'): string {
  let index = existingNames.length;
  let name = `${prefix}${index}`;
  while (existingNames.includes(name)) {
    index++;
    name = `${prefix}${index}`;
  }
  return name;
}

/**
 * 生成唯一返回值名
 */
function generateUniqueReturnName(existingNames: string[], prefix: string = 'result'): string {
  let index = existingNames.length;
  let name = `${prefix}${index}`;
  while (existingNames.includes(name)) {
    index++;
    name = `${prefix}${index}`;
  }
  return name;
}

/**
 * 添加函数参数
 * 
 * @param functionDef - 当前函数定义
 * @param constraint - 参数约束（默认 'AnyType'）
 * @returns 更新后的函数定义
 */
export function handleParameterAdd(
  functionDef: FunctionDef,
  constraint: string = 'AnyType'
): FunctionDef {
  const existingNames = functionDef.parameters.map(p => p.name);
  const newName = generateUniqueParameterName(existingNames);
  
  return {
    ...functionDef,
    parameters: [
      ...functionDef.parameters,
      { name: newName, constraint },
    ],
  };
}

/**
 * 移除函数参数
 * 
 * @param functionDef - 当前函数定义
 * @param parameterName - 参数名
 * @returns 更新后的函数定义
 */
export function handleParameterRemove(
  functionDef: FunctionDef,
  parameterName: string
): FunctionDef {
  return {
    ...functionDef,
    parameters: functionDef.parameters.filter(p => p.name !== parameterName),
    // 同时从 traits 中移除引用该参数的端口
    traits: functionDef.traits?.map(trait => ({
      ...trait,
      ports: trait.ports.filter(port => port !== parameterName),
    })).filter(trait => trait.ports.length >= 2),
  };
}

/**
 * 重命名函数参数
 * 
 * @param functionDef - 当前函数定义
 * @param oldName - 旧名称
 * @param newName - 新名称
 * @returns 更新后的函数定义
 */
export function handleParameterRename(
  functionDef: FunctionDef,
  oldName: string,
  newName: string
): FunctionDef {
  // 检查新名称是否已存在
  if (functionDef.parameters.some(p => p.name === newName)) {
    return functionDef;
  }
  
  return {
    ...functionDef,
    parameters: functionDef.parameters.map(p =>
      p.name === oldName ? { ...p, name: newName } : p
    ),
    // 同时更新 traits 中的引用
    traits: functionDef.traits?.map(trait => ({
      ...trait,
      ports: trait.ports.map(port => port === oldName ? newName : port),
    })),
  };
}

/**
 * 添加返回值
 * 
 * @param functionDef - 当前函数定义
 * @param constraint - 返回值约束（默认 'AnyType'）
 * @returns 更新后的函数定义
 */
export function handleReturnTypeAdd(
  functionDef: FunctionDef,
  constraint: string = 'AnyType'
): FunctionDef {
  const existingNames = functionDef.returnTypes.map(r => r.name);
  const newName = generateUniqueReturnName(existingNames);
  
  return {
    ...functionDef,
    returnTypes: [
      ...functionDef.returnTypes,
      { name: newName, constraint },
    ],
  };
}

/**
 * 移除返回值
 * 
 * @param functionDef - 当前函数定义
 * @param returnName - 返回值名
 * @returns 更新后的函数定义
 */
export function handleReturnTypeRemove(
  functionDef: FunctionDef,
  returnName: string
): FunctionDef {
  const portName = `return:${returnName}`;
  return {
    ...functionDef,
    returnTypes: functionDef.returnTypes.filter(r => r.name !== returnName),
    // 同时从 traits 中移除引用该返回值的端口
    traits: functionDef.traits?.map(trait => ({
      ...trait,
      ports: trait.ports.filter(port => port !== portName),
    })).filter(trait => trait.ports.length >= 2),
  };
}

/**
 * 重命名返回值
 * 
 * @param functionDef - 当前函数定义
 * @param oldName - 旧名称
 * @param newName - 新名称
 * @returns 更新后的函数定义
 */
export function handleReturnTypeRename(
  functionDef: FunctionDef,
  oldName: string,
  newName: string
): FunctionDef {
  // 检查新名称是否已存在
  if (functionDef.returnTypes.some(r => r.name === newName)) {
    return functionDef;
  }
  
  const oldPortName = `return:${oldName}`;
  const newPortName = `return:${newName}`;
  
  return {
    ...functionDef,
    returnTypes: functionDef.returnTypes.map(r =>
      r.name === oldName ? { ...r, name: newName } : r
    ),
    // 同时更新 traits 中的引用
    traits: functionDef.traits?.map(trait => ({
      ...trait,
      ports: trait.ports.map(port => port === oldPortName ? newPortName : port),
    })),
  };
}

// ============================================================
// Traits 管理
// ============================================================

/**
 * 验证 Traits
 * 
 * @param traits - Traits 列表
 * @returns 验证结果
 */
export function validateTraits(traits: FunctionTrait[]): { valid: boolean; error?: string } {
  for (const trait of traits) {
    if (trait.kind === 'SameType' && trait.ports.length < 2) {
      return { valid: false, error: 'SameType trait 至少需要 2 个端口' };
    }
  }
  return { valid: true };
}

/**
 * 更新函数 Traits
 * 
 * @param functionDef - 当前函数定义
 * @param traits - 新的 Traits 列表
 * @returns 更新后的函数定义，或 null（验证失败）
 */
export function handleTraitsChange(
  functionDef: FunctionDef,
  traits: FunctionTrait[]
): FunctionDef | null {
  const validation = validateTraits(traits);
  if (!validation.valid) {
    return null;
  }
  
  return {
    ...functionDef,
    traits,
  };
}
