/**
 * 统一的类型变更处理逻辑
 * 
 * 所有四种节点类型（Operation、Call、Entry、Return）使用完全相同的逻辑：
 * 1. 判断是否应该 pin（固定）类型
 * 2. 更新节点的 pinnedTypes
 * 3. 触发类型传播
 * 
 * 数据存储统一：
 * - pinnedTypes：用户显式选择的类型
 * - inputTypes/outputTypes：传播结果
 * - displayType = pinnedTypes[portId] > inputTypes/outputTypes[portName] > 原始约束
 * 
 * Entry/Return 的显示类型就是函数签名。
 * 
 * 设计原则：
 * - 使用框架无关的 EditorNode/EditorEdge 类型
 * - 不依赖任何渲染框架（React Flow、Vue Flow 等）
 * - 只有"有效 pin"才存入 pinnedTypes（真正产生收窄效果的选择）
 */

import type { EditorNode, EditorEdge } from '../editor/types';
import type { FunctionDef, PortState } from '../types';
import { triggerTypePropagationWithSignature } from './typePropagation';
import { computeOptionsExcludingSelf } from './typePropagation/propagator';
import { PortRef } from './port';

/**
 * 计算新的 pinnedTypes
 * 
 * 逻辑：
 * 1. 如果用户选择的类型等于原始约束 → 取消 pin（恢复默认）
 * 2. 如果用户选择的类型等于 displayType，且当前没有 pin → 不 pin（传播已给出这个类型）
 * 3. 如果用户选择的类型等于 displayType，且当前已 pin 同样的类型 → 保持 pin
 * 4. 如果用户选择的类型不等于 displayType → pin 新类型
 * 
 * @param type - 用户选择的类型
 * @param portId - 端口 ID（handleId）
 * @param currentPinnedTypes - 当前的 pinnedTypes
 * @param displayType - 当前显示的类型（来自 portState）
 * @param originalConstraint - 端口的原始约束
 * @returns 新的 pinnedTypes，如果不需要更新返回 null
 */
export function computeNewPinnedTypes(
  type: string,
  portId: string,
  currentPinnedTypes: Record<string, string>,
  displayType: string,
  originalConstraint: string
): Record<string, string> | null {
  const currentPinned = currentPinnedTypes[portId];
  
  // 1. 选择原始约束 = 取消 pin，恢复默认
  if (type === originalConstraint) {
    if (currentPinned !== undefined) {
      const newPinnedTypes = { ...currentPinnedTypes };
      delete newPinnedTypes[portId];
      return newPinnedTypes;
    }
    return null;
  }
  
  // 2. 选择当前显示的类型
  if (type === displayType) {
    if (currentPinned === type) {
      // displayType 来自自己的 pin，保持不变
      return null;
    } else {
      // displayType 来自传播，不需要 pin
      if (currentPinned !== undefined) {
        const newPinnedTypes = { ...currentPinnedTypes };
        delete newPinnedTypes[portId];
        return newPinnedTypes;
      }
      return null;
    }
  }
  
  // 3. 选择不同的类型，pin 它
  return {
    ...currentPinnedTypes,
    [portId]: type,
  };
}

/**
 * 类型变更处理的依赖项
 */
export interface TypeChangeHandlerDeps {
  edges: EditorEdge[];
  getCurrentFunction: () => FunctionDef | null;
  getConstraintElements: (constraint: string) => string[];
  pickConstraintName: (types: string[], nodeDialect: string | null, pinnedName: string | null) => string | null;
  findSubsetConstraints: (E: string[]) => string[];
}

/**
 * 更新节点的 pinnedTypes 并触发类型传播
 * 
 * 逻辑：
 * - 从节点的 portStates 获取 displayType 和 constraint
 * - 根据 displayType、originalConstraint 和当前 pinnedTypes 决定是否 pin
 * - 判断 pin 是否有效（是否真正产生收窄效果）
 * - 触发类型传播
 */
export function handlePinnedTypeChange<T extends { pinnedTypes?: Record<string, string>; portStates?: Record<string, PortState> }>(
  nodeId: string,
  portId: string,
  type: string,
  _originalConstraint: string | undefined, // 备用，优先使用 portState.constraint
  currentNodes: EditorNode[],
  deps: TypeChangeHandlerDeps
): EditorNode[] {
  // 1. 找到目标节点，获取 displayType、constraint 和当前 pinnedTypes
  const targetNode = currentNodes.find(n => n.id === nodeId);
  if (!targetNode) return currentNodes;
  
  const nodeData = targetNode.data as T;
  const currentPinnedTypes = nodeData.pinnedTypes || {};
  const portState = nodeData.portStates?.[portId];
  const displayType = portState?.displayType || _originalConstraint || type;
  const originalConstraint = portState?.constraint || _originalConstraint || type;
  
  // 2. 计算新的 pinnedTypes（初步判断）
  let newPinnedTypes = computeNewPinnedTypes(type, portId, currentPinnedTypes, displayType, originalConstraint);
  
  // 如果不需要更新，直接返回
  if (newPinnedTypes === null) {
    return currentNodes;
  }

  // 3. 判断 pin 是否有效（是否真正产生收窄效果）
  // 只有当要添加 pin 时才需要判断
  if (newPinnedTypes[portId] !== undefined) {
    const currentFunction = deps.getCurrentFunction() ?? undefined;
    
    // 构建 portKey
    const portRef = PortRef.fromHandle(nodeId, portId);
    if (portRef) {
      // 计算排除自己后的有效集合
      const optionsWithoutSelf = computeOptionsExcludingSelf(
        portRef.key,
        currentNodes,
        deps.edges,
        currentFunction,
        deps.getConstraintElements
      );
      
      // 获取用户选择的类型的元素集合
      const selectedElements = deps.getConstraintElements(type);
      
      // 有效 pin = 用户选择后产生了收窄效果
      // 即：selectedElements.length < optionsWithoutSelf.length
      const isEffectivePin = selectedElements.length < optionsWithoutSelf.length;
      
      if (!isEffectivePin) {
        // 不是有效 pin，删除它
        newPinnedTypes = { ...newPinnedTypes };
        delete newPinnedTypes[portId];
        
        // 如果 pinnedTypes 变成空对象，且原来也是空的，不需要更新
        if (Object.keys(newPinnedTypes).length === 0 && Object.keys(currentPinnedTypes).length === 0) {
          return currentNodes;
        }
      }
    }
  }

  // 4. 更新节点的 pinnedTypes
  const updatedNodes = currentNodes.map(node => {
    if (node.id === nodeId) {
      return {
        ...node,
        data: {
          ...nodeData,
          pinnedTypes: newPinnedTypes,
        },
      };
    }
    return node;
  });

  // 5. 触发类型传播
  const currentFunction = deps.getCurrentFunction() ?? undefined;
  const result = triggerTypePropagationWithSignature(
    updatedNodes,
    deps.edges,
    currentFunction,
    deps.getConstraintElements,
    deps.pickConstraintName,
    deps.findSubsetConstraints
  );

  return result.nodes;
}

/**
 * 触发类型传播（不更新节点数据）
 * 
 * 用于需要重新传播但不改变 pinnedTypes 的场景，如：
 * - 添加/删除连线后
 * - 添加/删除参数后
 */
export function triggerPropagationOnly(
  currentNodes: EditorNode[],
  deps: TypeChangeHandlerDeps
): EditorNode[] {
  const currentFunction = deps.getCurrentFunction() ?? undefined;
  const result = triggerTypePropagationWithSignature(
    currentNodes,
    deps.edges,
    currentFunction,
    deps.getConstraintElements,
    deps.pickConstraintName,
    deps.findSubsetConstraints
  );

  // 注意：签名同步不再在这里同步执行，而是在 MainLayout 的 useEffect 中异步处理
  // 这样可以避免在渲染期间更新其他组件

  return result.nodes;
}
