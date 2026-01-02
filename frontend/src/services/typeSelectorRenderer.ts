/**
 * 统一的类型选择器渲染逻辑
 *
 * 所有节点类型（Operation、Call、Entry、Return）使用相同的渲染逻辑：
 * 1. 计算 displayType（显示的类型）
 * 2. 计算 options 和 canEdit（可选集和是否可编辑）
 * 3. 渲染 UnifiedTypeSelector
 * 
 * 设计原则：
 * - 使用框架无关的 EditorNode/EditorEdge 类型
 * - 不依赖任何渲染框架（React Flow、Vue Flow 等）
 */

import type { EditorNode, EditorEdge } from '../editor/types';
import type { DataPin, FunctionDef, TypePropagationData } from '../types';
import { computeTypeSelectionState } from './typeSelection';
import { PortRef } from './port';

/**
 * 统一的 displayType 计算逻辑
 * 
 * 优先级：portStates > pinnedTypes > 有效集合第一个元素 > 原始约束
 * 
 * 所有节点类型都使用这个函数，不要在组件中重复定义！
 */
export function getDisplayType(
  pin: DataPin,
  data: TypePropagationData
): string {
  const { pinnedTypes = {}, inputTypes = {}, outputTypes = {}, portStates = {} } = data;
  
  // 1. 从 portStates 获取（如果有）
  if (portStates[pin.id]?.displayType) {
    return portStates[pin.id].displayType;
  }
  
  // 2. 用户显式选择（pinnedTypes 键是 handleId）
  if (pinnedTypes[pin.id]) {
    return pinnedTypes[pin.id];
  }
  
  // 3. 从有效集合获取（inputTypes/outputTypes 键是端口名，值是 string[]）
  const parsed = PortRef.parseHandleId(pin.id);
  const portName = parsed?.name || '';
  
  if (parsed?.kind === 'data-in' && inputTypes[portName]) {
    const effectiveSet = inputTypes[portName];
    if (Array.isArray(effectiveSet) && effectiveSet.length > 0) {
      // 单一元素直接显示，多元素显示第一个（实际应该由 portStates 提供）
      return effectiveSet.length === 1 ? effectiveSet[0] : effectiveSet[0];
    }
  }
  if (parsed?.kind === 'data-out' && outputTypes[portName]) {
    const effectiveSet = outputTypes[portName];
    if (Array.isArray(effectiveSet) && effectiveSet.length > 0) {
      return effectiveSet.length === 1 ? effectiveSet[0] : effectiveSet[0];
    }
  }
  
  // 4. 原始约束
  return pin.typeConstraint;
}

/**
 * 类型选择器渲染所需的参数
 */
export interface TypeSelectorRenderParams {
  /** 节点 ID */
  nodeId: string;
  /** 节点的类型传播数据 */
  data: TypePropagationData;
  /** 当前图的所有节点（EditorNode 类型） */
  nodes: EditorNode[];
  /** 当前图的所有边（EditorEdge 类型） */
  edges: EditorEdge[];
  /** 当前函数定义 */
  currentFunction: FunctionDef | undefined;
  /** 获取约束映射到的类型约束集合元素 */
  getConstraintElements: (constraint: string) => string[];
  /** 找出所有元素集合是有效集合子集的约束名 */
  findSubsetConstraints: (E: string[]) => string[];
  /** 类型选择回调 */
  onTypeSelect: (portId: string, type: string, originalConstraint: string) => void;
}

/**
 * 计算类型选择器的渲染状态
 *
 * @returns { displayType, options, canEdit, onSelect }
 */
export function computeTypeSelectorState(
  pin: DataPin,
  params: TypeSelectorRenderParams
): {
  displayType: string;
  options: string[];
  canEdit: boolean;
  onSelect: (type: string) => void;
} {
  const { nodeId, data, nodes, edges, currentFunction, getConstraintElements, findSubsetConstraints, onTypeSelect } = params;
  const { portStates = {} } = data;

  // 1. 计算显示类型（统一逻辑）
  const displayType = getDisplayType(pin, data);

  // 2. 优先从 portStates 获取 options 和 canEdit
  const portState = portStates[pin.id];
  if (portState) {
    return {
      displayType: portState.displayType,
      options: portState.options,
      canEdit: portState.canEdit,
      onSelect: (type: string) => onTypeSelect(pin.id, type, pin.typeConstraint),
    };
  }

  // 3. 否则计算可选集和 canEdit
  const { options, canEdit } = computeTypeSelectionState(
    nodeId,
    pin.id,
    nodes,
    edges,
    currentFunction,
    getConstraintElements,
    findSubsetConstraints
  );

  // 4. 构造回调
  const onSelect = (type: string) => onTypeSelect(pin.id, type, pin.typeConstraint);

  return { displayType, options, canEdit, onSelect };
}
