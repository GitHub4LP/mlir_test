/**
 * 统一的类型选择器渲染逻辑
 *
 * 所有节点类型（Operation、Call、Entry、Return）使用相同的渲染逻辑：
 * 1. 计算 displayType（显示的类型）
 * 2. 计算 options 和 canEdit（可选集和是否可编辑）
 * 3. 渲染 UnifiedTypeSelector
 */

import type { Node, Edge } from '@xyflow/react';
import type { DataPin, FunctionDef, TypePropagationData } from '../types';
import { computeTypeSelectionState } from './typeSelection';
import { PortRef } from './port';

/**
 * 统一的 displayType 计算逻辑
 * 
 * 优先级：pinnedTypes > propagatedTypes > narrowedConstraints > 原始约束
 * 
 * 所有节点类型都使用这个函数，不要在组件中重复定义！
 */
export function getDisplayType(
  pin: DataPin,
  data: TypePropagationData
): string {
  const { pinnedTypes = {}, inputTypes = {}, outputTypes = {}, narrowedConstraints = {} } = data;
  
  // 1. 用户显式选择（pinnedTypes 键是 handleId）
  if (pinnedTypes[pin.id]) {
    return pinnedTypes[pin.id];
  }
  
  // 2. 传播结果（inputTypes/outputTypes 键是端口名）
  const parsed = PortRef.parseHandleId(pin.id);
  const portName = parsed?.name || '';
  
  if (parsed?.kind === 'data-in' && inputTypes[portName]) {
    return inputTypes[portName];
  }
  if (parsed?.kind === 'data-out' && outputTypes[portName]) {
    return outputTypes[portName];
  }
  
  // 3. 收窄后的约束
  if (narrowedConstraints[portName]) {
    return narrowedConstraints[portName];
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
  /** 当前图的所有节点 */
  nodes: Node[];
  /** 当前图的所有边 */
  edges: Edge[];
  /** 当前函数定义 */
  currentFunction: FunctionDef | undefined;
  /** 获取约束的具体类型列表 */
  getConcreteTypes: (constraint: string) => string[];
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
  const { nodeId, data, nodes, edges, currentFunction, getConcreteTypes, onTypeSelect } = params;

  // 1. 计算显示类型（统一逻辑）
  const displayType = getDisplayType(pin, data);

  // 2. 计算可选集和 canEdit
  const { options, canEdit } = computeTypeSelectionState(
    nodeId,
    pin.id,
    nodes,
    edges,
    currentFunction,
    getConcreteTypes
  );

  // 3. 构造回调
  const onSelect = (type: string) => onTypeSelect(pin.id, type, pin.typeConstraint);

  return { displayType, options, canEdit, onSelect };
}
