/**
 * TypeSystemService
 * 
 * 统一管理类型系统状态，作为单一数据源。
 * 
 * 概念区分：
 * - 类型（Type）：具体的，如 I32, F32
 * - 类型约束（TypeConstraint）：范围，如 SignlessIntegerLike
 * 
 * 端口角色：
 * - definition：定义类型（Entry/Return），存储具体类型
 * - fixed：固定类型（Call 节点），由函数签名决定
 * - constrained：约束类型（操作节点），有类型约束
 * 
 * 函数签名类型编辑：
 * - 内部约束：函数内部连接到该端口的操作约束
 * - 外部约束：所有调用处传入/接收的具体类型
 * - 可选范围 = 内部约束交集 ∩ 外部类型兼容集
 * 
 * 设计原则：
 * - 使用框架无关的 EditorNode/EditorEdge 类型
 * - 不依赖任何渲染框架（React Flow、Vue Flow 等）
 */

import { typeConstraintStore } from '../stores';
import type { EditorNode, EditorEdge } from '../editor/types';
import type { 
  Project, 
  BlueprintNodeData, 
  FunctionCallData,
  FunctionEntryData,
  FunctionReturnData 
} from '../types';
import { PortRef, PortKind } from './port';

/**
 * 端口类型状态
 */
export interface PortTypeState {
  displayType: string;                    // 显示的类型
  source: 'pinned' | 'propagated' | 'constraint';  // 类型来源
  canEdit: boolean;                       // 是否可编辑
  options: string[] | null;               // 可选类型（null 表示使用约束默认）
}

/**
 * 获取约束的可选类型列表
 */
function getConstraintOptions(constraint: string): string[] | null {
  if (!constraint) return null;
  
  const { buildableTypes, isLoaded, getConstraintElements } = typeConstraintStore.getState();
  
  if (!isLoaded) return null;
  
  const types = getConstraintElements(constraint);
  if (types.length > 0) return types;
  
  if (buildableTypes.includes(constraint)) {
    return [constraint];
  }
  
  return null;
}

/**
 * 计算端口的类型状态
 * 
 * 核心模型：
 * - displayType = pinned > propagated > effectiveConstraint
 * - effectiveConstraint = narrowedConstraint ?? originalConstraint
 * - concreteOptions = getConstraintElements(effectiveConstraint)
 * - canEdit = !isExternallyDetermined && concreteOptions.length > 1
 * 
 * 关键概念：
 * - "外部决定"：类型由传播（非自己 pin）决定，用户不能修改
 * - 自己 pin 的类型不算"外部决定"，用户可以修改自己的选择
 */
export function computePortTypeState(params: {
  portId: string;
  nodeId: string;
  constraint: string;                     // 原始约束（如 SignlessIntegerLike）
  pinnedTypes: Record<string, string>;    // 用户选择
  propagatedType: string | null;          // 传播得到的类型
  narrowedConstraint: string | null;      // 收窄后的约束
  isConnected: boolean;                   // 是否有连接（保留接口兼容）
}): PortTypeState {
  const { portId, constraint, pinnedTypes, propagatedType, narrowedConstraint } = params;
  
  const isPinned = portId in pinnedTypes && !!pinnedTypes[portId];
  const pinnedType = pinnedTypes[portId];
  
  // 有效约束：收窄后约束 > 原始约束
  const effectiveConstraint = narrowedConstraint ?? constraint;
  
  // 可选的具体类型列表
  const options = getConstraintOptions(effectiveConstraint);
  
  // 确定显示类型和来源
  // 优先级：pinned > propagated > effectiveConstraint
  let displayType: string;
  let source: 'pinned' | 'propagated' | 'constraint';
  
  if (isPinned) {
    displayType = pinnedType;
    source = 'pinned';
  } else if (propagatedType) {
    displayType = propagatedType;
    source = 'propagated';
  } else {
    displayType = effectiveConstraint;
    source = 'constraint';
  }
  
  // 可编辑性判断：
  // 1. 如果类型被外部传播决定（非自己 pin）→ 不可编辑
  // 2. 否则，看具体选项数是否 > 1
  const isExternallyDetermined = propagatedType !== null && !isPinned;
  const canEdit = !isExternallyDetermined && (options?.length ?? 0) > 1;
  
  return {
    displayType,
    source,
    canEdit,
    options,
  };
}

/**
 * 计算函数签名端口（Entry/Return）的可编辑选项
 * 
 * 函数签名可以选择：
 * 1. 具体类型（如 I32）
 * 2. 类型约束（如 SignlessIntegerLike）
 * 
 * 规则：
 * - 候选必须满足所有内部约束（是其子集）
 * - 候选必须满足所有外部约束（是其子集）
 * 
 * @param internalConstraints - 函数内部连接的约束/类型列表
 * @param externalConstraints - 外部调用处连接的约束/类型列表
 * @returns 可选的类型/约束列表，null 表示无限制
 */
export function computeSignaturePortOptions(
  internalConstraints: string[],
  externalConstraints: string[]
): string[] | null {
  const { buildableTypes, constraintDefs, isLoaded, getConstraintElements: storeGetConstraintElements } = typeConstraintStore.getState();
  
  if (!isLoaded) return null;
  
  // 无连接，可选所有
  if (internalConstraints.length === 0 && externalConstraints.length === 0) {
    return null;
  }
  
  const allConstraints = [...internalConstraints, ...externalConstraints];
  
  // 检查候选是否满足所有约束
  const satisfiesAllConstraints = (candidateTypes: string[]): boolean => {
    for (const constraint of allConstraints) {
      const constraintTypes = storeGetConstraintElements(constraint);
      for (const t of candidateTypes) {
        if (!constraintTypes.includes(t)) {
          return false;
        }
      }
    }
    return true;
  };
  
  const validOptions: string[] = [];
  
  // 1. 检查具体类型
  for (const type of buildableTypes) {
    if (satisfiesAllConstraints([type])) {
      validOptions.push(type);
    }
  }
  
  // 2. 检查约束
  for (const [name] of constraintDefs) {
    if (buildableTypes.includes(name)) continue;
    
    const candidateTypes = storeGetConstraintElements(name);
    if (candidateTypes.length > 0 && satisfiesAllConstraints(candidateTypes)) {
      validOptions.push(name);
    }
  }
  
  return validOptions.length > 0 ? validOptions : null;
}

/**
 * 计算函数签名端口的类型状态
 */
export function computeSignaturePortState(params: {
  portId: string;
  nodeId: string;
  currentType: string;
  isConnected: boolean;
  isMainFunction: boolean;
  internalConstraints: string[];  // 函数内连接的约束
  externalTypes: string[];        // 调用处连接的类型
}): PortTypeState {
  const { currentType, isMainFunction, internalConstraints, externalTypes } = params;
  
  // main 函数签名不可编辑
  if (isMainFunction) {
    return {
      displayType: currentType,
      source: 'pinned',
      canEdit: false,
      options: null,
    };
  }
  
  // 计算可选类型
  const options = computeSignaturePortOptions(internalConstraints, externalTypes);
  
  // 有连接时，只能选择兼容的类型
  // 无连接时，可选所有类型
  return {
    displayType: currentType,
    source: 'pinned',
    canEdit: true,  // 签名端口始终可编辑（除了 main）
    options,
  };
}

/**
 * 计算 Call 节点端口的类型状态
 * Call 节点的类型由被调用函数的签名决定，不可编辑
 */
export function computeCallPortState(params: {
  portId: string;
  displayType: string;
}): PortTypeState {
  return {
    displayType: params.displayType,
    source: 'pinned',
    canEdit: false,  // Call 节点类型固定
    options: null,
  };
}

/**
 * 检查端口是否有连接
 */
export function isPortConnected(
  nodeId: string,
  portId: string,
  edges: Array<{ source: string; target: string; sourceHandle?: string | null; targetHandle?: string | null }>
): boolean {
  return edges.some(e =>
    (e.source === nodeId && e.sourceHandle === portId) ||
    (e.target === nodeId && e.targetHandle === portId)
  );
}

/**
 * 从 inputTypes/outputTypes 获取传播类型
 * 
 * 注意：返回传播的类型或收窄后的约束名
 */
export function getPropagatedType(
  portId: string,
  inputTypes: Record<string, string>,
  outputTypes: Record<string, string>
): string | null {
  let type: string | undefined;
  
  // 使用 PortRef 解析端口 ID
  const parsed = PortRef.parseHandleId(portId);
  if (parsed) {
    // 移除 variadic 索引后缀
    const name = parsed.name.replace(/_\d+$/, '');
    if (parsed.kind === PortKind.DataIn) {
      type = inputTypes[name];
    } else if (parsed.kind === PortKind.DataOut) {
      type = outputTypes[name];
    }
  }
  
  if (!type) return null;
  
  return type;
}

// ============================================================================
// 函数签名类型编辑辅助函数
// ============================================================================

/**
 * 获取函数内部连接到指定端口的约束列表
 * 
 * Entry 端口（参数）：查找连接到该输出端口的目标节点的输入约束
 * Return 端口（返回值）：查找连接到该输入端口的源节点的输出约束
 * 
 * @param portId - 端口 handleId（如 "data-out-a" 或 "data-in-result"）
 * @param nodeId - Entry/Return 节点 ID
 * @param nodes - 函数图中的所有节点（EditorNode 类型）
 * @param edges - 函数图中的所有边（EditorEdge 类型）
 */
export function getInternalConnectedConstraints(
  portId: string,
  nodeId: string,
  nodes: EditorNode[],
  edges: EditorEdge[]
): string[] {
  const constraints: string[] = [];
  const parsed = PortRef.parseHandleId(portId);
  if (!parsed) return constraints;
  
  // Entry 端口是输出（source），查找目标节点的输入约束
  if (parsed.kind === PortKind.DataOut) {
    const connectedEdges = edges.filter(
      e => e.source === nodeId && e.sourceHandle === portId
    );
    
    for (const edge of connectedEdges) {
      const targetNode = nodes.find(n => n.id === edge.target);
      if (!targetNode || targetNode.type !== 'operation') continue;
      
      const data = targetNode.data as BlueprintNodeData;
      const targetHandle = edge.targetHandle;
      if (!targetHandle) continue;
      
      // 使用 PortRef 解析目标端口
      const targetParsed = PortRef.parseHandleId(targetHandle);
      if (targetParsed && targetParsed.kind === PortKind.DataIn) {
        const inputName = targetParsed.name.replace(/_\d+$/, '');
        const arg = data.operation.arguments.find(
          a => a.kind === 'operand' && a.name === inputName
        );
        if (arg) {
          constraints.push(arg.typeConstraint);
        }
      }
    }
  }
  
  // Return 端口是输入（target），查找源节点的输出约束
  if (parsed.kind === PortKind.DataIn) {
    const connectedEdges = edges.filter(
      e => e.target === nodeId && e.targetHandle === portId
    );
    
    for (const edge of connectedEdges) {
      const sourceNode = nodes.find(n => n.id === edge.source);
      if (!sourceNode) continue;
      
      if (sourceNode.type === 'operation') {
        const data = sourceNode.data as BlueprintNodeData;
        const sourceHandle = edge.sourceHandle;
        if (!sourceHandle) continue;
        
        // 使用 PortRef 解析源端口
        const sourceParsed = PortRef.parseHandleId(sourceHandle);
        if (sourceParsed && sourceParsed.kind === PortKind.DataOut) {
          const outputName = sourceParsed.name.replace(/_\d+$/, '');
          const result = data.operation.results.find(r => r.name === outputName);
          if (result) {
            constraints.push(result.typeConstraint);
          }
        }
      } else if (sourceNode.type === 'function-entry') {
        // 连接自 Entry 节点的参数
        const data = sourceNode.data as FunctionEntryData;
        const port = data.outputs.find(p => p.id === edge.sourceHandle);
        if (port) {
          constraints.push(port.typeConstraint);
        }
      }
    }
  }
  
  return constraints;
}

/**
 * 获取所有调用处连接到指定函数端口的约束/类型列表
 * 
 * 对于参数端口：查找连接到 Call 节点输入的源端口类型
 * 对于返回值端口：查找连接到 Call 节点输出的目标端口约束
 * 
 * @param functionId - 函数 ID
 * @param portName - 端口名（参数名或返回值名）
 * @param portKind - 端口类型：'param' 或 'return'
 * @param project - 项目数据
 */
export function getExternalConnectedConstraints(
  functionId: string,
  portName: string,
  portKind: 'param' | 'return',
  project: Project
): string[] {
  const types: string[] = [];
  
  // 遍历所有函数图，查找调用此函数的 Call 节点
  const allFunctions = [project.mainFunction, ...project.customFunctions];
  
  for (const func of allFunctions) {
    const { nodes, edges } = func.graph;
    
    for (const node of nodes) {
      if (node.type !== 'function-call') continue;
      
      const callData = node.data as FunctionCallData;
      if (callData.functionId !== functionId) continue;
      
      // 找到对应的端口
      if (portKind === 'param') {
        // 参数端口：Call 节点的输入，查找连接到它的源类型
        const inputPort = callData.inputs.find(p => p.name === portName);
        if (!inputPort) continue;
        
        const connectedEdges = edges.filter(
          e => e.target === node.id && e.targetHandle === inputPort.id
        );
        
        for (const edge of connectedEdges) {
          const sourceNode = nodes.find(n => n.id === edge.source);
          if (!sourceNode) continue;
          
          const sourceType = getSourcePortType(sourceNode, edge.sourceHandle || '');
          if (sourceType) {
            types.push(sourceType);
          }
        }
      } else {
        // 返回值端口：Call 节点的输出，查找连接到它的目标约束
        const outputPort = callData.outputs.find(p => p.name === portName);
        if (!outputPort) continue;
        
        const connectedEdges = edges.filter(
          e => e.source === node.id && e.sourceHandle === outputPort.id
        );
        
        for (const edge of connectedEdges) {
          const targetNode = nodes.find(n => n.id === edge.target);
          if (!targetNode) continue;
          
          const targetConstraint = getTargetPortConstraint(targetNode, edge.targetHandle || '');
          if (targetConstraint) {
            types.push(targetConstraint);
          }
        }
      }
    }
  }
  
  return types;
}

/**
 * 获取源端口的类型/约束
 * 优先返回具体类型，否则返回约束
 */
function getSourcePortType(node: EditorNode, portId: string): string | null {
  const parsed = PortRef.parseHandleId(portId);
  
  switch (node.type) {
    case 'operation': {
      const data = node.data as BlueprintNodeData;
      if (parsed && parsed.kind === PortKind.DataOut) {
        const name = parsed.name.replace(/_\d+$/, '');
        // 优先使用传播后的类型，否则使用原始约束
        const propagatedType = data.outputTypes?.[name];
        if (propagatedType) return propagatedType;
        // 回退到原始约束
        const result = data.operation.results.find(r => r.name === name);
        return result?.typeConstraint || null;
      }
      break;
    }
    case 'function-entry': {
      const data = node.data as FunctionEntryData;
      // 使用端口名称查找
      const portName = parsed?.name;
      const port = data.outputs.find(p => p.name === portName);
      return port?.typeConstraint || null;
    }
    case 'function-call': {
      const data = node.data as FunctionCallData;
      const portName = parsed?.name;
      const port = data.outputs.find(p => p.name === portName);
      return port?.typeConstraint || null;
    }
  }
  return null;
}

/**
 * 获取目标端口的约束
 */
function getTargetPortConstraint(node: EditorNode, portId: string): string | null {
  const parsed = PortRef.parseHandleId(portId);
  
  switch (node.type) {
    case 'operation': {
      const data = node.data as BlueprintNodeData;
      if (parsed && parsed.kind === PortKind.DataIn) {
        const name = parsed.name.replace(/_\d+$/, '');
        const arg = data.operation.arguments.find(
          a => a.kind === 'operand' && a.name === name
        );
        return arg?.typeConstraint || null;
      }
      break;
    }
    case 'function-return': {
      const data = node.data as FunctionReturnData;
      const portName = parsed?.name;
      const port = data.inputs.find(p => p.name === portName);
      return port?.typeConstraint || null;
    }
    case 'function-call': {
      const data = node.data as FunctionCallData;
      const portName = parsed?.name;
      const port = data.inputs.find(p => p.name === portName);
      return port?.typeConstraint || null;
    }
  }
  return null;
}
