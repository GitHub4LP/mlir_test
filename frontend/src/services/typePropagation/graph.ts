/**
 * 传播图构建
 * 
 * 从 Traits 和连线构建类型传播图。
 * 传播图描述类型如何从一个端口流向另一个端口。
 * 
 * 支持两种边类型：
 * - 'full': 完整类型传播（SameOperandsAndResultType, SameTypeOperands）
 * - 'element': 元素类型传播（SameOperandsElementType, SameOperandsAndResultElementType）
 */

import type { EditorNode, EditorEdge } from '../../editor/types';
import type { PropagationGraph, ExtendedPropagationGraph, VariableId } from './types';
import { makeVariableId } from './types';
import type { BlueprintNodeData, FunctionDef, FunctionCallData } from '../../types';
import { 
  hasSameOperandsAndResultTypeTrait, 
  hasSameTypeOperandsTrait,
  hasSameOperandsElementTypeTrait,
  hasSameOperandsAndResultElementTypeTrait
} from '../typeSystem';
import { PortRef, dataIn, dataOut } from '../port';

/**
 * 函数查找器类型
 */
export type FunctionLookup = (functionName: string) => FunctionDef | null;

/**
 * 收集操作节点的所有输入端口（operands）
 */
function collectOperandPorts(
  node: EditorNode,
  data: BlueprintNodeData
): VariableId[] {
  const ports: VariableId[] = [];
  const variadicCounts = data.variadicCounts || {};

  for (const arg of data.operation.arguments) {
    if (arg.kind === 'operand') {
      if (arg.isVariadic) {
        const count = variadicCounts[arg.name] ?? 1;
        for (let i = 0; i < count; i++) {
          ports.push(makeVariableId(dataIn(node.id, `${arg.name}_${i}`)));
        }
      } else {
        ports.push(makeVariableId(dataIn(node.id, arg.name)));
      }
    }
  }

  return ports;
}

/**
 * 收集操作节点的所有输出端口（results）
 */
function collectResultPorts(
  node: EditorNode,
  data: BlueprintNodeData
): VariableId[] {
  const ports: VariableId[] = [];
  const variadicCounts = data.variadicCounts || {};

  for (const result of data.operation.results) {
    if (result.isVariadic) {
      const count = variadicCounts[result.name] ?? 1;
      for (let i = 0; i < count; i++) {
        ports.push(makeVariableId(dataOut(node.id, `${result.name}_${i}`)));
      }
    } else {
      ports.push(makeVariableId(dataOut(node.id, result.name)));
    }
  }

  return ports;
}

/**
 * 在端口列表中的所有端口之间添加双向边
 */
function connectAllPorts(
  ports: VariableId[],
  addBidirectionalEdge: (a: VariableId, b: VariableId) => void
): void {
  for (let i = 0; i < ports.length; i++) {
    for (let j = i + 1; j < ports.length; j++) {
      addBidirectionalEdge(ports[i], ports[j]);
    }
  }
}

/**
 * 构建传播图
 * 
 * 传播图描述类型如何从一个端口流向另一个端口：
 * 1. 操作节点内传播：由操作的 Trait 决定（如 SameOperandsAndResultType）
 * 2. Call 节点内传播：由被调用函数的 Traits 决定
 * 3. 函数级别传播：由函数的 Traits 决定
 * 4. 节点间传播：由连线决定（双向）
 * 
 * @param nodes - 当前函数图的节点（EditorNode 类型）
 * @param edges - 当前函数图的边（EditorEdge 类型）
 * @param currentFunction - 当前函数定义（用于获取函数级别 Traits）
 * @param getFunctionById - 函数查找器（用于获取 Call 节点被调用函数的 Traits）
 */
export function buildPropagationGraph(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction?: FunctionDef,
  getFunctionById?: FunctionLookup
): PropagationGraph {
  const graph: PropagationGraph = new Map();

  // 辅助函数：添加边（确保双向）
  const addEdge = (from: VariableId, to: VariableId) => {
    if (!graph.has(from)) {
      graph.set(from, new Set());
    }
    graph.get(from)!.add(to);
  };

  // 辅助函数：添加双向边
  const addBidirectionalEdge = (a: VariableId, b: VariableId) => {
    addEdge(a, b);
    addEdge(b, a);
  };

  // 1. 从操作 Traits 构建节点内传播边
  for (const node of nodes) {
    if (node.type !== 'operation') continue;

    const data = node.data as BlueprintNodeData;
    const operation = data.operation;

    // 收集端口
    const operandPorts = collectOperandPorts(node, data);
    const resultPorts = collectResultPorts(node, data);

    // SameOperandsAndResultType：所有数据端口类型相同
    // 注意：使用独立的 if 而非 else if，支持多 trait 组合（边的并集）
    if (hasSameOperandsAndResultTypeTrait(operation)) {
      const allPorts = [...operandPorts, ...resultPorts];
      connectAllPorts(allPorts, addBidirectionalEdge);
    }
    
    // SameTypeOperands：仅输入端口类型相同（不包括输出）
    if (hasSameTypeOperandsTrait(operation)) {
      connectAllPorts(operandPorts, addBidirectionalEdge);
    }
  }

  // 2. 从 Call 节点的被调用函数 Traits 构建传播边
  // Call 节点的 inputs = operands，outputs = results
  for (const node of nodes) {
    if (node.type !== 'function-call') continue;

    const callData = node.data as FunctionCallData;
    const calledFunction = getFunctionById?.(callData.functionName);
    
    if (!calledFunction) continue;

    // 如果被调用函数没有 traits，跳过（traits 会在用户进入该函数时推断）
    // 注意：这里不实时推断，因为推断需要加载被调用函数的图，开销较大
    if (!calledFunction.traits || calledFunction.traits.length === 0) continue;

    // 收集 Call 节点的输入端口（对应函数参数）
    const inputPorts: VariableId[] = [];
    for (const param of calledFunction.parameters) {
      inputPorts.push(makeVariableId(dataIn(node.id, param.name)));
    }

    // 收集 Call 节点的输出端口（对应函数返回值）
    const outputPorts: VariableId[] = [];
    for (const ret of calledFunction.returnTypes) {
      outputPorts.push(makeVariableId(dataOut(node.id, ret.name)));
    }

    // 应用被调用函数的 traits（与 operation 节点相同逻辑）
    for (const trait of calledFunction.traits) {
      if (trait.kind === 'SameOperandsAndResultType') {
        // 所有输入和输出端口类型相同
        const allPorts = [...inputPorts, ...outputPorts];
        connectAllPorts(allPorts, addBidirectionalEdge);
      } else if (trait.kind === 'SameTypeOperands') {
        // 所有输入端口类型相同
        connectAllPorts(inputPorts, addBidirectionalEdge);
      }
      // 注意：元素类型 traits 在简单传播图中不处理
      // 它们在 buildExtendedPropagationGraph 中处理
    }
  }

  // 3. 从函数级别 Traits 构建传播边
  if (currentFunction?.traits) {
    // 找到 Entry 和 Return 节点
    const entryNode = nodes.find(n => n.type === 'function-entry');
    const returnNode = nodes.find(n => n.type === 'function-return');

    for (const trait of currentFunction.traits) {
      if (trait.kind === 'SameOperandsAndResultType') {
        // 所有参数和返回值类型相同
        const ports: VariableId[] = [];

        // 添加所有参数端口
        if (entryNode && currentFunction.parameters) {
          for (const param of currentFunction.parameters) {
            ports.push(makeVariableId(dataOut(entryNode.id, param.name)));
          }
        }

        // 添加所有返回值端口
        if (returnNode && currentFunction.returnTypes) {
          for (const ret of currentFunction.returnTypes) {
            ports.push(makeVariableId(dataIn(returnNode.id, ret.name)));
          }
        }

        // 任意两个端口之间双向传播
        connectAllPorts(ports, addBidirectionalEdge);
      } else if (trait.kind === 'SameTypeOperands') {
        // 所有参数类型相同
        const ports: VariableId[] = [];

        if (entryNode && currentFunction.parameters) {
          for (const param of currentFunction.parameters) {
            ports.push(makeVariableId(dataOut(entryNode.id, param.name)));
          }
        }

        connectAllPorts(ports, addBidirectionalEdge);
      }
    }
  }

  // 4. 从连线构建节点间传播边（双向）
  for (const edge of edges) {
    // 跳过执行边
    if (edge.sourceHandle?.startsWith('exec-') || edge.targetHandle?.startsWith('exec-')) {
      continue;
    }

    if (!edge.sourceHandle || !edge.targetHandle) continue;

    // 使用 PortRef 从 handle 创建变量 ID
    const sourceRef = PortRef.fromHandle(edge.source, edge.sourceHandle);
    const targetRef = PortRef.fromHandle(edge.target, edge.targetHandle);
    
    if (sourceRef && targetRef) {
      addBidirectionalEdge(sourceRef.key, targetRef.key);
    }
  }

  return graph;
}

/**
 * 构建扩展传播图（支持边类型标记）
 * 
 * 与 buildPropagationGraph 类似，但返回的图包含边类型信息，
 * 用于支持元素类型传播。
 * 
 * @param nodes - 当前函数图的节点
 * @param edges - 当前函数图的边
 * @param currentFunction - 当前函数定义
 * @param getFunctionById - 函数查找器（用于获取 Call 节点被调用函数的 Traits）
 */
export function buildExtendedPropagationGraph(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction?: FunctionDef,
  getFunctionById?: FunctionLookup
): ExtendedPropagationGraph {
  const graph: ExtendedPropagationGraph = new Map();

  // 辅助函数：添加边
  const addEdge = (from: VariableId, to: VariableId, kind: 'full' | 'element') => {
    if (!graph.has(from)) {
      graph.set(from, []);
    }
    const edgeList = graph.get(from)!;
    // 避免重复边（同一目标同一类型）
    if (!edgeList.some(e => e.target === to && e.kind === kind)) {
      edgeList.push({ target: to, kind });
    }
  };

  // 辅助函数：添加双向边
  const addBidirectionalEdge = (a: VariableId, b: VariableId, kind: 'full' | 'element') => {
    addEdge(a, b, kind);
    addEdge(b, a, kind);
  };

  // 辅助函数：连接所有端口
  const connectAllPortsWithKind = (
    ports: VariableId[],
    kind: 'full' | 'element'
  ) => {
    for (let i = 0; i < ports.length; i++) {
      for (let j = i + 1; j < ports.length; j++) {
        addBidirectionalEdge(ports[i], ports[j], kind);
      }
    }
  };

  // 1. 从操作 Traits 构建节点内传播边
  for (const node of nodes) {
    if (node.type !== 'operation') continue;

    const data = node.data as BlueprintNodeData;
    const operation = data.operation;

    // 收集端口
    const operandPorts = collectOperandPorts(node, data);
    const resultPorts = collectResultPorts(node, data);

    // 完整类型传播 traits
    if (hasSameOperandsAndResultTypeTrait(operation)) {
      const allPorts = [...operandPorts, ...resultPorts];
      connectAllPortsWithKind(allPorts, 'full');
    }
    
    if (hasSameTypeOperandsTrait(operation)) {
      connectAllPortsWithKind(operandPorts, 'full');
    }

    // 元素类型传播 traits
    if (hasSameOperandsElementTypeTrait(operation)) {
      connectAllPortsWithKind(operandPorts, 'element');
    }
    
    if (hasSameOperandsAndResultElementTypeTrait(operation)) {
      const allPorts = [...operandPorts, ...resultPorts];
      connectAllPortsWithKind(allPorts, 'element');
    }
  }

  // 2. 从 Call 节点的被调用函数 Traits 构建传播边
  for (const node of nodes) {
    if (node.type !== 'function-call') continue;

    const callData = node.data as FunctionCallData;
    const calledFunction = getFunctionById?.(callData.functionName);
    
    if (!calledFunction?.traits) continue;

    // 收集 Call 节点的输入端口
    const inputPorts: VariableId[] = [];
    for (const param of calledFunction.parameters) {
      inputPorts.push(makeVariableId(dataIn(node.id, param.name)));
    }

    // 收集 Call 节点的输出端口
    const outputPorts: VariableId[] = [];
    for (const ret of calledFunction.returnTypes) {
      outputPorts.push(makeVariableId(dataOut(node.id, ret.name)));
    }

    // 应用被调用函数的 traits
    for (const trait of calledFunction.traits) {
      if (trait.kind === 'SameOperandsAndResultType') {
        // 完整类型传播：所有输入和输出端口
        const allPorts = [...inputPorts, ...outputPorts];
        connectAllPortsWithKind(allPorts, 'full');
      } else if (trait.kind === 'SameTypeOperands') {
        // 完整类型传播：所有输入端口
        connectAllPortsWithKind(inputPorts, 'full');
      } else if (trait.kind === 'SameOperandsAndResultElementType') {
        // 元素类型传播：所有输入和输出端口
        const allPorts = [...inputPorts, ...outputPorts];
        connectAllPortsWithKind(allPorts, 'element');
      } else if (trait.kind === 'SameOperandsElementType') {
        // 元素类型传播：所有输入端口
        connectAllPortsWithKind(inputPorts, 'element');
      }
    }
  }

  // 3. 从函数级别 Traits 构建传播边（完整类型）
  if (currentFunction?.traits) {
    const entryNode = nodes.find(n => n.type === 'function-entry');
    const returnNode = nodes.find(n => n.type === 'function-return');

    for (const trait of currentFunction.traits) {
      if (trait.kind === 'SameOperandsAndResultType') {
        // 所有参数和返回值类型相同
        const ports: VariableId[] = [];

        if (entryNode && currentFunction.parameters) {
          for (const param of currentFunction.parameters) {
            ports.push(makeVariableId(dataOut(entryNode.id, param.name)));
          }
        }

        if (returnNode && currentFunction.returnTypes) {
          for (const ret of currentFunction.returnTypes) {
            ports.push(makeVariableId(dataIn(returnNode.id, ret.name)));
          }
        }

        connectAllPortsWithKind(ports, 'full');
      } else if (trait.kind === 'SameTypeOperands') {
        // 所有参数类型相同
        const ports: VariableId[] = [];

        if (entryNode && currentFunction.parameters) {
          for (const param of currentFunction.parameters) {
            ports.push(makeVariableId(dataOut(entryNode.id, param.name)));
          }
        }

        connectAllPortsWithKind(ports, 'full');
      } else if (trait.kind === 'SameOperandsAndResultElementType') {
        // 所有参数和返回值的元素类型相同
        const ports: VariableId[] = [];

        if (entryNode && currentFunction.parameters) {
          for (const param of currentFunction.parameters) {
            ports.push(makeVariableId(dataOut(entryNode.id, param.name)));
          }
        }

        if (returnNode && currentFunction.returnTypes) {
          for (const ret of currentFunction.returnTypes) {
            ports.push(makeVariableId(dataIn(returnNode.id, ret.name)));
          }
        }

        connectAllPortsWithKind(ports, 'element');
      } else if (trait.kind === 'SameOperandsElementType') {
        // 所有参数的元素类型相同
        const ports: VariableId[] = [];

        if (entryNode && currentFunction.parameters) {
          for (const param of currentFunction.parameters) {
            ports.push(makeVariableId(dataOut(entryNode.id, param.name)));
          }
        }

        connectAllPortsWithKind(ports, 'element');
      }
    }
  }

  // 4. 从连线构建节点间传播边（完整类型）
  for (const edge of edges) {
    if (edge.sourceHandle?.startsWith('exec-') || edge.targetHandle?.startsWith('exec-')) {
      continue;
    }

    if (!edge.sourceHandle || !edge.targetHandle) continue;

    const sourceRef = PortRef.fromHandle(edge.source, edge.sourceHandle);
    const targetRef = PortRef.fromHandle(edge.target, edge.targetHandle);
    
    if (sourceRef && targetRef) {
      addBidirectionalEdge(sourceRef.key, targetRef.key, 'full');
    }
  }

  return graph;
}

/**
 * 将扩展传播图转换为简单传播图（向后兼容）
 * 
 * 忽略边类型，只保留连接关系。
 */
export function toSimplePropagationGraph(extended: ExtendedPropagationGraph): PropagationGraph {
  const simple: PropagationGraph = new Map();
  
  for (const [from, edgeList] of extended) {
    const targets = new Set<VariableId>();
    for (const edge of edgeList) {
      targets.add(edge.target);
    }
    simple.set(from, targets);
  }
  
  return simple;
}
