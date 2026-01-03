/**
 * 传播图构建
 * 
 * 从 Traits 和连线构建类型传播图。
 * 传播图描述类型如何从一个端口流向另一个端口。
 */

import type { EditorNode, EditorEdge } from '../../editor/types';
import type { PropagationGraph, VariableId } from './types';
import { makeVariableId } from './types';
import type { BlueprintNodeData, FunctionDef } from '../../types';
import { hasSameOperandsAndResultTypeTrait } from '../typeSystem';
import { PortRef, dataIn, dataOut } from '../port';

/**
 * 构建传播图
 * 
 * 传播图描述类型如何从一个端口流向另一个端口：
 * 1. 操作节点内传播：由操作的 Trait 决定（如 SameOperandsAndResultType）
 * 2. 函数级别传播：由函数的 Traits 决定（如 SameType）
 * 3. 节点间传播：由连线决定（双向）
 * 
 * @param nodes - 当前函数图的节点（EditorNode 类型）
 * @param edges - 当前函数图的边（EditorEdge 类型）
 * @param currentFunction - 当前函数定义（用于获取函数级别 Traits）
 */
export function buildPropagationGraph(
  nodes: EditorNode[],
  edges: EditorEdge[],
  currentFunction?: FunctionDef
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
    const variadicCounts = data.variadicCounts || {};

    // SameOperandsAndResultType：所有数据端口类型相同
    if (hasSameOperandsAndResultTypeTrait(operation)) {
      const ports: VariableId[] = [];

      // 收集所有数据端口（包括 variadic 展开的实例）
      for (const arg of operation.arguments) {
        if (arg.kind === 'operand') {
          if (arg.isVariadic) {
            // Variadic 端口：展开为多个实例
            const count = variadicCounts[arg.name] ?? 1;
            for (let i = 0; i < count; i++) {
              ports.push(makeVariableId(dataIn(node.id, `${arg.name}_${i}`)));
            }
          } else {
            ports.push(makeVariableId(dataIn(node.id, arg.name)));
          }
        }
      }
      for (const result of operation.results) {
        if (result.isVariadic) {
          const count = variadicCounts[result.name] ?? 1;
          for (let i = 0; i < count; i++) {
            ports.push(makeVariableId(dataOut(node.id, `${result.name}_${i}`)));
          }
        } else {
          ports.push(makeVariableId(dataOut(node.id, result.name)));
        }
      }

      // 任意两个端口之间双向传播
      for (let i = 0; i < ports.length; i++) {
        for (let j = i + 1; j < ports.length; j++) {
          addBidirectionalEdge(ports[i], ports[j]);
        }
      }
    }
  }

  // 2. 从函数级别 Traits 构建传播边
  if (currentFunction?.traits) {
    // 找到 Entry 和 Return 节点
    const entryNode = nodes.find(n => n.type === 'function-entry');
    const returnNode = nodes.find(n => n.type === 'function-return');

    for (const trait of currentFunction.traits) {
      if (trait.kind === 'SameType') {
        const ports: VariableId[] = [];

        for (const portName of trait.ports) {
          if (portName.startsWith('return:')) {
            // 返回值端口（FunctionReturn 的输入）
            const returnName = portName.slice(7);
            if (returnNode) {
              ports.push(makeVariableId(dataIn(returnNode.id, returnName)));
            }
          } else {
            // 参数端口（FunctionEntry 的输出）
            if (entryNode) {
              ports.push(makeVariableId(dataOut(entryNode.id, portName)));
            }
          }
        }

        // 任意两个端口之间双向传播
        for (let i = 0; i < ports.length; i++) {
          for (let j = i + 1; j < ports.length; j++) {
            addBidirectionalEdge(ports[i], ports[j]);
          }
        }
      }
    }
  }

  // 3. 从连线构建节点间传播边（双向）
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
