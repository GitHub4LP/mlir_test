/**
 * Traits 推断器
 * 
 * 根据函数图结构自动推断函数签名的 Traits。
 * 
 * 算法：
 * 1. 构建类型等价图（基于操作 traits 和连线）
 * 2. 计算连通分量（等价类）
 * 3. 检查函数端口是否符合 MLIR 标准 traits 模式
 * 
 * 支持的 MLIR 标准 traits：
 * - SameOperandsAndResultType：所有参数和返回值类型相同
 * - SameTypeOperands：所有参数类型相同
 * - SameOperandsElementType：所有参数的元素类型相同（容器类型）
 * - SameOperandsAndResultElementType：所有参数和返回值的元素类型相同（容器类型）
 */

import type { EditorNode, EditorEdge } from '../../editor/types';
import type { FunctionDef, BlueprintNodeData } from '../../types';
import { 
  hasSameOperandsAndResultTypeTrait, 
  hasSameTypeOperandsTrait,
  hasSameOperandsElementTypeTrait,
  hasSameOperandsAndResultElementTypeTrait
} from '../typeSystem';

/**
 * 函数 Trait（使用 MLIR 标准名称）
 */
export interface InferredFunctionTrait {
  /** MLIR 标准 trait 名称 */
  kind: 'SameOperandsAndResultType' | 'SameTypeOperands' | 'SameOperandsElementType' | 'SameOperandsAndResultElementType';
}

/**
 * 端口标识符格式：nodeId:direction:portName
 */
type PortId = string;

/**
 * Union-Find 数据结构
 */
class UnionFind {
  private parent: Map<PortId, PortId> = new Map();
  private rank: Map<PortId, number> = new Map();

  find(x: PortId): PortId {
    if (!this.parent.has(x)) {
      this.parent.set(x, x);
      this.rank.set(x, 0);
    }
    
    if (this.parent.get(x) !== x) {
      this.parent.set(x, this.find(this.parent.get(x)!));
    }
    return this.parent.get(x)!;
  }

  union(x: PortId, y: PortId): void {
    const rootX = this.find(x);
    const rootY = this.find(y);
    
    if (rootX === rootY) return;
    
    const rankX = this.rank.get(rootX) || 0;
    const rankY = this.rank.get(rootY) || 0;
    
    if (rankX < rankY) {
      this.parent.set(rootX, rootY);
    } else if (rankX > rankY) {
      this.parent.set(rootY, rootX);
    } else {
      this.parent.set(rootY, rootX);
      this.rank.set(rootX, rankX + 1);
    }
  }

  /**
   * 获取所有连通分量
   */
  getComponents(): Map<PortId, PortId[]> {
    const components = new Map<PortId, PortId[]>();
    
    for (const [node] of this.parent) {
      const root = this.find(node);
      if (!components.has(root)) {
        components.set(root, []);
      }
      components.get(root)!.push(node);
    }
    
    return components;
  }
}

/**
 * 推断函数签名的 Traits
 * 
 * @param nodes - 函数图的节点
 * @param edges - 函数图的边
 * @param functionDef - 函数定义
 * @returns 推断的 MLIR 标准 traits 列表
 */
export function inferFunctionTraits(
  nodes: EditorNode[],
  edges: EditorEdge[],
  functionDef: FunctionDef
): InferredFunctionTrait[] {
  // 1. 构建类型等价图（完整类型）
  const fullUf = buildTypeEquivalenceGraph(nodes, edges, 'full');
  
  // 2. 构建元素类型等价图
  const elementUf = buildTypeEquivalenceGraph(nodes, edges, 'element');
  
  // 3. 找到函数端口
  const entryNode = nodes.find(n => n.type === 'function-entry');
  const returnNode = nodes.find(n => n.type === 'function-return');
  
  if (!entryNode && !returnNode) {
    return [];
  }
  
  // 收集参数端口和返回值端口
  const paramPorts: PortId[] = [];
  const returnPorts: PortId[] = [];
  
  if (entryNode) {
    for (const param of functionDef.parameters) {
      paramPorts.push(makePortId(entryNode.id, param.name, 'out'));
    }
  }
  
  if (returnNode) {
    for (const ret of functionDef.returnTypes) {
      returnPorts.push(makePortId(returnNode.id, ret.name, 'in'));
    }
  }
  
  const traits: InferredFunctionTrait[] = [];
  
  // 4. 检查完整类型 traits
  // SameOperandsAndResultType：所有参数和返回值在同一等价类
  if (paramPorts.length > 0 && returnPorts.length > 0) {
    const allPorts = [...paramPorts, ...returnPorts];
    if (allInSameComponent(fullUf, allPorts)) {
      traits.push({ kind: 'SameOperandsAndResultType' });
      // SameOperandsAndResultType 隐含 SameTypeOperands，不再检查
    } else if (paramPorts.length >= 2 && allInSameComponent(fullUf, paramPorts)) {
      // 只有参数在同一等价类
      traits.push({ kind: 'SameTypeOperands' });
    }
  } else if (paramPorts.length >= 2) {
    // 没有返回值，只检查 SameTypeOperands
    if (allInSameComponent(fullUf, paramPorts)) {
      traits.push({ kind: 'SameTypeOperands' });
    }
  }
  
  // 5. 检查元素类型 traits（只有在没有完整类型 traits 时才检查）
  // 因为完整类型相同隐含元素类型相同
  const hasFullTypeTrait = traits.some(t => 
    t.kind === 'SameOperandsAndResultType' || t.kind === 'SameTypeOperands'
  );
  
  if (!hasFullTypeTrait) {
    // SameOperandsAndResultElementType：所有参数和返回值的元素类型在同一等价类
    if (paramPorts.length > 0 && returnPorts.length > 0) {
      const allPorts = [...paramPorts, ...returnPorts];
      if (allInSameComponent(elementUf, allPorts)) {
        traits.push({ kind: 'SameOperandsAndResultElementType' });
      } else if (paramPorts.length >= 2 && allInSameComponent(elementUf, paramPorts)) {
        // 只有参数的元素类型在同一等价类
        traits.push({ kind: 'SameOperandsElementType' });
      }
    } else if (paramPorts.length >= 2) {
      // 没有返回值，只检查 SameOperandsElementType
      if (allInSameComponent(elementUf, paramPorts)) {
        traits.push({ kind: 'SameOperandsElementType' });
      }
    }
  }
  
  return traits;
}

/**
 * 检查所有端口是否在同一连通分量
 */
function allInSameComponent(uf: UnionFind, ports: PortId[]): boolean {
  if (ports.length < 2) return true;
  
  const firstRoot = uf.find(ports[0]);
  for (let i = 1; i < ports.length; i++) {
    if (uf.find(ports[i]) !== firstRoot) {
      return false;
    }
  }
  return true;
}

/**
 * 构建类型等价图
 * 
 * 使用 Union-Find 数据结构，将类型相同的端口合并到同一集合
 * 
 * @param nodes - 函数图的节点
 * @param edges - 函数图的边
 * @param mode - 'full' 完整类型等价，'element' 元素类型等价
 */
function buildTypeEquivalenceGraph(
  nodes: EditorNode[],
  edges: EditorEdge[],
  mode: 'full' | 'element' = 'full'
): UnionFind {
  const uf = new UnionFind();
  
  // 1. 从操作 Traits 建立等价关系
  for (const node of nodes) {
    if (node.type !== 'operation') continue;
    
    const data = node.data as BlueprintNodeData;
    const operation = data.operation;
    const variadicCounts = data.variadicCounts || {};
    
    // 收集端口
    const operandPorts: PortId[] = [];
    const resultPorts: PortId[] = [];
    
    for (const arg of operation.arguments) {
      if (arg.kind === 'operand') {
        if (arg.isVariadic) {
          const count = variadicCounts[arg.name] ?? 1;
          for (let i = 0; i < count; i++) {
            operandPorts.push(makePortId(node.id, `${arg.name}_${i}`, 'in'));
          }
        } else {
          operandPorts.push(makePortId(node.id, arg.name, 'in'));
        }
      }
    }
    
    for (const result of operation.results) {
      if (result.isVariadic) {
        const count = variadicCounts[result.name] ?? 1;
        for (let i = 0; i < count; i++) {
          resultPorts.push(makePortId(node.id, `${result.name}_${i}`, 'out'));
        }
      } else {
        resultPorts.push(makePortId(node.id, result.name, 'out'));
      }
    }
    
    if (mode === 'full') {
      // 完整类型等价
      // SameOperandsAndResultType：所有端口等价
      if (hasSameOperandsAndResultTypeTrait(operation)) {
        const allPorts = [...operandPorts, ...resultPorts];
        unionAll(uf, allPorts);
      }
      
      // SameTypeOperands：所有输入端口等价
      if (hasSameTypeOperandsTrait(operation)) {
        unionAll(uf, operandPorts);
      }
    } else {
      // 元素类型等价
      // SameOperandsAndResultElementType：所有端口的元素类型等价
      if (hasSameOperandsAndResultElementTypeTrait(operation)) {
        const allPorts = [...operandPorts, ...resultPorts];
        unionAll(uf, allPorts);
      }
      
      // SameOperandsElementType：所有输入端口的元素类型等价
      if (hasSameOperandsElementTypeTrait(operation)) {
        unionAll(uf, operandPorts);
      }
      
      // 完整类型相同隐含元素类型相同
      if (hasSameOperandsAndResultTypeTrait(operation)) {
        const allPorts = [...operandPorts, ...resultPorts];
        unionAll(uf, allPorts);
      }
      
      if (hasSameTypeOperandsTrait(operation)) {
        unionAll(uf, operandPorts);
      }
    }
  }
  
  // 2. 从连线建立等价关系
  // 连线传播完整类型，也传播元素类型（因为类型相同则元素类型也相同）
  for (const edge of edges) {
    // 跳过执行边
    if (edge.sourceHandle?.startsWith('exec-') || edge.targetHandle?.startsWith('exec-')) {
      continue;
    }
    
    if (!edge.sourceHandle || !edge.targetHandle) continue;
    
    // 解析 handle ID
    const sourcePort = parseHandleToPortId(edge.source, edge.sourceHandle);
    const targetPort = parseHandleToPortId(edge.target, edge.targetHandle);
    
    if (sourcePort && targetPort) {
      uf.union(sourcePort, targetPort);
    }
  }
  
  return uf;
}

/**
 * 将多个端口合并到同一集合
 */
function unionAll(uf: UnionFind, ports: PortId[]): void {
  if (ports.length < 2) return;
  
  for (let i = 1; i < ports.length; i++) {
    uf.union(ports[0], ports[i]);
  }
}

/**
 * 创建端口 ID
 */
function makePortId(nodeId: string, portName: string, direction: 'in' | 'out'): PortId {
  return `${nodeId}:${direction}:${portName}`;
}

/**
 * 从 handle ID 解析端口 ID
 */
function parseHandleToPortId(nodeId: string, handleId: string): PortId | null {
  // handle 格式: data-in-{name} 或 data-out-{name}
  const match = handleId.match(/^data-(in|out)-(.+)$/);
  if (!match) return null;
  
  const [, direction, name] = match;
  return makePortId(nodeId, name, direction as 'in' | 'out');
}
