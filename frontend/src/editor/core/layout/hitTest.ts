/**
 * 基于 LayoutBox 的命中检测
 * 使用深度优先遍历，支持 hitTestBehavior
 */

import type { LayoutBox, HitResult } from './types';

// ============================================================================
// 命中检测核心函数
// ============================================================================

/**
 * 检查点是否在盒子内（相对坐标）
 */
function isPointInBox(x: number, y: number, box: LayoutBox): boolean {
  return x >= box.x && x < box.x + box.width && y >= box.y && y < box.y + box.height;
}

/**
 * 递归命中检测
 * @param box - 当前盒子
 * @param x - 相对于父容器的 X 坐标
 * @param y - 相对于父容器的 Y 坐标
 * @param path - 从根到当前盒子的路径
 * @returns 命中结果，或 null
 */
function hitTestRecursive(
  box: LayoutBox,
  x: number,
  y: number,
  path: LayoutBox[]
): HitResult | null {
  // 1. 检查是否在边界框内
  if (!isPointInBox(x, y, box)) {
    return null;
  }

  // 2. 计算相对坐标（相对于当前盒子的左上角）
  const localX = x - box.x;
  const localY = y - box.y;

  // 3. 构建当前路径
  const currentPath = [...path, box];

  // 4. 深度优先遍历子节点（后绘制的在上面，所以从后往前遍历）
  for (let i = box.children.length - 1; i >= 0; i--) {
    const child = box.children[i];
    const result = hitTestRecursive(child, localX, localY, currentPath);

    if (result) {
      // 检查命中行为
      const behavior = result.box.interactive?.hitTestBehavior ?? 'translucent';

      if (behavior === 'opaque') {
        // 阻挡：返回此结果，不继续检查
        return result;
      } else if (behavior === 'translucent') {
        // 穿透但记录：返回此结果
        return result;
      }
      // transparent: 继续检查其他子节点
    }
  }

  // 5. 没有子节点命中，检查自身
  const selfBehavior = box.interactive?.hitTestBehavior ?? 'translucent';
  if (selfBehavior !== 'transparent') {
    return {
      box,
      path: currentPath,
      localX,
      localY,
    };
  }

  return null;
}

/**
 * 命中检测入口函数
 * @param root - 根 LayoutBox
 * @param x - 相对于根 LayoutBox 左上角的 X 坐标（不是相对于 root.x）
 * @param y - 相对于根 LayoutBox 左上角的 Y 坐标（不是相对于 root.y）
 * @returns 命中结果，或 null
 * 
 * 注意：传入的坐标应该是相对于节点左上角的坐标，
 * 即 canvasX - nodeX, canvasY - nodeY
 * 
 * 内部子元素的坐标是相对于 root 左上角的（从 0,0 开始），
 * 而 root.x, root.y 是节点在画布上的绝对位置。
 * 因此传入的相对坐标可以直接用于命中测试。
 */
export function hitTestLayoutBox(
  root: LayoutBox,
  x: number,
  y: number
): HitResult | null {
  // 创建一个临时的 root，将其位置设为 0,0，这样子元素的坐标就是相对坐标
  // 传入的 x, y 已经是相对于节点左上角的坐标
  const localRoot: LayoutBox = {
    ...root,
    x: 0,
    y: 0,
  };
  return hitTestRecursive(localRoot, x, y, []);
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 从命中路径中查找指定类型的盒子
 * @param result - 命中结果
 * @param type - 盒子类型
 * @returns 找到的盒子，或 undefined
 */
export function findBoxInPath(result: HitResult, type: string): LayoutBox | undefined {
  return result.path.find((box) => box.type === type);
}

/**
 * 从命中路径中查找具有指定 interactive.id 前缀的盒子
 * @param result - 命中结果
 * @param idPrefix - ID 前缀
 * @returns 找到的盒子，或 undefined
 */
export function findBoxByIdPrefix(
  result: HitResult,
  idPrefix: string
): LayoutBox | undefined {
  return result.path.find((box) => box.interactive?.id?.startsWith(idPrefix));
}

/**
 * 获取命中盒子的 interactive.id
 * @param result - 命中结果
 * @returns ID，或 undefined
 */
export function getHitId(result: HitResult): string | undefined {
  return result.box.interactive?.id;
}

// ============================================================================
// ID 解析
// ============================================================================

/**
 * 解析后的 interactive.id 信息
 */
export interface ParsedInteractiveId {
  /** 命中类型 */
  type: 'handle' | 'type-label' | 'param-add' | 'param-remove' | 'param-name' 
      | 'return-add' | 'return-remove' | 'return-name' | 'variadic' 
      | 'traits-toggle' | 'summary-toggle' | 'node' | 'unknown';
  /** Handle ID（用于 handle, type-label） */
  handleId?: string;
  /** 索引（用于 param-remove, param-name, return-remove, return-name） */
  index?: number;
  /** Variadic 组名 */
  group?: string;
  /** Variadic 动作 */
  action?: 'add' | 'remove';
  /** 节点 ID（用于 node） */
  nodeId?: string;
}

/**
 * 解析 interactive.id
 * 
 * 格式：
 * - handle-{handleId}
 * - type-label-{handleId}
 * - param-add
 * - param-remove-{index}
 * - param-name-{index}
 * - return-add
 * - return-remove-{index}
 * - return-name-{index}
 * - variadic-{group}-add
 * - variadic-{group}-remove
 * - traits-toggle
 * - summary-toggle
 * - node-{nodeId}
 */
export function parseInteractiveId(id: string | undefined): ParsedInteractiveId {
  if (!id) {
    return { type: 'unknown' };
  }

  // handle-{handleId}
  if (id.startsWith('handle-')) {
    return { type: 'handle', handleId: id.substring(7) };
  }

  // type-label-{handleId}
  if (id.startsWith('type-label-')) {
    return { type: 'type-label', handleId: id.substring(11) };
  }

  // param-add
  if (id === 'param-add') {
    return { type: 'param-add' };
  }

  // param-remove-{index}
  if (id.startsWith('param-remove-')) {
    return { type: 'param-remove', index: parseInt(id.substring(13), 10) };
  }

  // param-name-{index}
  if (id.startsWith('param-name-')) {
    return { type: 'param-name', index: parseInt(id.substring(11), 10) };
  }

  // return-add
  if (id === 'return-add') {
    return { type: 'return-add' };
  }

  // return-remove-{index}
  if (id.startsWith('return-remove-')) {
    return { type: 'return-remove', index: parseInt(id.substring(14), 10) };
  }

  // return-name-{index}
  if (id.startsWith('return-name-')) {
    return { type: 'return-name', index: parseInt(id.substring(12), 10) };
  }

  // variadic-{group}-add / variadic-{group}-remove
  if (id.startsWith('variadic-')) {
    const rest = id.substring(9);
    if (rest.endsWith('-add')) {
      return { type: 'variadic', group: rest.substring(0, rest.length - 4), action: 'add' };
    }
    if (rest.endsWith('-remove')) {
      return { type: 'variadic', group: rest.substring(0, rest.length - 7), action: 'remove' };
    }
  }

  // traits-toggle
  if (id === 'traits-toggle') {
    return { type: 'traits-toggle' };
  }

  // summary-toggle
  if (id === 'summary-toggle') {
    return { type: 'summary-toggle' };
  }

  // node-{nodeId}
  if (id.startsWith('node-')) {
    return { type: 'node', nodeId: id.substring(5) };
  }

  return { type: 'unknown' };
}

/**
 * 从 HitResult 中提取节点 ID
 */
export function extractNodeId(result: HitResult): string | undefined {
  // 查找 node 类型的盒子
  const nodeBox = result.path.find((box) => box.type === 'node');
  if (nodeBox?.interactive?.id) {
    const parsed = parseInteractiveId(nodeBox.interactive.id);
    return parsed.nodeId;
  }
  return undefined;
}

/**
 * 判断 handle 是否为输出端口
 */
export function isOutputHandle(handleId: string): boolean {
  return handleId.includes('-out') || handleId.startsWith('exec-out');
}

// ============================================================================
// Handle 位置提取
// ============================================================================

/**
 * Handle 位置信息
 */
export interface HandlePosition {
  handleId: string;
  x: number;  // 绝对 X 坐标（Handle 中心）
  y: number;  // 绝对 Y 坐标（Handle 中心）
  isOutput: boolean;
}

/**
 * 递归提取所有 Handle 的绝对位置
 */
function extractHandlesRecursive(
  box: LayoutBox,
  offsetX: number,
  offsetY: number,
  result: HandlePosition[]
): void {
  const absX = offsetX + box.x;
  const absY = offsetY + box.y;

  // 检查是否是 Handle
  if (box.type === 'handle' && box.interactive?.id) {
    const id = box.interactive.id;
    // id 格式: handle-{handleId}
    if (id.startsWith('handle-')) {
      const handleId = id.substring(7);
      result.push({
        handleId,
        x: absX + box.width / 2,  // Handle 中心
        y: absY + box.height / 2,
        isOutput: isOutputHandle(handleId),
      });
    }
  }

  // 递归处理子节点
  for (const child of box.children) {
    extractHandlesRecursive(child, absX, absY, result);
  }
}

/**
 * 从 LayoutBox 树中提取所有 Handle 的绝对位置
 * @param root - 根 LayoutBox（节点）
 * @returns Handle 位置列表
 */
export function extractHandlePositions(root: LayoutBox): HandlePosition[] {
  const result: HandlePosition[] = [];
  extractHandlesRecursive(root, 0, 0, result);
  return result;
}
