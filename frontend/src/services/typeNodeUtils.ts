/**
 * TypeNode 类型定义和工具函数
 * 
 * 用于类型选择器的类型结构表示和序列化/解析
 */

// ============ 类型定义 ============

export type TypeNode = ScalarNode | CompositeNode;

export interface ScalarNode {
  kind: 'scalar';
  name: string;  // 具体类型如 'f32' 或约束如 'AnyType'
}

export interface CompositeNode {
  kind: 'composite';
  wrapper: string;       // 'tensor', 'vector', 'memref', 'complex', etc.
  shape?: (number | null)[];  // null 表示动态维度 (?)
  element: TypeNode;
}

// 可嵌套的容器类型
export const WRAPPERS = [
  { name: 'tensor', hasShape: true },
  { name: 'vector', hasShape: true },
  { name: 'memref', hasShape: true },
  { name: 'complex', hasShape: false },
  { name: 'unranked_tensor', hasShape: false },
  { name: 'unranked_memref', hasShape: false },
] as const;

export type WrapperInfo = typeof WRAPPERS[number];

// ============ 工具函数 ============

/**
 * 将 TypeNode 序列化为字符串
 */
export function serializeType(node: TypeNode): string {
  if (node.kind === 'scalar') return node.name;

  const elem = serializeType(node.element);
  const { wrapper, shape } = node;

  if (!shape || shape.length === 0) {
    return `${wrapper}<${elem}>`;
  }

  const shapeStr = shape.map(d => d === null ? '?' : d).join('x');
  return `${wrapper}<${shapeStr}x${elem}>`;
}

/**
 * 将字符串解析为 TypeNode
 */
export function parseType(str: string): TypeNode {
  str = str.trim();

  for (const w of WRAPPERS) {
    if (str.startsWith(w.name + '<') && str.endsWith('>')) {
      const inner = str.slice(w.name.length + 1, -1);

      if (w.hasShape) {
        // 智能解析：找到 element 开始的位置
        // element 可能是：标量(f32)、约束(AnyFloat)、或复合类型(tensor<...>)
        // 从左到右扫描，找到第一个非 shape 部分
        const parts: string[] = [];
        let remaining = inner;

        while (remaining) {
          // 检查是否是数字或 ?
          const match = remaining.match(/^(\d+|\?)(x|$)/);
          if (match) {
            parts.push(match[1]);
            remaining = remaining.slice(match[0].length);
            if (!match[2]) break; // 没有 x 了
          } else {
            // 不是 shape 部分，剩余的是 element
            break;
          }
        }

        if (parts.length > 0 && remaining) {
          const shape = parts.map(s => s === '?' ? null : parseInt(s, 10));
          return { kind: 'composite', wrapper: w.name, shape, element: parseType(remaining) };
        }
      }
      return { kind: 'composite', wrapper: w.name, element: parseType(inner) };
    }
  }

  return { kind: 'scalar', name: str || 'AnyType' };
}

/**
 * 用指定的包装器包装一个 TypeNode
 */
export function wrapWith(node: TypeNode, wrapper: string): CompositeNode {
  const w = WRAPPERS.find(x => x.name === wrapper);
  return {
    kind: 'composite',
    wrapper,
    shape: w?.hasShape ? [4] : undefined,
    element: node,
  };
}

