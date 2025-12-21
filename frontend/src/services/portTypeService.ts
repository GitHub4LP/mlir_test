/**
 * 端口类型获取服务
 * 
 * 提供统一的端口类型获取逻辑，供所有渲染器复用。
 */

import { PortRef } from './port';

/**
 * 类型数据接口
 */
export interface TypeData {
  pinnedTypes?: Record<string, string>;
  inputTypes?: Record<string, string>;
  outputTypes?: Record<string, string>;
}

/**
 * 获取端口的当前类型
 * 
 * 优先级：pinnedTypes > inputTypes/outputTypes
 * 
 * @param pinId 端口 handle ID（如 'data-out-a'）
 * @param typeData 类型数据
 * @returns 端口类型，未找到返回 undefined
 */
export function getPortType(
  pinId: string,
  typeData: TypeData
): string | undefined {
  const { pinnedTypes = {}, inputTypes = {}, outputTypes = {} } = typeData;

  // 1. 优先使用 pinnedTypes
  if (pinnedTypes[pinId]) {
    return pinnedTypes[pinId];
  }

  // 2. 解析 handle ID
  const parsed = PortRef.parseHandleId(pinId);
  if (!parsed) return undefined;

  // 3. 处理 variadic 端口名（去掉 _0, _1 后缀）
  let portName = parsed.name;
  const match = portName.match(/^(.+)_\d+$/);
  if (match) {
    portName = match[1];
  }

  // 4. 根据端口方向返回类型
  if (parsed.kind === 'data-in') {
    return inputTypes[portName];
  } else if (parsed.kind === 'data-out') {
    return outputTypes[portName];
  }

  return undefined;
}

/**
 * 创建 getPortType 的包装函数（用于组件回调）
 * 
 * @param typeData 类型数据
 * @returns 包装后的函数
 */
export function createPortTypeGetter(typeData: TypeData): (pinId: string) => string | undefined {
  return (pinId: string) => getPortType(pinId, typeData);
}
