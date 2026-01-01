/**
 * SizeProvider 模块入口
 */

// 类型导出
export type { SizeProvider, DOMSizeProvider, InteractiveComponentType } from './types';

// 实现导出
export { ConfigSizeProvider, configSizeProvider } from './ConfigSizeProvider';
export { DOMSizeProvider as DOMSizeProviderImpl, domSizeProvider } from './DOMSizeProvider';

// 当前活跃的 SizeProvider
import type { SizeProvider } from './types';
import { configSizeProvider } from './ConfigSizeProvider';

let currentSizeProvider: SizeProvider = configSizeProvider;

/**
 * 设置当前 SizeProvider
 * @param provider - SizeProvider 实例
 */
export function setSizeProvider(provider: SizeProvider): void {
  currentSizeProvider = provider;
}

/**
 * 获取当前 SizeProvider
 */
export function getSizeProvider(): SizeProvider {
  return currentSizeProvider;
}
