/**
 * 平台抽象层入口
 * 
 * 提供平台检测和桥接实例获取。
 */

import type { PlatformBridge, PlatformType } from './bridge';
import { WebBridge } from './web';
import { VSCodeBridge } from './vscode';

/** 桥接实例（单例） */
let bridge: PlatformBridge | null = null;

/** 检测当前平台 */
export function detectPlatform(): PlatformType {
  // VS Code Webview 环境会注入 acquireVsCodeApi
  if (typeof window !== 'undefined' && 'acquireVsCodeApi' in window) {
    return 'vscode';
  }
  return 'web';
}

/** 获取平台桥接实例（单例） */
export function getPlatformBridge(): PlatformBridge {
  if (!bridge) {
    const platform = detectPlatform();
    bridge = platform === 'vscode' ? new VSCodeBridge() : new WebBridge();
  }
  return bridge;
}

/** 重置桥接实例（仅用于测试） */
export function resetPlatformBridge(): void {
  bridge = null;
}

/** 是否在 VS Code 环境中 */
export function isVSCode(): boolean {
  return detectPlatform() === 'vscode';
}

/** 是否在 Web 环境中 */
export function isWeb(): boolean {
  return detectPlatform() === 'web';
}

// 导出类型
export type { PlatformBridge, PlatformType, ApiOptions, DialogOptions, OutputType, NotificationType } from './bridge';
export type * from './messages';

// 导出实现类（用于测试或特殊场景）
export { WebBridge } from './web';
export { VSCodeBridge } from './vscode';
