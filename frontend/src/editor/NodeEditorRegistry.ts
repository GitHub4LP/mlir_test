/**
 * 节点编辑器注册表
 * 
 * 管理已注册的节点编辑器实现
 */

import type { INodeEditor } from './INodeEditor';

/** 编辑器工厂函数类型 */
export type NodeEditorFactory = () => INodeEditor;

/** 编辑器注册信息 */
interface EditorRegistration {
  name: string;
  factory: NodeEditorFactory;
  /** 检查是否可用（可选，默认 true） */
  isAvailable?: () => boolean;
}

/** 已注册的编辑器 */
const registrations: Map<string, EditorRegistration> = new Map();

/** 默认编辑器名称 */
let defaultEditorName: string | null = null;

/**
 * 注册节点编辑器
 */
export function registerNodeEditor(
  name: string,
  factory: NodeEditorFactory,
  options?: { isAvailable?: () => boolean; isDefault?: boolean }
): void {
  registrations.set(name, {
    name,
    factory,
    isAvailable: options?.isAvailable,
  });
  
  if (options?.isDefault || !defaultEditorName) {
    defaultEditorName = name;
  }
}

/**
 * 获取可用的编辑器列表
 */
export function getAvailableEditors(): string[] {
  return Array.from(registrations.entries())
    .filter(([, reg]) => !reg.isAvailable || reg.isAvailable())
    .map(([name]) => name);
}

/**
 * 创建编辑器实例
 */
export function createNodeEditor(name: string): INodeEditor | null {
  const registration = registrations.get(name);
  if (!registration) {
    console.warn(`Node editor "${name}" not found`);
    return null;
  }
  
  if (registration.isAvailable && !registration.isAvailable()) {
    console.warn(`Node editor "${name}" is not available`);
    return null;
  }
  
  return registration.factory();
}

/**
 * 获取默认编辑器名称
 */
export function getDefaultEditorName(): string | null {
  return defaultEditorName;
}

/**
 * 检查编辑器是否已注册
 */
export function hasNodeEditor(name: string): boolean {
  return registrations.has(name);
}

/**
 * 检查编辑器是否可用
 */
export function isNodeEditorAvailable(name: string): boolean {
  const registration = registrations.get(name);
  if (!registration) return false;
  return !registration.isAvailable || registration.isAvailable();
}
