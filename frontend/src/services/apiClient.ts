/**
 * API 客户端
 * 
 * 提供统一的 API 调用入口，自动适应 Web 和 VS Code 模式。
 * - Web 模式：直接 fetch，支持任意子路径部署
 * - VS Code 模式：通过 PlatformBridge 消息传递
 */

import { getPlatformBridge } from '../platform';

/**
 * 检测当前平台
 */
function detectPlatform(): 'web' | 'vscode' {
  // 检查 VS Code API 是否可用
  if (typeof window !== 'undefined' && 'acquireVsCodeApi' in window) {
    return 'vscode';
  }
  // 检查环境变量
  if (import.meta.env.VITE_PLATFORM === 'vscode') {
    return 'vscode';
  }
  return 'web';
}

const currentPlatform = detectPlatform();

/**
 * 获取 API base URL（仅 Web 模式使用）
 */
function detectApiBaseUrl(): string {
  if (currentPlatform === 'vscode') {
    return ''; // VS Code 模式不使用
  }
  
  const pathname = window.location.pathname;
  let basePath = pathname
    .replace(/\/[^/]*\.[^/]*$/, '')
    .replace(/\/$/, '');
  
  if (basePath && !basePath.startsWith('/')) {
    basePath = '/' + basePath;
  }
  
  return `${basePath}/api`;
}

export const API_BASE_URL = detectApiBaseUrl();

/**
 * 构建完整的 API URL（仅 Web 模式使用）
 */
export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

/**
 * 统一的 API GET 请求
 */
export async function apiGet<T>(path: string): Promise<T> {
  if (currentPlatform === 'vscode') {
    return getPlatformBridge().callApi<T>(path, { method: 'GET' });
  }
  
  const response = await fetch(apiUrl(path));
  if (!response.ok) {
    throw new Error(`API GET ${path} failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * 统一的 API POST 请求
 */
export async function apiPost<T>(path: string, data?: unknown): Promise<T> {
  if (currentPlatform === 'vscode') {
    return getPlatformBridge().callApi<T>(path, { method: 'POST', body: data });
  }
  
  const response = await fetch(apiUrl(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: data ? JSON.stringify(data) : undefined,
  });
  if (!response.ok) {
    throw new Error(`API POST ${path} failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * 封装的 fetch 函数（兼容旧代码）
 * @deprecated 请使用 apiGet 或 apiPost
 */
export async function apiFetch(path: string, options?: RequestInit): Promise<Response> {
  if (currentPlatform === 'vscode') {
    // VS Code 模式下模拟 Response
    const method = options?.method || 'GET';
    const body = options?.body ? JSON.parse(options.body as string) : undefined;
    const result = await getPlatformBridge().callApi(path, { method: method as 'GET' | 'POST', body });
    return new Response(JSON.stringify(result), { status: 200 });
  }
  return fetch(apiUrl(path), options);
}
