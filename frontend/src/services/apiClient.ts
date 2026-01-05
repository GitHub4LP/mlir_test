/**
 * API 客户端
 * 
 * 提供统一的 API 调用入口，自动适应任意子路径部署。
 * 类似 code-server 的设计，无需配置即可在任意路径下工作。
 */

/**
 * 获取 API base URL
 * 
 * 自动检测当前部署路径，支持：
 * - http://localhost:5173/ → /api
 * - http://example.com/mlir-editor/ → /mlir-editor/api
 * - http://example.com/tools/blueprint/ → /tools/blueprint/api
 */
function detectApiBaseUrl(): string {
  // 获取当前页面的 pathname
  const pathname = window.location.pathname;
  
  // 移除文件名部分（如 index.html）和尾部斜杠
  // /mlir-editor/index.html → /mlir-editor
  // /mlir-editor/ → /mlir-editor
  // / → (empty string)
  let basePath = pathname
    .replace(/\/[^/]*\.[^/]*$/, '')  // 移除文件名
    .replace(/\/$/, '');              // 移除尾部斜杠
  
  // 确保以 / 开头（除非是空字符串）
  if (basePath && !basePath.startsWith('/')) {
    basePath = '/' + basePath;
  }
  
  return `${basePath}/api`;
}

/**
 * API base URL
 * 
 * 在应用启动时检测一次，后续复用。
 * 例如：
 * - 根路径部署：/api
 * - 子路径部署：/mlir-editor/api
 */
export const API_BASE_URL = detectApiBaseUrl();

/**
 * 构建完整的 API URL
 * 
 * @param path - API 路径，如 '/dialects/' 或 'dialects/'
 * @returns 完整的 API URL
 * 
 * @example
 * apiUrl('/dialects/') // → '/api/dialects/' 或 '/mlir-editor/api/dialects/'
 * apiUrl('dialects/')  // → '/api/dialects/' 或 '/mlir-editor/api/dialects/'
 */
export function apiUrl(path: string): string {
  // 确保 path 以 / 开头
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

/**
 * 封装的 fetch 函数，自动添加 API base URL
 * 
 * @param path - API 路径
 * @param options - fetch 选项
 * @returns fetch Promise
 * 
 * @example
 * apiFetch('/dialects/').then(res => res.json())
 * apiFetch('/projects/save', { method: 'POST', body: JSON.stringify(data) })
 */
export async function apiFetch(path: string, options?: RequestInit): Promise<Response> {
  return fetch(apiUrl(path), options);
}

/**
 * GET 请求
 */
export async function apiGet<T>(path: string): Promise<T> {
  const response = await apiFetch(path);
  if (!response.ok) {
    throw new Error(`API GET ${path} failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * POST 请求
 */
export async function apiPost<T>(path: string, data?: unknown): Promise<T> {
  const response = await apiFetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: data ? JSON.stringify(data) : undefined,
  });
  if (!response.ok) {
    throw new Error(`API POST ${path} failed: ${response.statusText}`);
  }
  return response.json();
}
