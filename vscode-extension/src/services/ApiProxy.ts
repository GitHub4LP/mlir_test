/**
 * API 代理服务
 * 
 * 封装后端 API 调用，从 BackendManager 获取端口。
 */

import { BackendManager } from './BackendManager';

export interface ApiOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  body?: unknown;
  headers?: Record<string, string>;
}

export class ApiProxy {
  constructor(private backendManager: BackendManager) {}

  /** 获取后端 URL */
  private getBackendUrl(): string {
    return this.backendManager.getUrl();
  }

  /** 检查后端连接 */
  async checkConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.getBackendUrl()}/api/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /** 调用后端 API */
  async callApi<T>(endpoint: string, options: ApiOptions = {}): Promise<T> {
    const { method = 'GET', body, headers = {} } = options;
    const url = `${this.getBackendUrl()}/api${endpoint}`;
    
    console.log(`[ApiProxy] ${method} ${url}`, body ? JSON.stringify(body) : '');
    
    const response = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json', ...headers },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[ApiProxy] Error: ${response.status} ${response.statusText}`, errorText);
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json() as Promise<T>;
  }
}
