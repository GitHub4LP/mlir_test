/**
 * Web 模式平台桥接实现
 * 
 * 使用内置 UI 组件和直接 API 调用。
 */

import type { PlatformBridge, ApiOptions, OutputType, NotificationType } from './bridge';
import { useRendererStore } from '../stores/rendererStore';

export class WebBridge implements PlatformBridge {
  readonly platform = 'web' as const;
  private baseUrl: string;

  constructor() {
    // 从环境变量读取 API 基础路径，支持子路径部署
    this.baseUrl = import.meta.env.VITE_API_BASE_URL || '';
  }

  async callApi<T>(endpoint: string, options: ApiOptions = {}): Promise<T> {
    const { method = 'GET', body, headers = {} } = options;
    
    const response = await fetch(`${this.baseUrl}/api${endpoint}`, {
      method,
      headers: { 'Content-Type': 'application/json', ...headers },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  async showMlirCode(code: string, verified: boolean): Promise<void> {
    // Web 模式：更新 rendererStore 并切换到代码视图
    useRendererStore.getState().setMlirCode(code, verified);
    useRendererStore.getState().setViewMode('code');
  }

  appendOutput(message: string, type: OutputType = 'info'): void {
    useRendererStore.getState().addLog(type, message);
  }

  clearOutput(): void {
    useRendererStore.getState().clearLogs();
  }

  async showOpenDialog(): Promise<string | null> {
    // Web 模式使用内置对话框组件，返回 null 表示由 UI 组件处理
    return null;
  }

  async showSaveDialog(): Promise<string | null> {
    // Web 模式使用内置对话框组件
    return null;
  }

  showNotification(message: string, type: NotificationType): void {
    // Web 模式：输出到日志并打印到控制台
    if (type === 'error') {
      console.error(message);
      this.appendOutput(message, 'error');
    } else if (type === 'warning') {
      console.warn(message);
      this.appendOutput(message, 'info');
    } else {
      console.log(message);
      this.appendOutput(message, 'info');
    }
  }

  notifySelectionChanged(): void {
    // Web 模式同一页面，无需跨视图通讯
    // 选中状态通过 editorStore 直接共享
  }

  onPropertyChanged(): () => void {
    // Web 模式同一页面，无需监听
    // 属性变化通过直接调用处理
    return () => {};
  }

  async resolveOperationData(): Promise<unknown | null> {
    // Web 模式不需要此功能，直接返回 null
    // Web 模式下操作数据通过 dataTransfer.getData('application/json') 直接获取
    return null;
  }

  async resolveFunctionData(): Promise<unknown | null> {
    // Web 模式不需要此功能，直接返回 null
    return null;
  }
}
