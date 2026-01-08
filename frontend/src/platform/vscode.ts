/**
 * VS Code 模式平台桥接实现
 * 
 * 通过消息传递与 Extension 通讯。
 */

import type { PlatformBridge, ApiOptions, DialogOptions, OutputType, NotificationType } from './bridge';
import type { Message, BackendReadyMessage, PropertyChangedMessage } from './messages';

/** VS Code API 类型 */
interface VSCodeApi {
  postMessage(message: unknown): void;
  getState(): unknown;
  setState(state: unknown): void;
}

/** 获取 VS Code API（仅在 Webview 中可用） */
declare function acquireVsCodeApi(): VSCodeApi;

export class VSCodeBridge implements PlatformBridge {
  readonly platform = 'vscode' as const;
  
  private vscode: VSCodeApi;
  private requestId = 0;
  private pendingRequests = new Map<string, {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
  }>();
  
  private propertyChangedCallbacks: Array<(nodeId: string, property: string, value: unknown) => void> = [];
  private backendReadyCallbacks: Array<(port: number, url: string) => void> = [];
  private createProjectCallbacks: Array<(path: string) => void> = [];
  private openProjectCallbacks: Array<(path: string) => void> = [];
  private openProjectAndFunctionCallbacks: Array<(path: string, functionName: string) => void> = [];
  private mlirPreviewCallbacks: Array<() => void> = [];
  private saveProjectCallbacks: Array<() => void> = [];
  private addFunctionCallbacks: Array<(name: string) => void> = [];
  private renameFunctionCallbacks: Array<(functionName: string, newName: string) => void> = [];
  private deleteFunctionCallbacks: Array<(functionName: string) => void> = [];
  private selectFunctionCallbacks: Array<(functionName: string) => void> = [];
  private fileChangeCallbacks: Array<(changeType: 'created' | 'deleted', functionName: string) => void> = [];
  private fileRenameCallbacks: Array<(oldName: string, newName: string) => void> = [];
  private saveCurrentFunctionCallbacks: Array<() => void> = [];

  constructor() {
    this.vscode = acquireVsCodeApi();
    window.addEventListener('message', this.handleMessage);
    
    // 通知扩展 Webview 已就绪
    this.post('ready');
  }

  private handleMessage = (event: MessageEvent<Message>) => {
    const message = event.data;
    
    // 处理响应消息
    if (message.requestId && this.pendingRequests.has(message.requestId)) {
      const { resolve, reject } = this.pendingRequests.get(message.requestId)!;
      this.pendingRequests.delete(message.requestId);
      
      if (message.type === 'error') {
        reject(new Error(message.payload as string));
      } else {
        resolve(message.payload);
      }
      return;
    }

    // 处理事件消息
    switch (message.type) {
      case 'propertyChanged': {
        const { nodeId, property, value } = (message as PropertyChangedMessage).payload;
        this.propertyChangedCallbacks.forEach(cb => cb(nodeId, property, value));
        break;
      }
      case 'backendReady': {
        const { port, url } = (message as BackendReadyMessage).payload;
        this.backendReadyCallbacks.forEach(cb => cb(port, url));
        break;
      }
      case 'createProject': {
        const { path } = message.payload as { path: string };
        this.createProjectCallbacks.forEach(cb => cb(path));
        break;
      }
      case 'openProject': {
        const { path } = message.payload as { path: string };
        this.openProjectCallbacks.forEach(cb => cb(path));
        break;
      }
      case 'openProjectAndFunction': {
        const { path, functionName } = message.payload as { path: string; functionName: string };
        this.openProjectAndFunctionCallbacks.forEach(cb => cb(path, functionName));
        break;
      }
      case 'fileChange': {
        const { changeType, functionName } = message.payload as { changeType: 'created' | 'deleted'; functionName: string };
        this.fileChangeCallbacks.forEach(cb => cb(changeType, functionName));
        break;
      }
      case 'fileRename': {
        const { oldName, newName } = message.payload as { oldName: string; newName: string };
        this.fileRenameCallbacks.forEach(cb => cb(oldName, newName));
        break;
      }
      case 'requestMlirPreview': {
        this.mlirPreviewCallbacks.forEach(cb => cb());
        break;
      }
      case 'requestSaveProject': {
        this.saveProjectCallbacks.forEach(cb => cb());
        break;
      }
      case 'addFunction': {
        const { name } = message.payload as { name: string };
        this.addFunctionCallbacks.forEach(cb => cb(name));
        break;
      }
      case 'renameFunction': {
        const { functionName, newName } = message.payload as { functionName: string; newName: string };
        this.renameFunctionCallbacks.forEach(cb => cb(functionName, newName));
        break;
      }
      case 'deleteFunction': {
        const { functionName } = message.payload as { functionName: string };
        this.deleteFunctionCallbacks.forEach(cb => cb(functionName));
        break;
      }
      case 'selectFunction': {
        const { functionName } = message.payload as { functionName: string };
        this.selectFunctionCallbacks.forEach(cb => cb(functionName));
        break;
      }
      case 'saveCurrentFunction': {
        this.saveCurrentFunctionCallbacks.forEach(cb => cb());
        break;
      }
    }
  };


  /** 发送请求并等待响应 */
  private request<T>(type: string, payload?: unknown): Promise<T> {
    return new Promise((resolve, reject) => {
      const id = `req_${++this.requestId}`;
      this.pendingRequests.set(id, { 
        resolve: resolve as (v: unknown) => void, 
        reject 
      });
      
      this.vscode.postMessage({ type, requestId: id, payload });
      
      // 30 秒超时
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`Request ${type} timed out`));
        }
      }, 30000);
    });
  }

  /** 发送消息（不等待响应） */
  private post(type: string, payload?: unknown): void {
    this.vscode.postMessage({ type, payload });
  }

  async callApi<T>(endpoint: string, options: ApiOptions = {}): Promise<T> {
    return this.request<T>('callApi', { 
      endpoint, 
      method: options.method || 'GET',
      body: options.body,
      headers: options.headers,
    });
  }

  async showMlirCode(code: string, verified: boolean): Promise<void> {
    this.post('showMlirCode', { code, verified });
  }

  appendOutput(message: string, type: OutputType = 'info'): void {
    this.post('appendOutput', { message, type });
  }

  clearOutput(): void {
    this.post('clearOutput');
  }

  async showOpenDialog(options?: DialogOptions): Promise<string | null> {
    return this.request<string | null>('showOpenDialog', options);
  }

  async showSaveDialog(options?: DialogOptions): Promise<string | null> {
    return this.request<string | null>('showSaveDialog', options);
  }

  showNotification(message: string, type: NotificationType): void {
    this.post('showNotification', { message, type });
  }

  notifySelectionChanged(nodeIds: string[], nodeData: unknown): void {
    this.post('selectionChanged', { nodeIds, nodeData });
  }

  onPropertyChanged(callback: (nodeId: string, property: string, value: unknown) => void): () => void {
    this.propertyChangedCallbacks.push(callback);
    return () => {
      const index = this.propertyChangedCallbacks.indexOf(callback);
      if (index >= 0) this.propertyChangedCallbacks.splice(index, 1);
    };
  }

  onBackendReady(callback: (port: number, url: string) => void): () => void {
    this.backendReadyCallbacks.push(callback);
    return () => {
      const index = this.backendReadyCallbacks.indexOf(callback);
      if (index >= 0) this.backendReadyCallbacks.splice(index, 1);
    };
  }

  onCreateProject(callback: (path: string) => void): () => void {
    this.createProjectCallbacks.push(callback);
    return () => {
      const index = this.createProjectCallbacks.indexOf(callback);
      if (index >= 0) this.createProjectCallbacks.splice(index, 1);
    };
  }

  onOpenProject(callback: (path: string) => void): () => void {
    this.openProjectCallbacks.push(callback);
    return () => {
      const index = this.openProjectCallbacks.indexOf(callback);
      if (index >= 0) this.openProjectCallbacks.splice(index, 1);
    };
  }

  onOpenProjectAndFunction(callback: (path: string, functionName: string) => void): () => void {
    this.openProjectAndFunctionCallbacks.push(callback);
    return () => {
      const index = this.openProjectAndFunctionCallbacks.indexOf(callback);
      if (index >= 0) this.openProjectAndFunctionCallbacks.splice(index, 1);
    };
  }

  onFileChange(callback: (changeType: 'created' | 'deleted', functionName: string) => void): () => void {
    this.fileChangeCallbacks.push(callback);
    return () => {
      const index = this.fileChangeCallbacks.indexOf(callback);
      if (index >= 0) this.fileChangeCallbacks.splice(index, 1);
    };
  }

  onFileRename(callback: (oldName: string, newName: string) => void): () => void {
    this.fileRenameCallbacks.push(callback);
    return () => {
      const index = this.fileRenameCallbacks.indexOf(callback);
      if (index >= 0) this.fileRenameCallbacks.splice(index, 1);
    };
  }

  onMlirPreviewRequest(callback: () => void): () => void {
    this.mlirPreviewCallbacks.push(callback);
    return () => {
      const index = this.mlirPreviewCallbacks.indexOf(callback);
      if (index >= 0) this.mlirPreviewCallbacks.splice(index, 1);
    };
  }

  onSaveProjectRequest(callback: () => void): () => void {
    this.saveProjectCallbacks.push(callback);
    return () => {
      const index = this.saveProjectCallbacks.indexOf(callback);
      if (index >= 0) this.saveProjectCallbacks.splice(index, 1);
    };
  }

  onAddFunction(callback: (name: string) => void): () => void {
    this.addFunctionCallbacks.push(callback);
    return () => {
      const index = this.addFunctionCallbacks.indexOf(callback);
      if (index >= 0) this.addFunctionCallbacks.splice(index, 1);
    };
  }

  onRenameFunction(callback: (functionName: string, newName: string) => void): () => void {
    this.renameFunctionCallbacks.push(callback);
    return () => {
      const index = this.renameFunctionCallbacks.indexOf(callback);
      if (index >= 0) this.renameFunctionCallbacks.splice(index, 1);
    };
  }

  onDeleteFunction(callback: (functionName: string) => void): () => void {
    this.deleteFunctionCallbacks.push(callback);
    return () => {
      const index = this.deleteFunctionCallbacks.indexOf(callback);
      if (index >= 0) this.deleteFunctionCallbacks.splice(index, 1);
    };
  }

  onSelectFunction(callback: (functionName: string) => void): () => void {
    this.selectFunctionCallbacks.push(callback);
    return () => {
      const index = this.selectFunctionCallbacks.indexOf(callback);
      if (index >= 0) this.selectFunctionCallbacks.splice(index, 1);
    };
  }

  onSaveCurrentFunction(callback: () => void): () => void {
    this.saveCurrentFunctionCallbacks.push(callback);
    return () => {
      const index = this.saveCurrentFunctionCallbacks.indexOf(callback);
      if (index >= 0) this.saveCurrentFunctionCallbacks.splice(index, 1);
    };
  }

  /** 更新编辑器标题（显示当前函数和脏状态） */
  updateTitle(functionName: string, isDirty: boolean): void {
    this.post('updateTitle', { functionName, isDirty });
  }

  async resolveOperationData(fullName: string): Promise<unknown | null> {
    // 通过消息请求扩展查找完整的操作定义
    return this.request<unknown | null>('resolveOperationData', { fullName });
  }

  async resolveFunctionData(functionName: string): Promise<unknown | null> {
    // 通过消息请求扩展查找函数信息
    return this.request<unknown | null>('resolveFunctionData', { functionName });
  }

  updateFunctions(functions: unknown[], currentFunctionName: string | null): void {
    // 通知扩展更新函数列表（用于 TreeView 同步）
    this.post('updateFunctions', { functions, currentFunctionName });
  }

  /** 销毁桥接，清理资源 */
  dispose(): void {
    window.removeEventListener('message', this.handleMessage);
    this.pendingRequests.clear();
    this.propertyChangedCallbacks = [];
    this.backendReadyCallbacks = [];
    this.createProjectCallbacks = [];
    this.openProjectCallbacks = [];
    this.openProjectAndFunctionCallbacks = [];
    this.mlirPreviewCallbacks = [];
    this.saveProjectCallbacks = [];
    this.addFunctionCallbacks = [];
    this.renameFunctionCallbacks = [];
    this.deleteFunctionCallbacks = [];
    this.selectFunctionCallbacks = [];
    this.fileChangeCallbacks = [];
    this.fileRenameCallbacks = [];
    this.saveCurrentFunctionCallbacks = [];
  }
}
