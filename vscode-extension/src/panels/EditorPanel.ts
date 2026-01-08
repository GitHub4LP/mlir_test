/**
 * 节点编辑器 WebviewPanel
 * 
 * 承载 React 节点编辑器，处理：
 * - API 代理
 * - 消息传递
 * - MLIR 代码预览
 * - 文件对话框
 * - 操作数据解析（用于 TreeView 拖放）
 * - 文件关联：打开 .mlir.json 文件时加载对应函数
 * - 文件监听：响应文件系统变化
 */

import * as vscode from 'vscode';
import { ApiProxy } from '../services/ApiProxy';
import { StateManager } from '../services/StateManager';
import { OperationsTreeProvider } from '../views/OperationsProvider';
import { getWebviewContent, getPlaceholderContent } from '../utils/webview';
import * as fs from 'fs';
import * as path from 'path';

export class EditorPanel {
  public static current: EditorPanel | undefined;
  // 静态引用，用于访问 OperationsTreeProvider
  private static operationsProvider: OperationsTreeProvider | undefined;
  
  private readonly panel: vscode.WebviewPanel;
  private disposables: vscode.Disposable[] = [];
  private pendingProjectPath: string | null = null;
  private pendingFunctionName: string | null = null;
  private currentProjectPath: string | null = null;
  private webviewReady = false;

  private constructor(
    panel: vscode.WebviewPanel,
    private extensionUri: vscode.Uri,
    private apiProxy: ApiProxy,
    private stateManager: StateManager,
    private outputChannel: vscode.OutputChannel,
    private backendPort: number
  ) {
    this.panel = panel;
    this.updateContent();
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage(this.handleMessage.bind(this), null, this.disposables);
  }

  /**
   * 设置 OperationsTreeProvider 引用
   * 用于处理 Webview 的 resolveOperationData 请求
   */
  public static setOperationsProvider(provider: OperationsTreeProvider): void {
    EditorPanel.operationsProvider = provider;
  }

  /** 创建或显示面板 */
  public static createOrShow(
    extensionUri: vscode.Uri,
    apiProxy: ApiProxy,
    stateManager: StateManager,
    outputChannel: vscode.OutputChannel,
    backendPort: number
  ): void {
    if (EditorPanel.current) {
      EditorPanel.current.panel.reveal();
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      'mlirBlueprint.editorPanel',
      'MLIR Blueprint',
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')]
      }
    );

    EditorPanel.current = new EditorPanel(
      panel, 
      extensionUri, 
      apiProxy, 
      stateManager, 
      outputChannel,
      backendPort
    );
  }

  /** 更新 Webview 内容 */
  private updateContent(): void {
    // 检查资源是否存在
    const mediaPath = vscode.Uri.joinPath(this.extensionUri, 'media', 'editor.js').fsPath;
    if (!fs.existsSync(mediaPath)) {
      this.panel.webview.html = getPlaceholderContent(
        'Editor resources not found. Please run "npm run build:vscode" in the frontend directory.'
      );
      return;
    }
    
    this.panel.webview.html = getWebviewContent(
      this.panel.webview, 
      this.extensionUri, 
      'editor',
      this.backendPort
    );
  }

  /** 处理来自 Webview 的消息 */
  private async handleMessage(message: { type: string; requestId?: string; payload?: unknown }): Promise<void> {
    const { type, requestId, payload } = message;

    try {
      let result: unknown;

      switch (type) {
        case 'callApi': {
          const { endpoint, method, body } = payload as { endpoint: string; method: string; body?: unknown };
          result = await this.apiProxy.callApi(endpoint, { method: method as 'GET' | 'POST', body });
          break;
        }

        case 'showMlirCode': {
          const { code, verified } = payload as { code: string; verified: boolean };
          await this.showMlirCode(code, verified);
          break;
        }

        case 'appendOutput': {
          const { message: msg, type: logType } = payload as { message: string; type: string };
          this.outputChannel.appendLine(`[${logType}] ${msg}`);
          this.outputChannel.show(true);
          break;
        }

        case 'clearOutput':
          this.outputChannel.clear();
          break;

        case 'selectionChanged':
          this.stateManager.setSelection(payload);
          break;

        case 'showOpenDialog': {
          result = await this.showDialog('open', payload as Record<string, unknown>);
          break;
        }

        case 'showSaveDialog': {
          result = await this.showDialog('save', payload as Record<string, unknown>);
          break;
        }

        case 'showNotification': {
          const { message: notifMsg, type: notifType } = payload as { message: string; type: string };
          this.showNotification(notifMsg, notifType);
          break;
        }

        case 'ready':
          // Webview 就绪，发送后端端口
          this.webviewReady = true;
          this.panel.webview.postMessage({ 
            type: 'backendReady', 
            payload: { port: this.backendPort, url: `http://localhost:${this.backendPort}` } 
          });
          // 如果有待加载的项目，现在加载
          if (this.pendingProjectPath) {
            this.outputChannel.appendLine(`[info] Webview ready, loading pending project: ${this.pendingProjectPath}`);
            if (this.pendingFunctionName) {
              // 打开项目并切换到指定函数
              this.panel.webview.postMessage({ 
                type: 'openProjectAndFunction', 
                payload: { path: this.pendingProjectPath, functionName: this.pendingFunctionName } 
              });
              this.pendingFunctionName = null;
            } else {
              this.panel.webview.postMessage({ type: 'openProject', payload: { path: this.pendingProjectPath } });
            }
            this.currentProjectPath = this.pendingProjectPath;
            this.pendingProjectPath = null;
          }
          break;

        case 'resolveOperationData': {
          // 处理 Webview 的操作数据解析请求（用于 TreeView 拖放）
          const { fullName } = payload as { fullName: string };
          if (EditorPanel.operationsProvider) {
            result = EditorPanel.operationsProvider.getOperationByFullName(fullName);
          } else {
            result = null;
          }
          break;
        }

        case 'updateTitle': {
          // 更新编辑器标题
          const { functionName, isDirty } = payload as { functionName: string; isDirty: boolean };
          this.updateTitle(functionName, isDirty);
          break;
        }
      }

      if (requestId) {
        this.panel.webview.postMessage({ type: 'response', requestId, payload: result });
      }
    } catch (error) {
      if (requestId) {
        this.panel.webview.postMessage({ 
          type: 'error', 
          requestId, 
          payload: error instanceof Error ? error.message : 'Unknown error' 
        });
      }
    }
  }

  /** 显示 MLIR 代码 */
  private async showMlirCode(code: string, verified: boolean): Promise<void> {
    const doc = await vscode.workspace.openTextDocument({ 
      content: code, 
      language: 'mlir' 
    });
    await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
    
    if (verified) {
      this.outputChannel.appendLine('[info] MLIR code verified successfully');
    }
  }

  /** 显示文件对话框 */
  private async showDialog(
    dialogType: 'open' | 'save', 
    options: Record<string, unknown>
  ): Promise<string | null> {
    if (dialogType === 'open') {
      const uri = await vscode.window.showOpenDialog(options as vscode.OpenDialogOptions);
      return uri?.[0]?.fsPath ?? null;
    } else {
      const uri = await vscode.window.showSaveDialog(options as vscode.SaveDialogOptions);
      return uri?.fsPath ?? null;
    }
  }

  /** 显示通知 */
  private showNotification(message: string, type: string): void {
    switch (type) {
      case 'error': 
        vscode.window.showErrorMessage(message); 
        break;
      case 'warning': 
        vscode.window.showWarningMessage(message); 
        break;
      default: 
        vscode.window.showInformationMessage(message);
    }
  }

  /** 创建项目 */
  public createProject(folderPath: string): void {
    this.panel.webview.postMessage({ type: 'createProject', payload: { path: folderPath } });
  }

  /** 打开项目 */
  public openProject(filePath: string): void {
    if (this.webviewReady) {
      this.panel.webview.postMessage({ type: 'openProject', payload: { path: filePath } });
      this.currentProjectPath = filePath;
    } else {
      // Webview 还没准备好，保存路径等待
      this.outputChannel.appendLine(`[info] Webview not ready, queuing project: ${filePath}`);
      this.pendingProjectPath = filePath;
      this.pendingFunctionName = null;
    }
  }

  /** 打开项目并切换到指定函数 */
  public openProjectAndFunction(projectPath: string, functionName: string): void {
    if (this.webviewReady) {
      // 如果是同一个项目，只切换函数
      if (this.currentProjectPath === projectPath) {
        this.panel.webview.postMessage({ type: 'selectFunction', payload: { functionName } });
      } else {
        this.panel.webview.postMessage({ 
          type: 'openProjectAndFunction', 
          payload: { path: projectPath, functionName } 
        });
        this.currentProjectPath = projectPath;
      }
    } else {
      // Webview 还没准备好，保存路径等待
      this.outputChannel.appendLine(`[info] Webview not ready, queuing project: ${projectPath}, function: ${functionName}`);
      this.pendingProjectPath = projectPath;
      this.pendingFunctionName = functionName;
    }
  }

  /** 通知文件变化 */
  public notifyFileChange(type: 'created' | 'deleted', filePath: string): void {
    if (!this.webviewReady || !this.currentProjectPath) return;
    
    // 检查文件是否在当前项目目录下
    const fileDir = path.dirname(filePath);
    if (fileDir !== this.currentProjectPath) return;
    
    const fileName = path.basename(filePath);
    if (!fileName.endsWith('.mlir.json')) return;
    
    const functionName = fileName.replace('.mlir.json', '');
    
    this.panel.webview.postMessage({ 
      type: 'fileChange', 
      payload: { changeType: type, functionName, filePath } 
    });
  }

  /** 通知文件重命名 */
  public notifyFileRename(oldName: string, newName: string, newPath: string): void {
    if (!this.webviewReady || !this.currentProjectPath) return;
    
    // 检查文件是否在当前项目目录下
    const fileDir = path.dirname(newPath);
    if (fileDir !== this.currentProjectPath) return;
    
    this.panel.webview.postMessage({ 
      type: 'fileRename', 
      payload: { oldName, newName } 
    });
  }

  /** 请求 MLIR 预览 */
  public requestMlirPreview(): void {
    this.panel.webview.postMessage({ type: 'requestMlirPreview' });
  }

  /** 请求保存项目 */
  public requestSaveProject(): void {
    this.panel.webview.postMessage({ type: 'requestSaveProject' });
  }

  /** 添加函数 */
  public addFunction(name: string): void {
    this.panel.webview.postMessage({ type: 'addFunction', payload: { name } });
  }

  /** 重命名函数 */
  public renameFunction(functionName: string, newName: string): void {
    this.panel.webview.postMessage({ type: 'renameFunction', payload: { functionName, newName } });
  }

  /** 删除函数 */
  public deleteFunction(functionName: string): void {
    this.panel.webview.postMessage({ type: 'deleteFunction', payload: { functionName } });
  }

  /** 选择函数（切换到该函数进行编辑） */
  public selectFunction(functionName: string): void {
    this.panel.webview.postMessage({ type: 'selectFunction', payload: { functionName } });
  }

  /** 切换到指定函数（用于 Custom Editor 拦截） */
  public switchToFunction(projectPath: string, functionName: string): void {
    if (this.webviewReady) {
      // 如果是同一个项目，只切换函数
      if (this.currentProjectPath === projectPath) {
        this.panel.webview.postMessage({ type: 'selectFunction', payload: { functionName } });
      } else {
        // 不同项目，先加载项目再切换函数
        this.panel.webview.postMessage({ 
          type: 'openProjectAndFunction', 
          payload: { path: projectPath, functionName } 
        });
        this.currentProjectPath = projectPath;
      }
      // 确保编辑器面板可见
      this.panel.reveal();
    } else {
      // Webview 还没准备好，保存路径等待
      this.outputChannel.appendLine(`[info] Webview not ready, queuing: ${projectPath}, function: ${functionName}`);
      this.pendingProjectPath = projectPath;
      this.pendingFunctionName = functionName;
    }
  }

  /** 保存当前函数 */
  public saveCurrentFunction(): void {
    if (this.webviewReady) {
      this.panel.webview.postMessage({ type: 'saveCurrentFunction' });
    }
  }

  /** 更新标题 */
  public updateTitle(functionName: string, isDirty: boolean): void {
    const dirtyIndicator = isDirty ? ' •' : '';
    this.panel.title = `MLIR Blueprint - ${functionName}${dirtyIndicator}`;
  }

  /** 检查编辑器是否处于活动状态 */
  public isActive(): boolean {
    return this.panel.active;
  }

  /** 释放资源 */
  private dispose(): void {
    EditorPanel.current = undefined;
    this.panel.dispose();
    this.disposables.forEach(d => d.dispose());
  }
}
