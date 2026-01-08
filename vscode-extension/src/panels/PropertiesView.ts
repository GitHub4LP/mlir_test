/**
 * 属性面板 WebviewViewProvider
 * 
 * 显示选中节点的属性，支持：
 * - 监听选中状态变化
 * - 属性编辑
 * - 与 EditorPanel 同步
 */

import * as vscode from 'vscode';
import { StateManager, SelectionState } from '../services/StateManager';
import { getWebviewContent, getPlaceholderContent } from '../utils/webview';
import * as fs from 'fs';

export class PropertiesViewProvider implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;
  private selectionListener?: vscode.Disposable;

  constructor(
    private extensionUri: vscode.Uri,
    private stateManager: StateManager
  ) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this.view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'media')]
    };

    this.updateContent();

    // 监听选中状态变化
    this.selectionListener = this.stateManager.onSelectionChanged(this.handleSelectionChanged.bind(this));

    // 监听来自 Webview 的消息
    webviewView.webview.onDidReceiveMessage(this.handleMessage.bind(this));

    // 清理
    webviewView.onDidDispose(() => {
      this.selectionListener?.dispose();
    });
  }

  /** 更新 Webview 内容 */
  private updateContent(): void {
    if (!this.view) return;

    // 检查资源是否存在
    const mediaPath = vscode.Uri.joinPath(this.extensionUri, 'media', 'properties.js').fsPath;
    if (!fs.existsSync(mediaPath)) {
      this.view.webview.html = getPlaceholderContent(
        'Properties panel resources not found. Please run "npm run build:vscode" in the frontend directory.'
      );
      return;
    }

    this.view.webview.html = getWebviewContent(
      this.view.webview,
      this.extensionUri,
      'properties'
    );
  }

  /** 处理选中状态变化 */
  private handleSelectionChanged(selection: SelectionState): void {
    if (!this.view) return;

    this.view.webview.postMessage({
      type: 'selectionChanged',
      payload: selection
    });
  }

  /** 处理来自 Webview 的消息 */
  private handleMessage(message: { type: string; payload?: unknown }): void {
    const { type, payload } = message;

    switch (type) {
      case 'propertyChanged': {
        // 属性变化，通知 EditorPanel
        const { nodeId, property, value } = payload as { nodeId: string; property: string; value: unknown };
        // 通过 StateManager 或直接发送给 EditorPanel
        // 这里简化处理，实际可能需要更复杂的通讯机制
        console.log('Property changed:', nodeId, property, value);
        break;
      }

      case 'ready':
        // Webview 就绪，发送当前选中状态
        const currentSelection = this.stateManager.getSelection();
        if (currentSelection.nodeIds.length > 0) {
          this.handleSelectionChanged(currentSelection);
        }
        break;
    }
  }

  /** 刷新视图 */
  public refresh(): void {
    this.updateContent();
  }
}
