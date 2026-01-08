/**
 * FunctionEditorProvider
 * 
 * Custom Editor Provider 用于拦截 .mlir.json 文件打开。
 * 
 * 设计：
 * - 不创建新的 webview，而是切换现有的 EditorPanel
 * - 打开后立即关闭 Custom Editor 标签
 * - 实现单编辑器架构
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { EditorPanel } from '../panels/EditorPanel';
import { ApiProxy } from '../services/ApiProxy';
import { StateManager } from '../services/StateManager';

/**
 * 简单的 CustomDocument 实现
 * 只用于拦截文件打开，不存储实际数据
 */
class FunctionDocument implements vscode.CustomDocument {
  constructor(
    public readonly uri: vscode.Uri,
    public readonly functionName: string,
    public readonly projectPath: string
  ) {}

  dispose(): void {
    // 无需清理
  }
}

/**
 * Function Editor Provider
 * 
 * 拦截 .mlir.json 文件打开，切换到节点编辑器
 */
export class FunctionEditorProvider implements vscode.CustomReadonlyEditorProvider<FunctionDocument> {
  public static readonly viewType = 'mlirBlueprint.functionEditor';

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly apiProxy: ApiProxy,
    private readonly stateManager: StateManager,
    private readonly outputChannel: vscode.OutputChannel,
    private readonly getBackendPort: () => number
  ) {}

  /**
   * 注册 Custom Editor Provider
   */
  public static register(
    context: vscode.ExtensionContext,
    extensionUri: vscode.Uri,
    apiProxy: ApiProxy,
    stateManager: StateManager,
    outputChannel: vscode.OutputChannel,
    getBackendPort: () => number
  ): vscode.Disposable {
    const provider = new FunctionEditorProvider(
      extensionUri,
      apiProxy,
      stateManager,
      outputChannel,
      getBackendPort
    );

    return vscode.window.registerCustomEditorProvider(
      FunctionEditorProvider.viewType,
      provider,
      {
        webviewOptions: {
          retainContextWhenHidden: false, // 不需要保持，因为我们会立即关闭
        },
        supportsMultipleEditorsPerDocument: false,
      }
    );
  }

  /**
   * 打开文档
   */
  async openCustomDocument(
    uri: vscode.Uri,
    _openContext: vscode.CustomDocumentOpenContext,
    _token: vscode.CancellationToken
  ): Promise<FunctionDocument> {
    const filePath = uri.fsPath;
    const functionName = path.basename(filePath, '.mlir.json');
    const projectPath = path.dirname(filePath);

    this.outputChannel.appendLine(`[FunctionEditorProvider] Opening: ${functionName} from ${projectPath}`);

    return new FunctionDocument(uri, functionName, projectPath);
  }

  /**
   * 解析 Custom Editor
   * 
   * 这里不创建新的 webview，而是：
   * 1. 确保主编辑器存在
   * 2. 切换到目标函数
   * 3. 关闭这个 Custom Editor 标签
   */
  async resolveCustomEditor(
    document: FunctionDocument,
    webviewPanel: vscode.WebviewPanel,
    _token: vscode.CancellationToken
  ): Promise<void> {
    const { functionName, projectPath } = document;

    this.outputChannel.appendLine(`[FunctionEditorProvider] Resolving: ${functionName}`);

    // 1. 确保主编辑器存在
    EditorPanel.createOrShow(
      this.extensionUri,
      this.apiProxy,
      this.stateManager,
      this.outputChannel,
      this.getBackendPort()
    );

    // 2. 切换到目标函数
    EditorPanel.current?.switchToFunction(projectPath, functionName);

    // 3. 关闭这个 Custom Editor 标签
    // 使用 setTimeout 避免与 VS Code 的打开流程冲突
    setTimeout(() => {
      webviewPanel.dispose();
    }, 0);
  }
}
