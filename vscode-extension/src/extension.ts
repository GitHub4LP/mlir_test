/**
 * MLIR Blueprint VS Code 扩展入口
 * 
 * 提供：
 * - Operations TreeView（操作浏览器）
 * - Type Constraints TreeView（类型约束浏览器）
 * - Editor WebviewPanel（节点编辑器）
 * - Custom Editor：双击 .mlir.json 切换到节点编辑器
 * - 文件监听：监听 .mlir.json 文件变化
 * - 工作区自动检测（打开包含 main.mlir.json 的文件夹时自动加载）
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { BackendManager } from './services/BackendManager';
import { ApiProxy } from './services/ApiProxy';
import { StateManager } from './services/StateManager';
import { OperationsTreeProvider } from './views/OperationsProvider';
import { TypeConstraintsTreeProvider } from './views/TypeConstraintsProvider';
import { EditorPanel } from './panels/EditorPanel';

let backendManager: BackendManager;
let apiProxy: ApiProxy;
let stateManager: StateManager;
let outputChannel: vscode.OutputChannel;
let operationsProvider: OperationsTreeProvider;
let typeConstraintsProvider: TypeConstraintsTreeProvider;
let fileWatcher: vscode.FileSystemWatcher | undefined;

export async function activate(context: vscode.ExtensionContext) {
  // 创建输出通道
  outputChannel = vscode.window.createOutputChannel('MLIR Blueprint');
  outputChannel.appendLine('MLIR Blueprint extension activating...');
  
  // 初始化服务
  backendManager = new BackendManager(outputChannel);
  stateManager = new StateManager();
  
  // 尝试启动/连接后端
  let backendPort: number;
  try {
    backendPort = await backendManager.start();
    outputChannel.appendLine(`Backend available at port ${backendPort}`);
  } catch (error) {
    outputChannel.appendLine(`Failed to start backend: ${error}`);
    backendPort = 8000; // 使用默认端口
  }
  
  // 初始化 API 代理
  apiProxy = new ApiProxy(backendManager);
  
  // 检查后端连接
  const connected = await apiProxy.checkConnection();
  if (!connected) {
    vscode.window.showWarningMessage(
      'MLIR Blueprint: 无法连接到后端服务',
      '配置后端地址'
    ).then(selection => {
      if (selection) {
        vscode.commands.executeCommand('workbench.action.openSettings', 'mlirBlueprint.backendUrl');
      }
    });
  }

  // 注册 TreeView
  operationsProvider = new OperationsTreeProvider(apiProxy);
  typeConstraintsProvider = new TypeConstraintsTreeProvider(apiProxy);
  
  // 设置 EditorPanel 的 Provider 引用（用于 TreeView 拖放）
  EditorPanel.setOperationsProvider(operationsProvider);
  
  // 使用 createTreeView 注册 Operations TreeView（支持拖放）
  const operationsTreeView = vscode.window.createTreeView('mlirBlueprint.operationsView', {
    treeDataProvider: operationsProvider,
    dragAndDropController: operationsProvider,
  });
  
  // TypeConstraints TreeView 不需要拖放功能
  context.subscriptions.push(
    operationsTreeView,
    vscode.window.registerTreeDataProvider('mlirBlueprint.typeConstraintsView', typeConstraintsProvider)
  );

  // 拦截 .mlir.json 文件打开：立即关闭文本编辑器并切换到节点编辑器
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument(async (document) => {
      const filePath = document.uri.fsPath;
      if (!filePath.endsWith('.mlir.json')) {
        return;
      }
      
      // 延迟一帧，等待编辑器完全打开
      setTimeout(async () => {
        // 关闭刚打开的文本编辑器
        await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
        
        // 打开节点编辑器并切换到对应函数
        await openFunctionFile(context, document.uri);
      }, 10);
    })
  );

  // 注册命令
  context.subscriptions.push(
    vscode.commands.registerCommand('mlirBlueprint.openEditor', () => {
      EditorPanel.createOrShow(context.extensionUri, apiProxy, stateManager, outputChannel, backendManager.getPort() || 8000);
    }),
    
    vscode.commands.registerCommand('mlirBlueprint.createProject', async () => {
      const uri = await vscode.window.showOpenDialog({
        canSelectFolders: true,
        canSelectFiles: false,
        title: '选择项目目录'
      });
      if (uri?.[0]) {
        EditorPanel.createOrShow(context.extensionUri, apiProxy, stateManager, outputChannel, backendManager.getPort() || 8000);
        EditorPanel.current?.createProject(uri[0].fsPath);
      }
    }),
    
    vscode.commands.registerCommand('mlirBlueprint.openProject', async () => {
      const uri = await vscode.window.showOpenDialog({
        canSelectFolders: true,
        canSelectFiles: false,
        title: '打开项目目录'
      });
      if (uri?.[0]) {
        EditorPanel.createOrShow(context.extensionUri, apiProxy, stateManager, outputChannel, backendManager.getPort() || 8000);
        EditorPanel.current?.openProject(uri[0].fsPath);
      }
    }),
    
    vscode.commands.registerCommand('mlirBlueprint.checkConnection', async () => {
      const connected = await apiProxy.checkConnection();
      if (connected) {
        vscode.window.showInformationMessage('MLIR Blueprint: 后端连接正常');
      } else {
        vscode.window.showErrorMessage('MLIR Blueprint: 无法连接到后端服务');
      }
    }),
    
    vscode.commands.registerCommand('mlirBlueprint.refreshOperations', () => {
      operationsProvider.refresh();
    }),
    
    vscode.commands.registerCommand('mlirBlueprint.refreshTypeConstraints', () => {
      typeConstraintsProvider.refresh();
    }),
    
    vscode.commands.registerCommand('mlirBlueprint.previewMlir', async () => {
      if (!EditorPanel.current) {
        vscode.window.showWarningMessage('请先打开编辑器');
        return;
      }
      EditorPanel.current.requestMlirPreview();
    }),
    
    vscode.commands.registerCommand('mlirBlueprint.saveProject', async () => {
      if (!EditorPanel.current) {
        vscode.window.showWarningMessage('请先打开编辑器');
        return;
      }
      EditorPanel.current.requestSaveProject();
    }),
    
    // 创建新函数
    vscode.commands.registerCommand('mlirBlueprint.createFunction', async () => {
      if (!EditorPanel.current) {
        vscode.window.showWarningMessage('请先打开编辑器');
        return;
      }
      
      const name = await vscode.window.showInputBox({
        prompt: '输入函数名',
        placeHolder: 'my_function',
        validateInput: (value) => {
          if (!value) return '函数名不能为空';
          if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(value)) {
            return '函数名必须以字母或下划线开头，只能包含字母、数字和下划线';
          }
          if (value === 'main') return 'main 是保留名称';
          return null;
        }
      });
      
      if (name) {
        EditorPanel.current.addFunction(name);
      }
    }),
    
    // 从文件浏览器右键菜单打开函数
    vscode.commands.registerCommand('mlirBlueprint.openFunction', async (uri: vscode.Uri) => {
      if (!uri) return;
      await openFunctionFile(context, uri);
    })
  );

  // 刷新 TreeView 数据
  if (connected) {
    operationsProvider.refresh();
    typeConstraintsProvider.refresh();
  }
  
  // 设置文件监听
  setupFileWatcher(context);
  
  // 检测工作区是否有 main.mlir.json，自动打开编辑器并加载
  await autoLoadWorkspaceProject(context);
  
  // 监听工作区变化
  context.subscriptions.push(
    vscode.workspace.onDidChangeWorkspaceFolders(async () => {
      setupFileWatcher(context);
      await autoLoadWorkspaceProject(context);
    })
  );
  
  outputChannel.appendLine('MLIR Blueprint extension activated');
}

export function deactivate() {
  outputChannel?.appendLine('MLIR Blueprint extension deactivating...');
  fileWatcher?.dispose();
  backendManager?.stop();
  outputChannel?.dispose();
}

/**
 * 打开 .mlir.json 文件
 */
async function openFunctionFile(
  context: vscode.ExtensionContext,
  uri: vscode.Uri
): Promise<void> {
  const filePath = uri.fsPath;
  const fileName = path.basename(filePath);
  
  // 检查是否是 .mlir.json 文件
  if (!fileName.endsWith('.mlir.json')) {
    return;
  }
  
  // 获取函数名（去掉 .mlir.json 后缀）
  const functionName = fileName.replace('.mlir.json', '');
  
  // 获取项目目录
  const projectPath = path.dirname(filePath);
  
  // 检查项目目录是否有 main.mlir.json
  const mainJsonPath = path.join(projectPath, 'main.mlir.json');
  try {
    await vscode.workspace.fs.stat(vscode.Uri.file(mainJsonPath));
  } catch {
    vscode.window.showWarningMessage('该文件不在 MLIR Blueprint 项目中（缺少 main.mlir.json）');
    return;
  }
  
  outputChannel.appendLine(`Opening function: ${functionName} from ${projectPath}`);
  
  // 创建或显示编辑器
  EditorPanel.createOrShow(
    context.extensionUri,
    apiProxy,
    stateManager,
    outputChannel,
    backendManager.getPort() || 8000
  );
  
  // 打开项目并切换到指定函数
  EditorPanel.current?.openProjectAndFunction(projectPath, functionName);
}

/**
 * 设置文件监听器
 */
function setupFileWatcher(context: vscode.ExtensionContext): void {
  // 清理旧的监听器
  fileWatcher?.dispose();
  
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    return;
  }
  
  // 监听 .mlir.json 文件变化
  fileWatcher = vscode.workspace.createFileSystemWatcher('**/*.mlir.json');
  
  fileWatcher.onDidCreate((uri) => {
    outputChannel.appendLine(`File created: ${uri.fsPath}`);
    // 通知编辑器刷新函数列表
    EditorPanel.current?.notifyFileChange('created', uri.fsPath);
  });
  
  fileWatcher.onDidDelete((uri) => {
    outputChannel.appendLine(`File deleted: ${uri.fsPath}`);
    // 通知编辑器刷新函数列表
    EditorPanel.current?.notifyFileChange('deleted', uri.fsPath);
  });
  
  fileWatcher.onDidChange((_uri) => {
    // 文件内容变化，可能是外部编辑
    // 暂时不处理，避免与编辑器内保存冲突
  });
  
  context.subscriptions.push(fileWatcher);
  
  // 监听文件重命名（通过 onDidRenameFiles）
  context.subscriptions.push(
    vscode.workspace.onDidRenameFiles((event) => {
      for (const file of event.files) {
        const oldPath = file.oldUri.fsPath;
        const newPath = file.newUri.fsPath;
        
        // 只处理 .mlir.json 文件
        if (!oldPath.endsWith('.mlir.json') && !newPath.endsWith('.mlir.json')) {
          continue;
        }
        
        outputChannel.appendLine(`File renamed: ${oldPath} -> ${newPath}`);
        
        const oldName = path.basename(oldPath).replace('.mlir.json', '');
        const newName = path.basename(newPath).replace('.mlir.json', '');
        
        // 通知编辑器处理重命名
        EditorPanel.current?.notifyFileRename(oldName, newName, newPath);
      }
    })
  );
}

/**
 * 检测工作区是否包含 MLIR Blueprint 项目，自动打开编辑器并加载
 */
async function autoLoadWorkspaceProject(
  context: vscode.ExtensionContext
): Promise<void> {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    return;
  }
  
  // 检查第一个工作区文件夹是否有 main.mlir.json
  const workspaceRoot = workspaceFolders[0].uri.fsPath;
  const mainJsonPath = path.join(workspaceRoot, 'main.mlir.json');
  
  try {
    await vscode.workspace.fs.stat(vscode.Uri.file(mainJsonPath));
    
    // 找到 main.mlir.json，自动打开编辑器并加载项目
    outputChannel.appendLine(`Found main.mlir.json in workspace: ${workspaceRoot}`);
    
    // 创建或显示编辑器
    EditorPanel.createOrShow(
      context.extensionUri,
      apiProxy,
      stateManager,
      outputChannel,
      backendManager.getPort() || 8000
    );
    
    // 请求加载项目（EditorPanel 会在 webview 准备好后执行）
    EditorPanel.current?.openProject(workspaceRoot);
    
  } catch {
    // main.mlir.json 不存在，忽略
    outputChannel.appendLine(`No main.mlir.json found in workspace: ${workspaceRoot}`);
  }
}
