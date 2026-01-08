/**
 * 平台桥接接口
 * 
 * 定义平台相关操作的抽象接口，由 WebBridge 和 VSCodeBridge 实现。
 */

/** 平台类型 */
export type PlatformType = 'web' | 'vscode';

/** API 调用选项 */
export interface ApiOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  body?: unknown;
  headers?: Record<string, string>;
}

/** 对话框选项 */
export interface DialogOptions {
  title?: string;
  filters?: Array<{ name: string; extensions: string[] }>;
  canSelectFolders?: boolean;
}

/** 输出日志类型 */
export type OutputType = 'info' | 'success' | 'error' | 'output';

/** 通知类型 */
export type NotificationType = 'info' | 'warning' | 'error';

/** 平台桥接接口 */
export interface PlatformBridge {
  /** 平台类型 */
  readonly platform: PlatformType;

  /** 调用后端 API */
  callApi<T>(endpoint: string, options?: ApiOptions): Promise<T>;

  /** 显示 MLIR 代码 */
  showMlirCode(code: string, verified: boolean): Promise<void>;

  /** 追加输出日志 */
  appendOutput(message: string, type?: OutputType): void;

  /** 清空输出 */
  clearOutput(): void;

  /** 显示打开对话框 */
  showOpenDialog(options?: DialogOptions): Promise<string | null>;

  /** 显示保存对话框 */
  showSaveDialog(options?: DialogOptions): Promise<string | null>;

  /** 显示通知 */
  showNotification(message: string, type: NotificationType): void;

  /** 通知选中节点变化（跨视图同步） */
  notifySelectionChanged(nodeIds: string[], nodeData: unknown): void;

  /** 监听属性变化（从属性面板） */
  onPropertyChanged(callback: (nodeId: string, property: string, value: unknown) => void): () => void;

  /** 监听后端就绪（VS Code 模式） */
  onBackendReady?(callback: (port: number, url: string) => void): () => void;

  /**
   * 根据 fullName 获取操作数据
   * 
   * VS Code TreeView 拖放到 Webview 时，从 text/uri-list 提取 fullName，
   * 此方法通过消息请求扩展查找完整的操作定义。
   * 
   * @param fullName 操作的完整名称，如 "arith.addi"
   * @returns 完整的操作定义，如果未找到返回 null
   */
  resolveOperationData(fullName: string): Promise<unknown | null>;

  /**
   * 根据 functionName 获取函数数据
   * 
   * VS Code TreeView 拖放到 Webview 时，从 text/uri-list 提取 functionName，
   * 此方法通过消息请求扩展查找函数信息。
   * 
   * @param functionName 函数名
   * @returns 函数信息，如果未找到返回 null
   */
  resolveFunctionData(functionName: string): Promise<unknown | null>;

  /**
   * 通知扩展更新函数列表
   * 
   * 当项目加载或函数列表变化时调用，同步到 VS Code TreeView
   */
  updateFunctions?(functions: unknown[], currentFunctionName: string | null): void;
}
