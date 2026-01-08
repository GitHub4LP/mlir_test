/**
 * 操作浏览器 TreeDataProvider
 * 
 * 显示 MLIR 方言和操作列表，支持：
 * - 按方言分组
 * - 懒加载操作列表
 * - 拖拽到编辑器（通过 TreeDragAndDropController）
 */

import * as vscode from 'vscode';
import { ApiProxy } from '../services/ApiProxy';

// ============================================================
// 类型定义（与后端 API 返回结构一致）
// ============================================================

/** 枚举选项 */
interface EnumOption {
  str: string;
  symbol: string;
  value: number;
  summary: string;
}

/** 参数定义 */
interface ArgumentDef {
  name: string;
  kind: 'operand' | 'attribute';
  typeConstraint: string;
  displayName: string;
  description: string;
  isOptional: boolean;
  isVariadic: boolean;
  enumOptions?: EnumOption[];
  defaultValue?: string;
  allowedTypes?: string[];
}

/** 结果定义 */
interface ResultDef {
  name: string;
  typeConstraint: string;
  displayName: string;
  description: string;
  isVariadic: boolean;
  allowedTypes?: string[];
}

/** Block 参数定义 */
interface BlockArgDef {
  name: string;
  typeConstraint: string;
  sourceOperand?: string;
}

/** Region 定义 */
interface RegionDef {
  name: string;
  isVariadic: boolean;
  blockArgs: BlockArgDef[];
  hasYieldInputs: boolean;
}


/** 完整操作定义（与后端 OperationDef 对应） */
interface OperationDef {
  dialect: string;
  opName: string;
  fullName: string;
  summary: string;
  description: string;
  arguments: ArgumentDef[];
  results: ResultDef[];
  regions: RegionDef[];
  traits: string[];
  assemblyFormat: string;
  hasRegions: boolean;
  isTerminator: boolean;
  isPure: boolean;
}

/** 方言信息（API 返回） */
interface DialectInfo {
  name: string;
  operations: OperationDef[];
}

// ============================================================
// TreeView 节点
// ============================================================

/** 树节点类型 */
type TreeItemType = 'dialect' | 'operation';

/** 树节点 */
class OperationTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    tooltipText: string,
    collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly itemType: TreeItemType,
    public readonly fullName?: string
  ) {
    super(label, collapsibleState);
    // 不显示 description，只在 tooltip 显示
    this.tooltip = tooltipText || label;
    this.contextValue = itemType;
    
    // 操作节点支持拖拽
    if (itemType === 'operation' && fullName) {
      this.resourceUri = vscode.Uri.parse(`mlir-op://${fullName}`);
    }
  }
}


// ============================================================
// TreeDataProvider + TreeDragAndDropController
// ============================================================

export class OperationsTreeProvider 
  implements vscode.TreeDataProvider<OperationTreeItem>,
             vscode.TreeDragAndDropController<OperationTreeItem> {
  
  private _onDidChangeTreeData = new vscode.EventEmitter<OperationTreeItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  // TreeDragAndDropController 属性
  readonly dragMimeTypes = ['application/vnd.code.tree.mlirblueprint.operationsview'];
  readonly dropMimeTypes: string[] = [];

  private dialects: string[] = [];
  // 存储完整的操作定义（而非简化的 OperationInfo）
  private operationsCache = new Map<string, OperationDef[]>();

  constructor(private apiProxy: ApiProxy) {}

  // ============================================================
  // TreeDragAndDropController 方法
  // ============================================================

  /**
   * 处理拖拽开始
   * 设置 resourceUri 以便 VS Code 传递 text/uri-list
   */
  handleDrag(
    source: readonly OperationTreeItem[],
    dataTransfer: vscode.DataTransfer,
    _token: vscode.CancellationToken
  ): void {
    const item = source[0];
    if (item?.itemType === 'operation' && item.fullName) {
      // 设置自定义 MIME 类型（VS Code 会自动包含 resourceUri 到 text/uri-list）
      dataTransfer.set(
        'application/vnd.code.tree.mlirblueprint.operationsview',
        new vscode.DataTransferItem(item.fullName)
      );
    }
  }

  /**
   * 根据 fullName 获取操作数据
   * 供 EditorPanel 调用以响应 Webview 的 resolveOperationData 请求
   * 
   * @param fullName 操作的完整名称，如 "arith.addi"
   * @returns 完整的操作定义，如果未找到返回 null
   */
  getOperationByFullName(fullName: string): OperationDef | null {
    const dialectName = fullName.split('.')[0];
    const operations = this.operationsCache.get(dialectName);
    return operations?.find(op => op.fullName === fullName) ?? null;
  }


  // ============================================================
  // TreeDataProvider 方法
  // ============================================================

  /** 刷新数据 */
  async refresh(): Promise<void> {
    try {
      // API 返回字符串数组
      const data = await this.apiProxy.callApi<string[]>('/dialects/');
      this.dialects = data || [];
      this.operationsCache.clear();
      this._onDidChangeTreeData.fire(undefined);
    } catch (error) {
      console.error('Failed to load dialects:', error);
      vscode.window.showErrorMessage(`Failed to load dialects: ${error}`);
    }
  }

  getTreeItem(element: OperationTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: OperationTreeItem): Promise<OperationTreeItem[]> {
    if (!element) {
      // 根级别：显示方言列表
      return this.dialects.map(name => new OperationTreeItem(
        name,
        '',
        vscode.TreeItemCollapsibleState.Collapsed,
        'dialect'
      ));
    }

    if (element.itemType === 'dialect') {
      // 方言级别：显示操作列表
      const dialectName = element.label as string;
      let operations = this.operationsCache.get(dialectName);
      
      if (!operations) {
        try {
          // API 返回 { name, operations: OperationDef[], typeConstraints }
          const data = await this.apiProxy.callApi<DialectInfo>(`/dialects/${dialectName}`);
          operations = data.operations || [];
          // 存储完整的操作数据
          this.operationsCache.set(dialectName, operations);
        } catch (error) {
          console.error(`Failed to load operations for ${dialectName}:`, error);
          return [];
        }
      }

      return operations.map(op => {
        const shortName = op.fullName.split('.').pop() || op.fullName;
        // 移除描述文本前后的换行和空白
        const tooltip = (op.summary || '').trim();
        return new OperationTreeItem(
          shortName,
          tooltip,
          vscode.TreeItemCollapsibleState.None,
          'operation',
          op.fullName
        );
      });
    }

    return [];
  }
}
