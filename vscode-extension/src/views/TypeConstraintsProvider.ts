/**
 * 类型约束浏览器 TreeDataProvider
 * 
 * 显示类型约束列表，支持：
 * - 按类别分组（Constraints / Types）
 * - 展开显示具体类型
 */

import * as vscode from 'vscode';
import { ApiProxy } from '../services/ApiProxy';

/** 约束定义 */
interface ConstraintDef {
  name: string;
  summary: string;
  rule: unknown;
  dialect: string | null;
}

/** 类型定义 */
interface TypeDef {
  name: string;
  typeName: string;
  dialect: string;
  summary: string;
  isScalar: boolean;
}

/** API 响应 */
interface TypesResponse {
  constraintDefinitions: ConstraintDef[];
  typeDefinitions: TypeDef[];
  constraintEquivalences: Record<string, string[]>;
  containerTypes: unknown[];
}

/** 树节点类型 */
type TreeItemType = 'category' | 'constraint' | 'type';

/** 树节点 */
class TypeConstraintTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    description: string,
    collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly itemType: TreeItemType,
    public readonly data?: unknown
  ) {
    super(label, collapsibleState);
    this.description = description;
    this.tooltip = `${label}${description ? ': ' + description : ''}`;
    this.contextValue = itemType;
  }
}

export class TypeConstraintsTreeProvider implements vscode.TreeDataProvider<TypeConstraintTreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<TypeConstraintTreeItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private constraintDefs: ConstraintDef[] = [];
  private typeDefs: TypeDef[] = [];
  private equivalences: Record<string, string[]> = {};

  constructor(private apiProxy: ApiProxy) {}

  /** 刷新数据 */
  async refresh(): Promise<void> {
    try {
      const data = await this.apiProxy.callApi<TypesResponse>('/types/');
      this.constraintDefs = data.constraintDefinitions || [];
      this.typeDefs = data.typeDefinitions || [];
      this.equivalences = data.constraintEquivalences || {};
      
      this._onDidChangeTreeData.fire(undefined);
    } catch (error) {
      console.error('Failed to load type constraints:', error);
      vscode.window.showErrorMessage(`Failed to load type constraints: ${error}`);
    }
  }

  getTreeItem(element: TypeConstraintTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: TypeConstraintTreeItem): Promise<TypeConstraintTreeItem[]> {
    if (!element) {
      // 根级别：显示两个类别
      return [
        new TypeConstraintTreeItem(
          'Constraints',
          `${this.constraintDefs.length} items`,
          vscode.TreeItemCollapsibleState.Collapsed,
          'category',
          'constraints'
        ),
        new TypeConstraintTreeItem(
          'Scalar Types',
          `${this.typeDefs.filter(t => t.isScalar).length} items`,
          vscode.TreeItemCollapsibleState.Collapsed,
          'category',
          'scalar-types'
        ),
        new TypeConstraintTreeItem(
          'Compound Types',
          `${this.typeDefs.filter(t => !t.isScalar).length} items`,
          vscode.TreeItemCollapsibleState.Collapsed,
          'category',
          'compound-types'
        ),
      ];
    }

    if (element.itemType === 'category') {
      const categoryId = element.data as string;
      
      if (categoryId === 'constraints') {
        return this.constraintDefs.map(c => new TypeConstraintTreeItem(
          c.name,
          c.summary || '',
          vscode.TreeItemCollapsibleState.None,
          'constraint',
          c
        ));
      }
      
      if (categoryId === 'scalar-types') {
        return this.typeDefs
          .filter(t => t.isScalar)
          .map(t => new TypeConstraintTreeItem(
            t.name,
            t.summary || '',
            vscode.TreeItemCollapsibleState.None,
            'type',
            t
          ));
      }
      
      if (categoryId === 'compound-types') {
        return this.typeDefs
          .filter(t => !t.isScalar)
          .map(t => new TypeConstraintTreeItem(
            t.name,
            t.summary || '',
            vscode.TreeItemCollapsibleState.None,
            'type',
            t
          ));
      }
    }

    return [];
  }
}
