/**
 * 跨视图状态管理服务
 * 
 * 管理选中状态，在 EditorPanel 和 PropertiesView 之间同步。
 */

import * as vscode from 'vscode';

/** 选中状态 */
export interface SelectionState {
  nodeIds: string[];
  nodeData: unknown;
}

export class StateManager {
  private _onSelectionChanged = new vscode.EventEmitter<SelectionState>();
  readonly onSelectionChanged = this._onSelectionChanged.event;

  private selection: SelectionState = { nodeIds: [], nodeData: null };

  /** 设置选中状态 */
  setSelection(data: unknown): void {
    this.selection = data as SelectionState;
    this._onSelectionChanged.fire(this.selection);
  }

  /** 获取当前选中状态 */
  getSelection(): SelectionState {
    return this.selection;
  }

  /** 清空选中状态 */
  clearSelection(): void {
    this.selection = { nodeIds: [], nodeData: null };
    this._onSelectionChanged.fire(this.selection);
  }

  /** 释放资源 */
  dispose(): void {
    this._onSelectionChanged.dispose();
  }
}
