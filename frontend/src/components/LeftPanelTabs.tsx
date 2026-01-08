/**
 * LeftPanelTabs 组件
 * 
 * 左侧面板 Tab 容器，管理"节点"和"类型"两个 Tab 的切换。
 * - 节点 Tab：显示 NodePalette（函数管理 + 方言操作）
 * - 类型 Tab：显示 TypeConstraintPanel（类型约束浏览器）
 */

import { useState, useCallback } from 'react';
import type { OperationDef, FunctionDef } from '../types';
import { NodePalette } from './NodePalette';
import { TypeConstraintPanel } from './TypeConstraintPanel';

type TabId = 'nodes' | 'types';

export interface LeftPanelTabsProps {
  /** NodePalette: 操作拖拽回调 */
  onDragStart?: (event: React.DragEvent, operation: OperationDef) => void;
  /** NodePalette: 函数拖拽回调 */
  onFunctionDragStart?: (event: React.DragEvent, func: FunctionDef) => void;
  /** NodePalette: 函数选择回调 */
  onFunctionSelect?: (functionName: string) => void;
  /** NodePalette: 函数删除后回调 */
  onFunctionDeleted?: (functionName: string) => void;
}

const tabs: { id: TabId; label: string }[] = [
  { id: 'nodes', label: '节点' },
  { id: 'types', label: '类型' },
];

export function LeftPanelTabs({
  onDragStart,
  onFunctionDragStart,
  onFunctionSelect,
  onFunctionDeleted,
}: LeftPanelTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>('nodes');

  const handleTabClick = useCallback((tabId: TabId) => {
    setActiveTab(tabId);
  }, []);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Tab 切换栏 */}
      <div className="flex border-b border-gray-700 flex-shrink-0">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => handleTabClick(tab.id)}
            className={`flex-1 px-4 py-1.5 text-xs font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-gray-700 text-white border-b-2 border-blue-500'
                : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab 内容 */}
      <div className="flex-1 min-h-0 flex flex-col">
        {activeTab === 'nodes' ? (
          <NodePalette
            onDragStart={onDragStart}
            onFunctionDragStart={onFunctionDragStart}
            onFunctionSelect={onFunctionSelect}
            onFunctionDeleted={onFunctionDeleted}
          />
        ) : (
          <TypeConstraintPanel />
        )}
      </div>
    </div>
  );
}

export default LeftPanelTabs;
