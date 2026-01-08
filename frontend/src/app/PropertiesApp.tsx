/**
 * VS Code 属性面板入口
 * 
 * 仅包含属性面板，用于 VS Code WebviewView。
 * 监听选中状态消息，显示节点属性。
 */

import { StrictMode, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import type { Node } from '@xyflow/react';
import { PropertiesPanel } from '../components/layout/PropertiesPanel';

import '../index.css';

/** 选中状态 */
interface SelectionState {
  nodeIds: string[];
  nodeData: unknown;
}

/** 属性面板应用 */
function PropertiesApp() {
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [selectedCount, setSelectedCount] = useState(0);
  
  // 监听来自扩展的消息
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const message = event.data;
      
      if (message.type === 'selectionChanged') {
        const selection = message.payload as SelectionState;
        setSelectedCount(selection.nodeIds.length);
        
        if (selection.nodeIds.length === 1 && Array.isArray(selection.nodeData)) {
          // 单选，显示节点属性
          const nodeData = selection.nodeData[0];
          setSelectedNode(nodeData as Node);
        } else {
          setSelectedNode(null);
        }
      }
    };
    
    window.addEventListener('message', handleMessage);
    
    // 通知扩展 Webview 已就绪
    if (typeof acquireVsCodeApi !== 'undefined') {
      const vscode = acquireVsCodeApi();
      vscode.postMessage({ type: 'ready' });
    }
    
    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, []);
  
  return (
    <div className="w-full h-full bg-gray-800 text-white">
      <PropertiesPanel 
        selectedNode={selectedNode}
        selectedCount={selectedCount}
      />
      
      {selectedCount === 0 && (
        <div className="p-4 text-gray-400 text-sm">
          Select a node to view its properties
        </div>
      )}
    </div>
  );
}

// VS Code API 类型声明
declare function acquireVsCodeApi(): {
  postMessage(message: unknown): void;
  getState(): unknown;
  setState(state: unknown): void;
};

// Mount app
const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(
    <StrictMode>
      <PropertiesApp />
    </StrictMode>
  );
}

export default PropertiesApp;
