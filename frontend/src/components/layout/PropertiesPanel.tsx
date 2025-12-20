/**
 * Properties panel for editing selected node
 * 
 * Displays node information, position, attributes, and type information.
 */

import type { Node } from '@xyflow/react';
import type { BlueprintNodeData } from '../../types';

export interface PropertiesPanelProps {
  selectedNode: Node | null;
  onNodeUpdate?: (nodeId: string, data: unknown) => void;
}

export function PropertiesPanel({ selectedNode }: PropertiesPanelProps) {
  // 面板只在选中节点时显示，所以 selectedNode 不会为 null
  if (!selectedNode) return null;

  const nodeData = selectedNode.data as BlueprintNodeData | undefined;
  const operation = nodeData?.operation;

  return (
    <div className="p-4 overflow-y-auto h-full">
      <h2 className="text-lg font-semibold text-white mb-4">Properties</h2>

      {/* Node Info */}
      <div className="mb-4 p-3 bg-gray-700 rounded">
        <div className="text-sm text-gray-300">
          <span className="text-gray-500">ID:</span> {selectedNode.id}
        </div>
        {operation && (
          <>
            <div className="text-sm text-gray-300 mt-1">
              <span className="text-gray-500">Operation:</span> {operation.fullName}
            </div>
            {operation.summary && (
              <div className="text-xs text-gray-400 mt-2">
                {operation.summary}
              </div>
            )}
          </>
        )}
      </div>

      {/* Position */}
      <div className="mb-4">
        <h3 className="text-sm font-medium text-gray-300 mb-2">Position</h3>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-500">X</label>
            <input
              type="number"
              value={Math.round(selectedNode.position.x)}
              readOnly
              className="w-full bg-gray-700 text-white text-sm px-2 py-1 rounded border border-gray-600"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">Y</label>
            <input
              type="number"
              value={Math.round(selectedNode.position.y)}
              readOnly
              className="w-full bg-gray-700 text-white text-sm px-2 py-1 rounded border border-gray-600"
            />
          </div>
        </div>
      </div>

      {/* Attributes */}
      {nodeData?.attributes && Object.keys(nodeData.attributes).length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Attributes</h3>
          <div className="space-y-2">
            {Object.entries(nodeData.attributes).map(([key, value]) => (
              <div key={key} className="text-sm">
                <span className="text-gray-500">{key}:</span>{' '}
                <span className="text-gray-300">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Types */}
      {nodeData?.inputTypes && Object.keys(nodeData.inputTypes).length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Input Types</h3>
          <div className="space-y-1">
            {Object.entries(nodeData.inputTypes).map(([port, type]) => (
              <div key={port} className="text-xs flex justify-between">
                <span className="text-gray-400">{port}</span>
                <span className="text-blue-400">{type}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Output Types */}
      {nodeData?.outputTypes && Object.keys(nodeData.outputTypes).length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Output Types</h3>
          <div className="space-y-1">
            {Object.entries(nodeData.outputTypes).map(([port, type]) => (
              <div key={port} className="text-xs flex justify-between">
                <span className="text-gray-400">{port}</span>
                <span className="text-green-400">{type}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
