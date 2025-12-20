/**
 * Edge color utilities
 * 
 * Provides functions to calculate and update edge colors based on source port types.
 */

import type { Node, Edge } from '@xyflow/react';
import type { BlueprintNodeData, FunctionEntryData, FunctionCallData, DataPin } from '../types';
import { getTypeColor } from '../services/typeSystem';
import { getDisplayType } from '../services/typeSelectorRenderer';
import { PortRef, PortKind } from '../services/port';

/**
 * Gets the display type for a source port
 */
function getSourcePortDisplayType(
  sourceNode: Node,
  sourceHandleId: string
): string | null {
  if (sourceNode.type === 'function-entry') {
    const entryData = sourceNode.data as FunctionEntryData;
    const port = entryData.outputs?.find(p => p.id === sourceHandleId);
    if (port) {
      const dataPin: DataPin = {
        id: port.id,
        label: port.name,
        typeConstraint: port.typeConstraint,
        displayName: port.name,
      };
      return getDisplayType(dataPin, entryData);
    }
  } else if (sourceNode.type === 'function-call') {
    const callData = sourceNode.data as FunctionCallData;
    const port = callData.outputs?.find(p => p.id === sourceHandleId);
    if (port) {
      const dataPin: DataPin = {
        id: port.id,
        label: port.name,
        typeConstraint: port.typeConstraint,
        displayName: port.name,
      };
      return getDisplayType(dataPin, callData);
    }
  } else if (sourceNode.type === 'operation') {
    const nodeData = sourceNode.data as BlueprintNodeData;
    const parsed = PortRef.parseHandleId(sourceHandleId);
    if (nodeData.outputTypes && parsed && parsed.kind === PortKind.DataOut) {
      const portName = parsed.name;
      const operation = nodeData.operation;
      const result = operation.results.find(r => r.name === portName);
      if (result) {
        const dataPin: DataPin = {
          id: sourceHandleId,
          label: result.name,
          typeConstraint: result.typeConstraint,
          displayName: result.displayName || result.name,
        };
        return getDisplayType(dataPin, nodeData);
      }
    }
  }
  return null;
}

/**
 * Gets the color for a data edge based on the source handle type
 */
export function getEdgeColor(
  nodes: Node[],
  sourceNodeId: string,
  sourceHandleId: string | null | undefined
): string {
  if (!sourceHandleId) return '#95A5A6'; // 默认灰色

  const sourceNode = nodes.find(n => n.id === sourceNodeId);
  if (!sourceNode) return '#95A5A6';

  const displayType = getSourcePortDisplayType(sourceNode, sourceHandleId);
  return displayType ? getTypeColor(displayType) : '#95A5A6';
}

/**
 * Updates colors for all data edges based on current node types
 */
export function updateEdgeColors(nodes: Node[], edges: Edge[]): Edge[] {
  return edges.map(edge => {
    if (edge.type === 'execution' || !edge.sourceHandle) {
      return edge;
    }

    const sourceNode = nodes.find(n => n.id === edge.source);
    if (!sourceNode) return edge;

    const displayType = getSourcePortDisplayType(sourceNode, edge.sourceHandle);
    const newColor = displayType ? getTypeColor(displayType) : '#95A5A6';

    if (edge.data?.color !== newColor) {
      return { ...edge, data: { ...edge.data, color: newColor } };
    }
    return edge;
  });
}
