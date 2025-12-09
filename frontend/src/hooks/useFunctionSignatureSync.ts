/**
 * useFunctionSignatureSync Hook
 * 
 * Synchronizes function call nodes when function signatures change.
 * Updates all instances of a function call node when the function's
 * parameters or return types are modified.
 * 
 * Requirements: 13.2, 14.3, 14.4
 */

import { useEffect, useCallback, useRef } from 'react';
import type { Node, Edge } from '@xyflow/react';
import type { FunctionDef, FunctionCallData } from '../types';
import { 
  updateFunctionCallNodeData, 
  createInputPortsFromParams,
  createOutputPortsFromReturns,
} from '../services/functionNodeGenerator';

interface UseFunctionSignatureSyncOptions {
  /** All custom functions in the project */
  customFunctions: FunctionDef[];
  /** Current nodes in the graph */
  nodes: Node[];
  /** Function to update nodes */
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
  /** Function to update edges (to remove invalid edges) */
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
}

/**
 * Creates a signature hash for comparison
 */
function createSignatureHash(func: FunctionDef): string {
  const params = func.parameters.map(p => `${p.name}:${p.type}`).join(',');
  const returns = func.returnTypes.map(r => `${r.name}:${r.type}`).join(',');
  return `${func.name}|${params}|${returns}`;
}

/**
 * Hook to synchronize function call nodes when function signatures change
 */
export function useFunctionSignatureSync({
  customFunctions,
  nodes,
  setNodes,
  setEdges,
}: UseFunctionSignatureSyncOptions) {
  // Store previous signatures for comparison
  const prevSignatures = useRef<Map<string, string>>(new Map());

  // Update function call nodes when a function signature changes
  const updateFunctionCallNodes = useCallback((updatedFunc: FunctionDef) => {
    setNodes((currentNodes) => {
      return currentNodes.map((node) => {
        if (node.type === 'function-call') {
          const nodeData = node.data as unknown as FunctionCallData;
          if (nodeData.functionId === updatedFunc.id) {
            const newData = updateFunctionCallNodeData(nodeData, updatedFunc);
            return { ...node, data: newData as unknown as Record<string, unknown> };
          }
        }
        return node;
      });
    });

    // Remove edges that connect to ports that no longer exist
    const newInputPortIds = new Set(
      createInputPortsFromParams(updatedFunc).map(p => p.id)
    );
    const newOutputPortIds = new Set(
      createOutputPortsFromReturns(updatedFunc).map(p => p.id)
    );

    // Find all function call nodes for this function
    const functionCallNodeIds = nodes
      .filter((n) => {
        if (n.type !== 'function-call') return false;
        const nodeData = n.data as unknown as FunctionCallData;
        return nodeData.functionId === updatedFunc.id;
      })
      .map(n => n.id);

    if (functionCallNodeIds.length > 0) {
      setEdges((currentEdges) => {
        return currentEdges.filter((edge) => {
          // Check if edge connects to a function call node
          if (functionCallNodeIds.includes(edge.source)) {
            // Source is a function call node - check if output port still exists
            if (!newOutputPortIds.has(edge.sourceHandle || '')) {
              return false;
            }
          }
          if (functionCallNodeIds.includes(edge.target)) {
            // Target is a function call node - check if input port still exists
            if (!newInputPortIds.has(edge.targetHandle || '')) {
              return false;
            }
          }
          return true;
        });
      });
    }
  }, [nodes, setNodes, setEdges]);

  // Check for signature changes and update nodes
  useEffect(() => {
    const currentSignatures = new Map<string, string>();
    
    for (const func of customFunctions) {
      const signature = createSignatureHash(func);
      currentSignatures.set(func.id, signature);
      
      const prevSignature = prevSignatures.current.get(func.id);
      if (prevSignature && prevSignature !== signature) {
        // Signature changed - update all function call nodes
        updateFunctionCallNodes(func);
      }
    }
    
    prevSignatures.current = currentSignatures;
  }, [customFunctions, updateFunctionCallNodes]);

  return {
    updateFunctionCallNodes,
  };
}

export default useFunctionSignatureSync;
