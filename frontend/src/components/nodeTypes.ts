/**
 * Node Types Registry
 * 
 * Exports the node types map for React Flow.
 * Separated from component files to support fast refresh.
 */

import type { NodeTypes } from '@xyflow/react';
import { BlueprintNode } from './BlueprintNode';
import { FunctionEntryNode } from './FunctionEntryNode';
import { FunctionReturnNode } from './FunctionReturnNode';
import { FunctionCallNode } from './FunctionCallNode';

/**
 * Node types map for React Flow
 */
export const nodeTypes: NodeTypes = {
  operation: BlueprintNode,
  'function-entry': FunctionEntryNode,
  'function-return': FunctionReturnNode,
  'function-call': FunctionCallNode,
};
