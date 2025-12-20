/**
 * React Flow Node Types Registry
 * 
 * Exports node components and nodeTypes map for React Flow.
 */

import type { NodeTypes } from '@xyflow/react';
import { BlueprintNode } from './BlueprintNode';
import { FunctionEntryNode } from './FunctionEntryNode';
import { FunctionReturnNode } from './FunctionReturnNode';
import { FunctionCallNode } from './FunctionCallNode';

// Re-export components
export { BlueprintNode } from './BlueprintNode';
export { FunctionEntryNode } from './FunctionEntryNode';
export { FunctionReturnNode } from './FunctionReturnNode';
export { FunctionCallNode } from './FunctionCallNode';

// Re-export types
export type { BlueprintNodeType, BlueprintNodeProps } from './BlueprintNode';
export type { FunctionEntryNodeType, FunctionEntryNodeProps } from './FunctionEntryNode';
export type { FunctionReturnNodeType, FunctionReturnNodeProps } from './FunctionReturnNode';
export type { FunctionCallNodeType, FunctionCallNodeProps } from './FunctionCallNode';

/**
 * Node types map for React Flow
 */
export const nodeTypes: NodeTypes = {
  operation: BlueprintNode,
  'function-entry': FunctionEntryNode,
  'function-return': FunctionReturnNode,
  'function-call': FunctionCallNode,
};
