/**
 * React Flow Edge Types Registry
 * 
 * Exports edge components and edgeTypes map for React Flow.
 */

import { ExecutionEdge, DataEdge } from './CustomEdge';

// Re-export components
export { ExecutionEdge, DataEdge } from './CustomEdge';
export type { DataEdgeData, DataEdgeType } from './CustomEdge';

/**
 * Edge types map for React Flow
 */
export const edgeTypes = {
  execution: ExecutionEdge,
  data: DataEdge,
};
