/**
 * Edge Types Configuration
 * 
 * Maps edge type names to their React components.
 * Separated from CustomEdge.tsx to satisfy React Fast Refresh requirements.
 */

import { ExecutionEdge, DataEdge } from './CustomEdge';

/**
 * Edge types map for React Flow
 */
export const edgeTypes = {
  execution: ExecutionEdge,
  data: DataEdge,
};

export default edgeTypes;
