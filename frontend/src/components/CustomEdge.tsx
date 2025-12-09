/**
 * CustomEdge Components
 * 
 * Custom edge types for the blueprint editor:
 * - ExecutionEdge: White edge with arrow for execution flow
 * - DataEdge: Colored edge based on data type (color passed via data prop)
 */

import { memo } from 'react';
import {
  BaseEdge,
  getBezierPath,
  type EdgeProps,
  type Edge,
} from '@xyflow/react';

/**
 * Execution Edge - White with animated arrow marker
 */
export const ExecutionEdge = memo(function ExecutionEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      {/* Arrow marker definition */}
      <defs>
        <marker
          id={`exec-arrow-${id}`}
          markerWidth="12"
          markerHeight="12"
          refX="8"
          refY="6"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <path d="M2,2 L10,6 L2,10 L4,6 Z" fill="white" />
        </marker>
      </defs>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: 'white',
          strokeWidth: selected ? 3 : 2,
          filter: selected ? 'drop-shadow(0 0 4px white)' : undefined,
        }}
        markerEnd={`url(#exec-arrow-${id})`}
      />
      {/* Animated flow indicator */}
      <circle r="3" fill="white">
        <animateMotion dur="1.5s" repeatCount="indefinite" path={edgePath} />
      </circle>
    </>
  );
});

/**
 * Data Edge - Colored based on type (color from data.color)
 */
export interface DataEdgeData {
  color?: string;
  [key: string]: unknown;
}

export type DataEdgeType = Edge<DataEdgeData, 'data'>;

export const DataEdge = memo(function DataEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  selected,
  style,
}: EdgeProps<DataEdgeType>) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  // Use color from data, or from style, or default
  const color = data?.color || (style as React.CSSProperties)?.stroke || '#4A90D9';

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      style={{
        stroke: color,
        strokeWidth: selected ? 3 : 2,
        filter: selected ? `drop-shadow(0 0 4px ${color})` : undefined,
      }}
    />
  );
});


