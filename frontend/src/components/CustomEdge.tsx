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
 * Execution Edge - White edge, same style as DataEdge but thicker
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
    <BaseEdge
      id={id}
      path={edgePath}
      style={{
        stroke: 'white',
        strokeWidth: selected ? 4 : 3,  // 比数据线粗（数据线是 selected ? 3 : 2）
        filter: selected ? 'drop-shadow(0 0 4px white)' : undefined,
      }}
    />
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


