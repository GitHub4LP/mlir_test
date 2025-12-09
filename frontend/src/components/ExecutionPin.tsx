/**
 * ExecutionPin Component
 * 
 * UE5-style execution pin for control flow.
 * White triangular connectors that define execution order.
 */

import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { ExecutionPinConfig } from '../types';

interface ExecutionPinProps {
  pin: ExecutionPinConfig;
  position: Position;
  isConnectable?: boolean;
}

/**
 * ExecutionPin - White triangular execution flow connector
 */
export const ExecutionPin = memo(function ExecutionPin({
  pin,
  position,
  isConnectable = true,
}: ExecutionPinProps) {
  const isInput = position === Position.Left;
  
  return (
    <div className={`flex items-center gap-1 py-1 ${isInput ? 'flex-row' : 'flex-row-reverse'}`}>
      <Handle
        type={isInput ? 'target' : 'source'}
        position={position}
        id={pin.id}
        isConnectable={isConnectable}
        style={{
          width: 0,
          height: 0,
          borderStyle: 'solid',
          borderWidth: isInput ? '6px 8px 6px 0' : '6px 0 6px 8px',
          borderColor: isInput 
            ? 'transparent white transparent transparent'
            : 'transparent transparent transparent white',
          backgroundColor: 'transparent',
          borderRadius: 0,
        }}
      />
      {pin.label && (
        <span className="text-xs text-white/70">{pin.label}</span>
      )}
    </div>
  );
});

export default ExecutionPin;
