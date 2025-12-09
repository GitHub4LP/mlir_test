/**
 * FunctionCallNode Component
 * 
 * A node that represents a call to a custom function.
 * Uses unified pin rendering:
 * - Left side: exec-in + input parameters
 * - Right side: exec-outs + output return values
 * - Multiple exec-outs supported (one per Return node in function)
 * 
 * Requirements: 13.1, 13.2, 13.3
 */

import { memo, useCallback, useMemo } from 'react';
import { type NodeProps, type Node } from '@xyflow/react';
import type { FunctionCallData, DataPin } from '../types';
import { getTypeColor } from '../services/typeSystem';
import { useTypeStore } from '../stores/typeStore';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { NodePins } from './NodePins';
import { buildPinRows } from '../services/pinUtils';

export type FunctionCallNodeType = Node<FunctionCallData, 'function-call'>;
export type FunctionCallNodeProps = NodeProps<FunctionCallNodeType>;

export const FunctionCallNode = memo(function FunctionCallNode({
  id,
  data,
  selected,
}: FunctionCallNodeProps) {
  const { functionName, inputs, outputs, execIn, execOuts } = data;
  const setPortType = useTypeStore(state => state.setPortType);
  const getPortType = useTypeStore(state => state.getPortType);

  const handleTypeChange = useCallback((portId: string, type: string) => {
    setPortType(id, portId, type);
  }, [id, setPortType]);

  // Build pin rows using unified model
  const pinRows = useMemo(() => {
    const dataInputs: DataPin[] = inputs.map((port) => ({
      id: port.id,
      label: port.name,
      typeConstraint: port.typeConstraint,
      displayName: port.typeConstraint,
      color: port.color || getTypeColor(port.typeConstraint),
    }));

    const dataOutputs: DataPin[] = outputs.map((port) => ({
      id: port.id,
      label: port.name,
      typeConstraint: port.typeConstraint,
      displayName: port.typeConstraint,
      color: port.color || getTypeColor(port.typeConstraint),
    }));

    return buildPinRows({
      execIn,
      execOuts: execOuts || [],
      dataInputs,
      dataOutputs,
    });
  }, [inputs, outputs, execIn, execOuts]);

  // Render type selector for data pins
  const renderTypeSelector = useCallback((pin: DataPin, selectedType?: string) => (
    <UnifiedTypeSelector
      selectedType={selectedType || pin.typeConstraint}
      onTypeSelect={(type) => handleTypeChange(pin.id, type)}
      constraint={pin.typeConstraint}
    />
  ), [handleTypeChange]);

  // Get port type: 优先使用节点数据中的类型
  const getPortTypeWrapper = useCallback((pinId: string) => {
    // 从 inputs/outputs 中查找具体类型
    const inputPort = inputs.find(p => p.id === pinId);
    if (inputPort?.concreteType) return inputPort.concreteType;
    
    const outputPort = outputs.find(p => p.id === pinId);
    if (outputPort?.concreteType) return outputPort.concreteType;
    
    // 回退到 typeStore
    return getPortType(id, pinId);
  }, [id, inputs, outputs, getPortType]);

  return (
    <div className={`min-w-48 rounded-lg overflow-visible shadow-lg ${selected ? 'ring-2 ring-blue-400' : ''}`}
      style={{ backgroundColor: '#2d2d3d', border: `1px solid ${selected ? '#60a5fa' : '#3d3d4d'}` }}>
      
      {/* Header */}
      <div className="px-3 py-2" style={{ backgroundColor: '#9B59B6' }}>
        <span className="text-xs font-medium text-white/70 uppercase">call</span>
        <span className="text-sm font-semibold text-white ml-1">{functionName}</span>
      </div>

      {/* Body - Unified pin rendering */}
      <NodePins
        rows={pinRows}
        nodeId={id}
        getPortType={getPortTypeWrapper}
        renderTypeSelector={renderTypeSelector}
      />
    </div>
  );
});

export default FunctionCallNode;
