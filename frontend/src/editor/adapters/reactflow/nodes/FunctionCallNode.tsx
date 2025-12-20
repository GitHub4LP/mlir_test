/**
 * FunctionCallNode 组件
 * 
 * 自定义函数调用节点：
 * - 左侧：exec-in + 输入参数
 * - 右侧：exec-out + 返回值
 */

import { memo, useCallback, useMemo } from 'react';
import { type NodeProps, type Node, useEdges, useNodes } from '@xyflow/react';
import type { FunctionCallData, DataPin } from '../../../../types';
import { getTypeColor } from '../../../../services/typeSystem';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { NodePins } from '../../../../components/NodePins';
import { buildPinRows } from '../../../../services/pinUtils';
import { useProjectStore } from '../../../../stores/projectStore';
import { useTypeConstraintStore } from '../../../../stores/typeConstraintStore';
import { dataInHandle, dataOutHandle, PortRef } from '../../../../services/port';
import { computeTypeSelectionState } from '../../../../services/typeSelection';
import { getDisplayType } from '../../../../services/typeSelectorRenderer';
import { useTypeChangeHandler } from '../../../../hooks';
import { StyleSystem } from '../../../core/StyleSystem';
import { getNodeContainerStyle, getNodeHeaderStyle } from '../../../../components/shared';
import { toEditorNodes, toEditorEdges } from '../typeConversions';

export type FunctionCallNodeType = Node<FunctionCallData, 'function-call'>;
export type FunctionCallNodeProps = NodeProps<FunctionCallNodeType>;

export const FunctionCallNode = memo(function FunctionCallNode({
  id,
  data,
  selected,
}: FunctionCallNodeProps) {
  const { functionName, inputs, outputs, execIn, execOuts, pinnedTypes = {} } = data;
  const edges = useEdges();
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);
  const getConstraintElements = useTypeConstraintStore(state => state.getConstraintElements);

  const { inputTypes = {}, outputTypes = {} } = data;

  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  // Build pin rows
  const pinRows = useMemo(() => {
    const dataInputs: DataPin[] = inputs.map((port) => ({
      id: dataInHandle(port.name),
      label: port.name,
      typeConstraint: port.typeConstraint,
      displayName: port.typeConstraint,
      color: getTypeColor(inputTypes[port.name] || port.typeConstraint),
    }));

    const dataOutputs: DataPin[] = outputs.map((port) => ({
      id: dataOutHandle(port.name),
      label: port.name,
      typeConstraint: port.typeConstraint,
      displayName: port.typeConstraint,
      color: getTypeColor(outputTypes[port.name] || port.typeConstraint),
    }));

    return buildPinRows({ execIn, execOuts: execOuts || [], dataInputs, dataOutputs });
  }, [inputs, outputs, inputTypes, outputTypes, execIn, execOuts]);

  // Render type selector
  const nodes = useNodes();
  const renderTypeSelector = useCallback((pin: DataPin) => {
    const displayType = getDisplayType(pin, data);

    const currentFunction = getCurrentFunction();
    const editorNodes = toEditorNodes(nodes);
    const editorEdges = toEditorEdges(edges);
    const { options, canEdit } = computeTypeSelectionState(
      id, pin.id, editorNodes, editorEdges, currentFunction ?? undefined, getConstraintElements
    );

    return (
      <UnifiedTypeSelector
        selectedType={displayType}
        onTypeSelect={(type) => handleTypeChange(pin.id, type, pin.typeConstraint)}
        constraint={pin.typeConstraint}
        allowedTypes={options.length > 0 ? options : undefined}
        disabled={!canEdit}
      />
    );
  }, [handleTypeChange, id, data, edges, nodes, getCurrentFunction, getConstraintElements]);

  // Get port type
  const getPortTypeWrapper = useCallback((pinId: string) => {
    if (pinnedTypes[pinId]) return pinnedTypes[pinId];
    
    const parsed = PortRef.parseHandleId(pinId);
    if (!parsed) return undefined;
    
    if (parsed.kind === 'data-in') {
      const inputPort = inputs.find(p => p.name === parsed.name);
      if (inputPort) return inputTypes[inputPort.name];
    } else if (parsed.kind === 'data-out') {
      const outputPort = outputs.find(p => p.name === parsed.name);
      if (outputPort) return outputTypes[outputPort.name];
    }
    return undefined;
  }, [pinnedTypes, inputs, outputs, inputTypes, outputTypes]);

  const headerColor = StyleSystem.getDialectColor('scf');
  const nodeStyle = StyleSystem.getNodeStyle();

  return (
    <div className="overflow-visible shadow-lg"
      style={{
        ...getNodeContainerStyle(selected),
        minWidth: `${nodeStyle.minWidth}px`,
      }}>
      <div style={getNodeHeaderStyle(headerColor)}>
        <span className="text-xs font-medium text-white/70 uppercase">call</span>
        <span className="text-sm font-semibold text-white ml-1">{functionName}</span>
      </div>
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
