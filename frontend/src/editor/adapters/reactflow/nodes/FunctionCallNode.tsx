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
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { NodePins } from '../../../../components/NodePins';
import { buildPinRows, buildCallDataPins } from '../../../../services/pinUtils';
import { useReactStore, projectStore, typeConstraintStore } from '../../../../stores';
import { computeTypeSelectionState } from '../../../../services/typeSelection';
import { getDisplayType } from '../../../../services/typeSelectorRenderer';
import { useTypeChangeHandler } from '../../../../hooks';
import { toEditorNodes, toEditorEdges } from '../typeConversions';
import { getPortType } from '../../../../services/portTypeService';
import {
  getNodeContainerStyle,
  getNodeHeaderStyle,
  getDialectColor,
  tokens,
} from '../../shared/styles';

export type FunctionCallNodeType = Node<FunctionCallData, 'function-call'>;
export type FunctionCallNodeProps = NodeProps<FunctionCallNodeType>;

export const FunctionCallNode = memo(function FunctionCallNode({
  id,
  data,
  selected,
}: FunctionCallNodeProps) {
  const { functionName, inputs, outputs, execIn, execOuts, pinnedTypes = {} } = data;
  const edges = useEdges();
  const getCurrentFunction = useReactStore(projectStore, state => state.getCurrentFunction);
  const getConstraintElements = useReactStore(typeConstraintStore, state => state.getConstraintElements);

  const { inputTypes = {}, outputTypes = {} } = data;

  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  // Build pin rows (使用公用服务)
  const pinRows = useMemo(() => {
    const { inputs: dataInputs, outputs: dataOutputs } = buildCallDataPins(
      inputs,
      outputs,
      { inputTypes, outputTypes }
    );

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

  // Get port type (使用公用服务)
  const getPortTypeWrapper = useCallback((pinId: string) => {
    return getPortType(pinId, { pinnedTypes, inputTypes, outputTypes });
  }, [pinnedTypes, inputTypes, outputTypes]);

  const headerColor = getDialectColor('scf');

  return (
    <div
      className="rf-node"
      style={{
        ...getNodeContainerStyle(selected),
        minWidth: tokens.node.minWidth,
      }}
    >
      {/* Header */}
      <div style={getNodeHeaderStyle(headerColor)}>
        <div className="rf-node-header">
          <div className="rf-node-header-left">
            <span className="rf-node-dialect">call</span>
            <span className="rf-node-opname">{functionName}</span>
          </div>
        </div>
      </div>

      {/* Body */}
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
