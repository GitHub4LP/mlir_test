/**
 * FunctionCallNode 组件
 * 
 * 自定义函数调用节点：
 * - 左侧：exec-in + 输入参数
 * - 右侧：exec-out + 返回值
 */

import { memo, useCallback, useMemo } from 'react';
import { type NodeProps, type Node } from '@xyflow/react';
import type { FunctionCallData, DataPin } from '../../../../types';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { NodePins } from '../../../../components/NodePins';
import { buildPinRows, buildCallDataPins } from '../../../../services/pinUtils';
import { useReactStore, typeConstraintStore, usePortStateStore } from '../../../../stores';
import { getDisplayType } from '../../../../services/typeSelectorRenderer';
import { useTypeChangeHandler } from '../../../../hooks';
import { getPortType } from '../../../../services/portTypeService';
import {
  getNodeContainerStyle,
  getHeaderContentStyle,
  getNodeTypeColor,
  NODE_MIN_WIDTH,
} from '../../shared/figmaStyles';

export type FunctionCallNodeType = Node<FunctionCallData, 'function-call'>;
export type FunctionCallNodeProps = NodeProps<FunctionCallNodeType>;

export const FunctionCallNode = memo(function FunctionCallNode({
  id,
  data,
  selected,
}: FunctionCallNodeProps) {
  const { functionName, inputs, outputs, execIn, execOuts, pinnedTypes = {} } = data;
  const getConstraintElements = useReactStore(typeConstraintStore, state => state.getConstraintElements);
  
  // 从 portStateStore 获取端口状态
  const getPortState = usePortStateStore(state => state.getPortState);

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
  const renderTypeSelector = useCallback((pin: DataPin) => {
    const displayType = getDisplayType(pin, data);
    
    // 从 portStateStore 读取端口状态
    const portState = getPortState(id, pin.id);
    // 如果 portState 不存在，默认不可编辑（等待类型传播完成）
    const canEdit = portState?.canEdit ?? false;
    const constraint = portState?.constraint ?? pin.typeConstraint;
    const options = getConstraintElements(constraint);

    return (
      <UnifiedTypeSelector
        selectedType={displayType}
        onTypeSelect={(type) => handleTypeChange(pin.id, type, pin.typeConstraint)}
        constraint={pin.typeConstraint}
        allowedTypes={options.length > 0 ? options : undefined}
        disabled={!canEdit}
      />
    );
  }, [handleTypeChange, id, data, getPortState, getConstraintElements]);

  // Get port type (使用公用服务)
  const getPortTypeWrapper = useCallback((pinId: string) => {
    return getPortType(pinId, { pinnedTypes, inputTypes, outputTypes });
  }, [pinnedTypes, inputTypes, outputTypes]);

  const headerColor = getNodeTypeColor('call');

  return (
    <div
      className="rf-node"
      style={{
        ...getNodeContainerStyle(selected),
        minWidth: NODE_MIN_WIDTH,
      }}
    >
      {/* Header */}
      <div style={getHeaderContentStyle(headerColor)}>
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
