/**
 * BlueprintNode 组件
 * 
 * UE5 风格蓝图节点组件。
 * - 左侧：exec-in + 输入操作数
 * - 右侧：exec-out + 输出结果
 * 
 * 类型传播模型：
 * - pinnedTypes：用户显式选择的类型（持久化）
 * - inputTypes/outputTypes：显示用的类型（传播结果）
 */

import { memo, useCallback, useMemo } from 'react';
import { type NodeProps, type Node } from '@xyflow/react';
import type { BlueprintNodeData, DataPin } from '../../../../types';
import { getOperands, getAttributes } from '../../../../services/dialectParser';
import { getDisplayType } from '../../../../services/typeSelectorRenderer';
import { AttributeEditor } from '../../../../components/AttributeEditor';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { NodePins } from '../../../../components/NodePins';
import { buildPinRows, buildOperationDataPins } from '../../../../services/pinUtils';
import { useReactStore, typeConstraintStore, usePortStateStore } from '../../../../stores';
import { useTypeChangeHandler } from '../../../../hooks';
import {
  getNodeContainerStyle,
  getHeaderContentStyle,
  getDialectColor,
  NODE_MIN_WIDTH,
} from '../../shared/figmaStyles';
import { getPortType } from '../../../../services/portTypeService';
import { incrementVariadicCount, decrementVariadicCount } from '../../../../services/variadicService';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';

export type BlueprintNodeType = Node<BlueprintNodeData, 'operation'>;
export type BlueprintNodeProps = NodeProps<BlueprintNodeType>;

export const BlueprintNode = memo(function BlueprintNode({ id, data, selected }: BlueprintNodeProps) {
  const { operation, attributes, inputTypes = {}, outputTypes = {}, execIn, execOuts, regionPins, pinnedTypes = {} } = data;
  const dialectColor = getDialectColor(operation.dialect);

  // 直接更新 editorStore（数据一份，订阅更新）
  const { updateNodeData } = useEditorStoreUpdate<BlueprintNodeData>(id);
  
  const getConstraintElements = useReactStore(typeConstraintStore, state => state.getConstraintElements);
  
  // 从 portStateStore 获取端口状态
  const getPortState = usePortStateStore(state => state.getPortState);

  // 使用统一的 hook
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  const operands = getOperands(operation);
  const attrs = getAttributes(operation);
  const results = operation.results;

  /**
   * Updates node attribute - 直接更新 editorStore
   */
  const handleAttributeChange = useCallback((name: string, value: unknown) => {
    updateNodeData(nodeData => ({
      ...nodeData,
      attributes: {
        ...(nodeData.attributes as Record<string, string>),
        [name]: String(value),
      },
    }));
  }, [updateNodeData]);

  // Variadic 端口实例数量
  const variadicCounts = useMemo(() => data.variadicCounts ?? {}, [data.variadicCounts]);

  // Build pin rows using unified model (使用公用服务构建 DataPin)
  const pinRows = useMemo(() => {
    const { inputs: dataInputs, outputs: dataOutputs } = buildOperationDataPins(
      operands,
      results,
      { inputTypes, outputTypes }
    );

    return buildPinRows({
      execIn,
      execOuts,
      dataInputs,
      dataOutputs,
      regionPins,
      variadicCounts,
    });
  }, [operands, results, inputTypes, outputTypes, execIn, execOuts, regionPins, variadicCounts]);

  // Render type selector for data pins
  const renderTypeSelector = useCallback((pin: DataPin) => {
    const displayType = getDisplayType(pin, data);
    
    // 从 portStateStore 读取端口状态
    const portState = getPortState(id, pin.id);
    // 如果 portState 不存在，默认不可编辑（等待类型传播完成）
    const canEdit = portState?.canEdit ?? false;
    
    // 如果 portState 存在，使用其 constraint 计算 options
    // 否则回退到 pin.typeConstraint
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

  // Get port type from node data (使用公用服务)
  const getPortTypeWrapper = useCallback((pinId: string) => {
    return getPortType(pinId, { pinnedTypes, inputTypes, outputTypes });
  }, [pinnedTypes, inputTypes, outputTypes]);

  // Variadic port add - 直接更新 editorStore
  const handleVariadicAdd = useCallback((groupName: string) => {
    updateNodeData(nodeData => ({
      ...nodeData,
      variadicCounts: incrementVariadicCount(nodeData.variadicCounts || {}, groupName),
    }));
  }, [updateNodeData]);

  // Variadic port remove - 直接更新 editorStore
  const handleVariadicRemove = useCallback((groupName: string) => {
    updateNodeData(nodeData => ({
      ...nodeData,
      variadicCounts: decrementVariadicCount(nodeData.variadicCounts || {}, groupName, 0),
    }));
  }, [updateNodeData]);

  return (
    <div
      className="rf-node"
      style={{
        ...getNodeContainerStyle(selected),
        minWidth: NODE_MIN_WIDTH,
      }}
    >
      {/* Header */}
      <div
        style={getHeaderContentStyle(dialectColor)}
        title={operation.description || undefined}
      >
        <div className="rf-node-header">
          <div className="rf-node-header-left">
            <span className="rf-node-dialect">{operation.dialect}</span>
            <span className="rf-node-opname">{operation.opName}</span>
          </div>
          <div className="rf-node-header-right">
            {operation.isPure && (
              <span className="rf-node-badge" title="Pure - no side effects">ƒ</span>
            )}
            {operation.traits.includes('Commutative') && (
              <span className="rf-node-badge" title="Commutative - operand order doesn't matter">⇄</span>
            )}
          </div>
        </div>
      </div>

      {/* Body */}
      <NodePins
        rows={pinRows}
        nodeId={id}
        getPortType={getPortTypeWrapper}
        renderTypeSelector={renderTypeSelector}
        onVariadicAdd={handleVariadicAdd}
        onVariadicRemove={handleVariadicRemove}
        variadicCounts={variadicCounts}
      />

      {/* Attributes */}
      {attrs.length > 0 && (
        <div className="rf-node-attrs">
          {attrs.map((attr) => (
            <AttributeEditor key={attr.name} attribute={attr} value={attributes[attr.name]} onChange={handleAttributeChange} />
          ))}
        </div>
      )}

      {/* Summary */}
      {operation.summary && (
        <div className="rf-node-summary">
          <span className="rf-node-summary-text">{operation.summary}</span>
        </div>
      )}
    </div>
  );
});

export default BlueprintNode;
