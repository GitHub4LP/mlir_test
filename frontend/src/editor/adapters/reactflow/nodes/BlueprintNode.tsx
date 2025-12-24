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
import { type NodeProps, type Node, useEdges, useNodes } from '@xyflow/react';
import type { BlueprintNodeData, DataPin } from '../../../../types';
import { getOperands, getAttributes } from '../../../../services/dialectParser';
import { getDisplayType } from '../../../../services/typeSelectorRenderer';
import { AttributeEditor } from '../../../../components/AttributeEditor';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { NodePins } from '../../../../components/NodePins';
import { buildPinRows, buildOperationDataPins } from '../../../../services/pinUtils';
import { useReactStore, projectStore, typeConstraintStore } from '../../../../stores';
import { computeTypeSelectionState } from '../../../../services/typeSelection';
import { useTypeChangeHandler } from '../../../../hooks';
import { StyleSystem } from '../../../core/StyleSystem';
import { getNodeContainerStyle, getNodeHeaderStyle } from '../../../../components/shared';
import { toEditorNodes, toEditorEdges } from '../typeConversions';
import { getPortType } from '../../../../services/portTypeService';
import { incrementVariadicCount, decrementVariadicCount } from '../../../../services/variadicService';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';

export type BlueprintNodeType = Node<BlueprintNodeData, 'operation'>;
export type BlueprintNodeProps = NodeProps<BlueprintNodeType>;

export const BlueprintNode = memo(function BlueprintNode({ id, data, selected }: BlueprintNodeProps) {
  const { operation, attributes, inputTypes = {}, outputTypes = {}, execIn, execOuts, regionPins, pinnedTypes = {} } = data;
  const dialectColor = StyleSystem.getDialectColor(operation.dialect);

  // 直接更新 editorStore（数据一份，订阅更新）
  const { updateNodeData } = useEditorStoreUpdate<BlueprintNodeData>(id);
  
  const edges = useEdges();
  const getCurrentFunction = useReactStore(projectStore, state => state.getCurrentFunction);
  const getConstraintElements = useReactStore(typeConstraintStore, state => state.getConstraintElements);

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

  const nodeStyle = StyleSystem.getNodeStyle();

  return (
    <div className="overflow-visible shadow-lg"
      style={{
        ...getNodeContainerStyle(selected),
        minWidth: `${nodeStyle.minWidth}px`,
      }}>

      {/* Header */}
      <div
        style={getNodeHeaderStyle(dialectColor)}
        title={operation.description || undefined}
      >
        <div className="flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-white/70 uppercase">{operation.dialect}</span>
            <span className="text-sm font-semibold text-white ml-1">{operation.opName}</span>
          </div>
          <div className="flex gap-1">
            {operation.isPure && (
              <span className="text-xs text-white/60" title="Pure - no side effects">ƒ</span>
            )}
            {operation.traits.includes('Commutative') && (
              <span className="text-xs text-white/60" title="Commutative - operand order doesn't matter">⇄</span>
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
        <div className="border-t border-gray-600 px-2 py-1">
          {attrs.map((attr) => (
            <AttributeEditor key={attr.name} attribute={attr} value={attributes[attr.name]} onChange={handleAttributeChange} />
          ))}
        </div>
      )}

      {/* Summary */}
      {operation.summary && (
        <div className="px-2 py-1 border-t border-gray-600 bg-gray-800/50">
          <span className="text-xs text-gray-500 line-clamp-1">{operation.summary}</span>
        </div>
      )}
    </div>
  );
});

export default BlueprintNode;
