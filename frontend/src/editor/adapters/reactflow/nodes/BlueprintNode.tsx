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
import { type NodeProps, type Node, useReactFlow, useEdges, useNodes } from '@xyflow/react';
import type { BlueprintNodeData, DataPin } from '../../../../types';
import { getTypeColor } from '../../../../services/typeSystem';
import { getOperands, getAttributes } from '../../../../services/dialectParser';
import { getDisplayType } from '../../../../services/typeSelectorRenderer';
import { AttributeEditor } from '../../../../components/AttributeEditor';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { NodePins } from '../../../../components/NodePins';
import { buildPinRows } from '../../../../services/pinUtils';
import { useProjectStore } from '../../../../stores/projectStore';
import { useTypeConstraintStore } from '../../../../stores/typeConstraintStore';
import { dataInHandle, dataOutHandle, PortRef } from '../../../../services/port';
import { computeTypeSelectionState } from '../../../../services/typeSelection';
import { useTypeChangeHandler } from '../../../../hooks';
import { StyleSystem } from '../../../core/StyleSystem';
import { getNodeContainerStyle, getNodeHeaderStyle } from '../../../../components/shared';
import { toEditorNodes, toEditorEdges } from '../typeConversions';

export type BlueprintNodeType = Node<BlueprintNodeData, 'operation'>;
export type BlueprintNodeProps = NodeProps<BlueprintNodeType>;

export const BlueprintNode = memo(function BlueprintNode({ id, data, selected }: BlueprintNodeProps) {
  const { operation, attributes, inputTypes = {}, outputTypes = {}, execIn, execOuts, regionPins, pinnedTypes = {} } = data;
  const dialectColor = StyleSystem.getDialectColor(operation.dialect);

  const { setNodes } = useReactFlow();
  const edges = useEdges();
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);
  const getConstraintElements = useTypeConstraintStore(state => state.getConstraintElements);

  // 使用统一的 hook
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  const operands = getOperands(operation);
  const attrs = getAttributes(operation);
  const results = operation.results;

  /**
   * Updates node attribute and persists to node data
   */
  const handleAttributeChange = useCallback((name: string, value: unknown) => {
    setNodes(nodes => nodes.map(node => {
      if (node.id === id) {
        const nodeData = node.data as BlueprintNodeData;
        return {
          ...node,
          data: {
            ...nodeData,
            attributes: {
              ...nodeData.attributes,
              [name]: value,
            },
          },
        };
      }
      return node;
    }));
  }, [id, setNodes]);

  // Variadic 端口实例数量
  const variadicCounts = useMemo(() => data.variadicCounts ?? {}, [data.variadicCounts]);

  // 计算端口数量约束
  const getQuantity = (isOptional: boolean, isVariadic: boolean): 'required' | 'optional' | 'variadic' => {
    if (isVariadic) return 'variadic';
    if (isOptional) return 'optional';
    return 'required';
  };

  // Build pin rows using unified model
  const pinRows = useMemo(() => {
    const dataInputs: DataPin[] = operands.map((operand) => {
      const portId = dataInHandle(operand.name);
      const quantity = getQuantity(operand.isOptional, operand.isVariadic);
      return {
        id: portId,
        label: operand.name,
        typeConstraint: operand.typeConstraint,
        displayName: operand.displayName || operand.typeConstraint,
        description: operand.description,
        color: getTypeColor(inputTypes[operand.name] || operand.typeConstraint),
        allowedTypes: operand.allowedTypes,
        quantity,
      };
    });

    const dataOutputs: DataPin[] = results.map((result, idx) => {
      const portId = dataOutHandle(result.name || `result_${idx}`);
      const quantity = getQuantity(false, result.isVariadic);
      return {
        id: portId,
        label: result.name || `result_${idx}`,
        typeConstraint: result.typeConstraint,
        displayName: result.displayName || result.typeConstraint,
        description: result.description,
        color: getTypeColor(outputTypes[result.name] || result.typeConstraint),
        allowedTypes: result.allowedTypes,
        quantity,
      };
    });

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

  // Get port type from node data
  const getPortTypeWrapper = useCallback((pinId: string) => {
    const pinnedType = pinnedTypes[pinId];
    if (pinnedType) return pinnedType;

    const parsed = PortRef.parseHandleId(pinId);
    if (!parsed) return undefined;
    
    let portName = parsed.name;
    const match = portName.match(/^(.+)_\d+$/);
    if (match) portName = match[1];
    
    if (parsed.kind === 'data-in') {
      return inputTypes[portName];
    } else if (parsed.kind === 'data-out') {
      return outputTypes[portName];
    }
    return undefined;
  }, [pinnedTypes, inputTypes, outputTypes]);

  // Variadic port add
  const handleVariadicAdd = useCallback((groupName: string) => {
    setNodes(nodes => nodes.map(node => {
      if (node.id === id) {
        const nodeData = node.data as BlueprintNodeData;
        const currentCounts = nodeData.variadicCounts || {};
        const currentCount = currentCounts[groupName] ?? 1;
        return {
          ...node,
          data: {
            ...nodeData,
            variadicCounts: {
              ...currentCounts,
              [groupName]: currentCount + 1,
            },
          },
        };
      }
      return node;
    }));
  }, [id, setNodes]);

  // Variadic port remove
  const handleVariadicRemove = useCallback((groupName: string) => {
    setNodes(nodes => nodes.map(node => {
      if (node.id === id) {
        const nodeData = node.data as BlueprintNodeData;
        const currentCounts = nodeData.variadicCounts || {};
        const currentCount = currentCounts[groupName] ?? 1;
        if (currentCount > 0) {
          return {
            ...node,
            data: {
              ...nodeData,
              variadicCounts: {
                ...currentCounts,
                [groupName]: currentCount - 1,
              },
            },
          };
        }
      }
      return node;
    }));
  }, [id, setNodes]);

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
