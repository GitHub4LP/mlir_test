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
import type { BlueprintNodeData, DataPin } from '../types';
import { getTypeColor } from '../services/typeSystem';
import { getOperands, getAttributes } from '../services/dialectParser';
import { getDisplayType } from '../services/typeSelectorRenderer';
import { AttributeEditor } from './AttributeEditor';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { NodePins } from './NodePins';
import { buildPinRows } from '../services/pinUtils';
import { useProjectStore } from '../stores/projectStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { dataInHandle, dataOutHandle, PortRef } from '../services/port';
import { computeTypeSelectionState } from '../services/typeSelection';
import { useTypeChangeHandler } from '../hooks';

export type BlueprintNodeType = Node<BlueprintNodeData, 'operation'>;
export type BlueprintNodeProps = NodeProps<BlueprintNodeType>;

function getDialectColor(dialect: string): string {
  const colors: Record<string, string> = {
    arith: '#4A90D9', func: '#50C878', scf: '#9B59B6', memref: '#E74C3C',
    tensor: '#1ABC9C', linalg: '#F39C12', vector: '#F1C40F', affine: '#E67E22',
    gpu: '#2ECC71', math: '#3498DB', cf: '#8E44AD', builtin: '#7F8C8D',
  };
  return colors[dialect] || '#95A5A6';
}

export const BlueprintNode = memo(function BlueprintNode({ id, data, selected }: BlueprintNodeProps) {
  const { operation, attributes, inputTypes = {}, outputTypes = {}, execIn, execOuts, regionPins, pinnedTypes = {} } = data;
  const dialectColor = getDialectColor(operation.dialect);

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
      const portId = dataInHandle(operand.name);  // 统一格式：data-in-{name}
      const quantity = getQuantity(operand.isOptional, operand.isVariadic);
      // typeConstraint 始终是操作定义的原始约束（用于判断是否可编辑）
      // selectedType 通过 getPortTypeWrapper 获取（用于显示当前选择）
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
      const portId = dataOutHandle(result.name || `result_${idx}`);  // 统一格式：data-out-{name}
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

  // Render type selector for data pins（使用统一的 getDisplayType）
  const nodes = useNodes();
  const renderTypeSelector = useCallback((pin: DataPin) => {
    // 使用统一的 getDisplayType
    const displayType = getDisplayType(pin, data);
    
    // 统一使用 computeTypeSelectionState 计算可选集和 canEdit
    const currentFunction = getCurrentFunction();
    const { options, canEdit } = computeTypeSelectionState(
      id, pin.id, nodes, edges, currentFunction ?? undefined, getConstraintElements
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

  // Get port type: 从节点数据中获取显示类型
  const getPortTypeWrapper = useCallback((pinId: string) => {
    // 优先检查是否是 pinned 的类型
    const pinnedType = pinnedTypes[pinId];
    if (pinnedType) return pinnedType;

    // 否则使用传播结果（存储在 inputTypes/outputTypes 中）
    // 使用 PortRef 解析端口 ID
    const parsed = PortRef.parseHandleId(pinId);
    if (!parsed) return undefined;
    
    let portName = parsed.name;
    // 移除 variadic 索引后缀：name_0 → name
    const match = portName.match(/^(.+)_\d+$/);
    if (match) portName = match[1];
    
    if (parsed.kind === 'data-in') {
      return inputTypes[portName];
    } else if (parsed.kind === 'data-out') {
      return outputTypes[portName];
    }
    return undefined;
  }, [pinnedTypes, inputTypes, outputTypes]);

  // Variadic 端口添加
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

  // Variadic 端口删除
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

  return (
    <div className={`min-w-56 rounded-lg overflow-visible shadow-lg ${selected ? 'ring-2 ring-blue-400' : ''}`}
      style={{ backgroundColor: '#2d2d3d', border: `1px solid ${selected ? '#60a5fa' : '#3d3d4d'}` }}>

      {/* Header - 悬停显示 description */}
      <div
        className="px-3 py-2"
        style={{ backgroundColor: dialectColor }}
        title={operation.description || undefined}
      >
        <div className="flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-white/70 uppercase">{operation.dialect}</span>
            <span className="text-sm font-semibold text-white ml-1">{operation.opName}</span>
          </div>
          {/* Traits 图标 */}
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

      {/* Body - Unified pin rendering */}
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
