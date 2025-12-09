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
import { type NodeProps, type Node, useReactFlow, useEdges } from '@xyflow/react';
import type { BlueprintNodeData, DataPin } from '../types';
import { getTypeColor, isAbstractConstraint } from '../services/typeSystem';
import { getOperands, getAttributes } from '../services/dialectParser';
import { AttributeEditor } from './AttributeEditor';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { NodePins } from './NodePins';
import { buildPinRows } from '../services/pinUtils';
import { buildPropagationGraph, propagateTypes, extractTypeSources, applyPropagationResult } from '../services/typePropagation/propagator';
import { useProjectStore } from '../stores/projectStore';

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
  const { operation, attributes, inputTypes, outputTypes, execIn, execOuts, regionPins, pinnedTypes = {} } = data;
  const dialectColor = getDialectColor(operation.dialect);

  const { setNodes } = useReactFlow();
  const edges = useEdges();
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);

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

  /**
   * 处理类型选择变更（传播模型）
   * 
   * - 选择具体类型（如 I32）：添加到 pinnedTypes，触发传播
   * - 选择原始约束或抽象约束：从 pinnedTypes 移除，触发传播
   * 
   * 核心概念：
   * - pinnedTypes 存储用户显式选择的类型（持久化）
   * - 传播是无状态的，每次从 pinnedTypes + 图结构重新计算
   */
  const handleTypeChange = useCallback((portId: string, type: string, originalConstraint?: string) => {
    // 判断是否应该 pin：
    // 1. 选择的类型等于原始约束 → 不 pin（恢复默认）
    // 2. 选择的是抽象约束 → 不 pin
    // 3. 其他情况 → pin
    const shouldPin = type && type !== originalConstraint && !isAbstractConstraint(type);

    console.log('handleTypeChange:', { portId, type, originalConstraint, shouldPin });

    // 更新节点数据并触发传播
    setNodes(currentNodes => {
      // 1. 更新当前节点的 pinnedTypes
      const updatedNodes = currentNodes.map(node => {
        if (node.id === id) {
          const nodeData = node.data as BlueprintNodeData;
          const newPinnedTypes = { ...(nodeData.pinnedTypes || {}) };

          if (shouldPin) {
            // 添加到 pinnedTypes
            newPinnedTypes[portId] = type;
            console.log('Pinned:', { portId, type });
          } else {
            // 从 pinnedTypes 移除
            delete newPinnedTypes[portId];
            console.log('Unpinned:', { portId });
          }

          return {
            ...node,
            data: {
              ...nodeData,
              pinnedTypes: newPinnedTypes,
            },
          };
        }
        return node;
      });

      // 2. 重新计算传播结果并更新所有节点（包含函数级别 Traits）
      const currentFunction = getCurrentFunction();
      const graph = buildPropagationGraph(updatedNodes, edges, currentFunction ?? undefined);
      const sources = extractTypeSources(updatedNodes);
      const propagationResult = propagateTypes(graph, sources);

      console.log('Propagation result:', {
        types: [...propagationResult.types.entries()]
      });

      // 3. 统一更新所有节点的显示类型
      return applyPropagationResult(updatedNodes, propagationResult);
    });
  }, [id, edges, setNodes]);

  // Variadic 端口实例数量
  const variadicCounts = data.variadicCounts || {};

  // 计算端口数量约束
  const getQuantity = (isOptional: boolean, isVariadic: boolean): 'required' | 'optional' | 'variadic' => {
    if (isVariadic) return 'variadic';
    if (isOptional) return 'optional';
    return 'required';
  };

  // Build pin rows using unified model
  const pinRows = useMemo(() => {
    const dataInputs: DataPin[] = operands.map((operand) => {
      const portId = `input-${operand.name}`;
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
      const portId = `output-${result.name || `result_${idx}`}`;
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
  // 传播模型：可选类型始终基于原始约束，不会收缩
  const renderTypeSelector = useCallback((pin: DataPin, selectedType?: string) => {
    return (
      <UnifiedTypeSelector
        selectedType={selectedType || pin.typeConstraint}
        onTypeSelect={(type) => handleTypeChange(pin.id, type, pin.typeConstraint)}
        constraint={pin.typeConstraint}
        allowedTypes={pin.allowedTypes}
      />
    );
  }, [handleTypeChange]);

  // Get port type: 从节点数据中获取显示类型
  const getPortTypeWrapper = useCallback((pinId: string) => {
    // 优先检查是否是 pinned 的类型
    const pinnedType = pinnedTypes[pinId];
    if (pinnedType) return pinnedType;

    // 否则使用传播结果（存储在 inputTypes/outputTypes 中）
    // 处理 variadic 端口：input-name_0 → name
    if (pinId.startsWith('input-')) {
      let inputName = pinId.slice('input-'.length);
      // 移除 variadic 索引后缀
      const match = inputName.match(/^(.+)_\d+$/);
      if (match) inputName = match[1];
      return inputTypes[inputName];
    } else if (pinId.startsWith('output-')) {
      let outputName = pinId.slice('output-'.length);
      const match = outputName.match(/^(.+)_\d+$/);
      if (match) outputName = match[1];
      return outputTypes[outputName];
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
