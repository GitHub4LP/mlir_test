/**
 * FunctionCallNode 组件
 * 
 * 自定义函数调用节点：
 * - 左侧：exec-in + 输入参数
 * - 右侧：exec-out + 返回值
 * - 类型选择逻辑与 Operation 节点相同
 */

import { memo, useCallback, useMemo } from 'react';
import { type NodeProps, type Node, useReactFlow, useEdges } from '@xyflow/react';
import type { FunctionCallData, DataPin } from '../types';
import { getTypeColor, isAbstractConstraint } from '../services/typeSystem';
import { computePortTypeState, isPortConnected, getPropagatedType } from '../services/typeSystemService';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { NodePins } from './NodePins';
import { buildPinRows } from '../services/pinUtils';
import { applyPropagationResult, computePropagationWithNarrowing } from '../services/typePropagation/propagator';
import { useProjectStore } from '../stores/projectStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { dataInHandle, dataOutHandle, PortRef } from '../services/port';

export type FunctionCallNodeType = Node<FunctionCallData, 'function-call'>;
export type FunctionCallNodeProps = NodeProps<FunctionCallNodeType>;

export const FunctionCallNode = memo(function FunctionCallNode({
  id,
  data,
  selected,
}: FunctionCallNodeProps) {
  const { functionName, inputs, outputs, execIn, execOuts, pinnedTypes = {} } = data;
  const { setNodes } = useReactFlow();
  const edges = useEdges();
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);
  const getConcreteTypes = useTypeConstraintStore(state => state.getConcreteTypes);
  const pickConstraintName = useTypeConstraintStore(state => state.pickConstraintName);

  // 构建显示类型映射
  const inputTypes = useMemo(() => {
    const types: Record<string, string> = {};
    for (const port of inputs) {
      types[port.name] = port.concreteType || port.typeConstraint;
    }
    return types;
  }, [inputs]);

  const outputTypes = useMemo(() => {
    const types: Record<string, string> = {};
    for (const port of outputs) {
      types[port.name] = port.concreteType || port.typeConstraint;
    }
    return types;
  }, [outputs]);

  // 处理类型选择（与 BlueprintNode 相同逻辑）
  const handleTypeChange = useCallback((portId: string, type: string, originalConstraint?: string) => {
    const shouldPin = type && type !== originalConstraint && !isAbstractConstraint(type);

    setNodes(currentNodes => {
      const updatedNodes = currentNodes.map(node => {
        if (node.id === id) {
          const nodeData = node.data as FunctionCallData;
          const newPinnedTypes = { ...(nodeData.pinnedTypes || {}) };

          if (shouldPin) {
            newPinnedTypes[portId] = type;
          } else {
            delete newPinnedTypes[portId];
          }

          return { ...node, data: { ...nodeData, pinnedTypes: newPinnedTypes } };
        }
        return node;
      });

      const currentFunction = getCurrentFunction();
      const propagationResult = computePropagationWithNarrowing(
        updatedNodes, edges, currentFunction ?? undefined, getConcreteTypes, pickConstraintName
      );

      return applyPropagationResult(updatedNodes, propagationResult);
    });
  }, [id, edges, setNodes, getCurrentFunction, getConcreteTypes, pickConstraintName]);

  // Build pin rows
  const pinRows = useMemo(() => {
    const dataInputs: DataPin[] = inputs.map((port) => ({
      id: dataInHandle(port.name),  // 统一格式：data-in-{name}
      label: port.name,
      typeConstraint: port.typeConstraint,
      displayName: port.typeConstraint,
      color: getTypeColor(inputTypes[port.name] || port.typeConstraint),
    }));

    const dataOutputs: DataPin[] = outputs.map((port) => ({
      id: dataOutHandle(port.name),  // 统一格式：data-out-{name}
      label: port.name,
      typeConstraint: port.typeConstraint,
      displayName: port.typeConstraint,
      color: getTypeColor(outputTypes[port.name] || port.typeConstraint),
    }));

    return buildPinRows({ execIn, execOuts: execOuts || [], dataInputs, dataOutputs });
  }, [inputs, outputs, inputTypes, outputTypes, execIn, execOuts]);

  // Render type selector
  const renderTypeSelector = useCallback((pin: DataPin) => {
    const propagatedType = getPropagatedType(pin.id, inputTypes, outputTypes);
    const connected = isPortConnected(id, pin.id, edges);

    const state = computePortTypeState({
      portId: pin.id,
      nodeId: id,
      constraint: pin.typeConstraint,
      pinnedTypes,
      propagatedType,
      narrowedConstraint: null,  // TODO: Call 节点的收窄约束处理
      isConnected: connected,
    });

    return (
      <UnifiedTypeSelector
        selectedType={state.displayType}
        onTypeSelect={(type) => handleTypeChange(pin.id, type, pin.typeConstraint)}
        constraint={pin.typeConstraint}
        allowedTypes={state.options ?? undefined}
        disabled={!state.canEdit}
      />
    );
  }, [handleTypeChange, id, pinnedTypes, inputTypes, outputTypes, edges]);

  // Get port type
  const getPortTypeWrapper = useCallback((pinId: string) => {
    if (pinnedTypes[pinId]) return pinnedTypes[pinId];
    
    // 使用 PortRef 解析端口 ID
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

  return (
    <div className={`min-w-48 rounded-lg overflow-visible shadow-lg ${selected ? 'ring-2 ring-blue-400' : ''}`}
      style={{ backgroundColor: '#2d2d3d', border: `1px solid ${selected ? '#60a5fa' : '#3d3d4d'}` }}>
      <div className="px-3 py-2" style={{ backgroundColor: '#9B59B6' }}>
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
