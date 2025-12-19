/**
 * FunctionReturnNode 组件
 * 
 * 函数返回节点（UE5 风格）：左侧显示 exec-in + 返回值输入
 * 每个 Return 节点可有 branchName 用于多出口函数
 */

import { memo, useCallback, useMemo, useEffect } from 'react';
import { Handle, Position, type NodeProps, type Node, useEdges, useNodes, useReactFlow } from '@xyflow/react';
import type { FunctionReturnData, DataPin } from '../types';
import { getTypeColor } from '../services/typeSystem';
import { useProjectStore } from '../stores/projectStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { computeTypeSelectorState, type TypeSelectorRenderParams } from '../services/typeSelectorRenderer';
import { EditableName, execPinStyle, dataPinStyle } from './shared';
import { dataInHandle } from '../services/port';
import { useCurrentFunction, useTypeChangeHandler } from '../hooks';

export type FunctionReturnNodeType = Node<FunctionReturnData, 'function-return'>;
export type FunctionReturnNodeProps = NodeProps<FunctionReturnNodeType>;

export const FunctionReturnNode = memo(function FunctionReturnNode({ id, data, selected }: FunctionReturnNodeProps) {
  const { branchName, execIn, isMain, pinnedTypes = {}, inputTypes = {} } = data;
  const edges = useEdges();
  const nodes = useNodes();
  const { setNodes } = useReactFlow();
  const addReturnType = useProjectStore(state => state.addReturnType);
  const removeReturnType = useProjectStore(state => state.removeReturnType);
  const updateReturnType = useProjectStore(state => state.updateReturnType);
  const getFunctionById = useProjectStore(state => state.getFunctionById);
  const getConstraintElements = useTypeConstraintStore(state => state.getConstraintElements);

  const currentFunction = useCurrentFunction();
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  const functionId = currentFunction?.id || '';

  const handleAddReturnType = useCallback(() => {
    const func = getFunctionById(functionId);
    const existingNames = func?.returnTypes.map(r => r.name || '') || [];
    let index = 0, newName = `ret${index}`;
    while (existingNames.includes(newName)) { index++; newName = `ret${index}`; }
    if (functionId) addReturnType(functionId, { name: newName, constraint: 'AnyType' });
  }, [functionId, addReturnType, getFunctionById]);

  const handleRemoveReturnType = useCallback((returnName: string) => {
    if (functionId) removeReturnType(functionId, returnName);
  }, [functionId, removeReturnType]);

  const handleRenameReturnType = useCallback((oldName: string, newName: string) => {
    const func = getFunctionById(functionId);
    const ret = func?.returnTypes.find(r => r.name === oldName);
    if (ret && functionId) updateReturnType(functionId, oldName, { ...ret, name: newName });
  }, [functionId, updateReturnType, getFunctionById]);

  // 同步 FunctionDef.returnTypes 到 React Flow 节点的 data.inputs
  // 这是为了让类型传播函数能读取到正确的端口列表
  useEffect(() => {
    if (isMain) return;
    
    const returnTypes = currentFunction?.returnTypes || [];
    
    // 构建新的 inputs
    const newInputs = returnTypes.map((ret, idx) => ({
      id: dataInHandle(ret.name || `result_${idx}`),
      name: ret.name || `result_${idx}`,
      kind: 'input' as const,
      typeConstraint: ret.constraint,
      color: getTypeColor(ret.constraint),
    }));
    
    // 检查是否需要更新（避免无限循环）
    const currentNames = (data.inputs || []).map((i: { name: string }) => i.name).join(',');
    const newNames = newInputs.map(i => i.name).join(',');
    
    if (currentNames !== newNames) {
      setNodes(ns => ns.map(n => 
        n.id === id ? { ...n, data: { ...n.data, inputs: newInputs } } : n
      ));
    }
  }, [id, isMain, currentFunction?.returnTypes, data.inputs, setNodes]);

  // 构建 DataPin 列表 - 直接从 FunctionDef.returnTypes 派生（单一数据源）
  const dataPins: DataPin[] = useMemo(() => {
    // main 函数返回固定的 I32，自定义函数从 FunctionDef 读取
    const returns = isMain 
      ? [{ name: 'result', constraint: 'I32' }] 
      : (currentFunction?.returnTypes || []);
    return returns.map((ret, idx) => {
      const name = ret.name || `result_${idx}`;
      const portId = dataInHandle(name);
      const constraint = ret.constraint;
    return {
      id: portId,
        label: name,
      typeConstraint: constraint,
      displayName: constraint,
        color: getTypeColor(inputTypes[name] || pinnedTypes[portId] || constraint),
    };
    });
  }, [isMain, currentFunction?.returnTypes, inputTypes, pinnedTypes]);

  const typeSelectorParams: TypeSelectorRenderParams = useMemo(() => ({
    nodeId: id,
    data,
    nodes,
    edges,
    currentFunction: currentFunction ?? undefined,
    getConstraintElements,
    onTypeSelect: (portId: string, type: string, originalConstraint: string) => {
      handleTypeChange(portId, type, originalConstraint);
    },
  }), [id, data, nodes, edges, currentFunction, getConstraintElements, handleTypeChange]);

  const headerColor = isMain ? '#dc2626' : '#ef4444';
  const headerText = branchName ? `Return "${branchName}"` : 'Return';

  return (
    <div className={`min-w-48 rounded-lg overflow-visible shadow-lg relative ${selected ? 'ring-2 ring-blue-400' : ''}`}
      style={{ backgroundColor: '#2d2d3d', border: `1px solid ${selected ? '#60a5fa' : '#3d3d4d'}` }}>
      <div className="px-3 py-2" style={{ backgroundColor: headerColor }}>
        <span className="text-sm font-semibold text-white">{headerText}</span>
        {isMain && <span className="ml-1 text-xs text-white/70">(main)</span>}
      </div>
      <div className="px-1 py-1">
        <div className="relative flex items-center py-1.5 min-h-7">
          <Handle type="target" position={Position.Left} id={execIn.id} isConnectable={true}
            className="!absolute !left-0 !top-1/2 !-translate-y-1/2 !-translate-x-1/2" style={execPinStyle} />
          <div className="ml-4" />
        </div>
        {dataPins.map((pin) => {
          const { displayType, options, canEdit, onSelect } = computeTypeSelectorState(pin, typeSelectorParams);
          return (
            <div key={pin.id} className="relative flex items-center py-1.5 min-h-7 group">
              <Handle type="target" position={Position.Left} id={pin.id} isConnectable={true}
                className="!absolute !left-0 !top-1/2 !-translate-y-1/2 !-translate-x-1/2"
                style={dataPinStyle(pin.color || getTypeColor(pin.typeConstraint))} />
              <div className="ml-4 flex flex-col items-start flex-1">
                {!isMain ? <EditableName value={pin.label} onChange={(n) => handleRenameReturnType(pin.label, n)} />
                  : <span className="text-xs text-gray-300">{pin.label}</span>}
                <UnifiedTypeSelector 
                  selectedType={displayType}
                  onTypeSelect={onSelect} 
                  constraint={pin.typeConstraint}
                  allowedTypes={options.length > 0 ? options : undefined}
                  disabled={!canEdit} />
              </div>
              {!isMain && <button onClick={() => handleRemoveReturnType(pin.label)}
                className="opacity-0 group-hover:opacity-100 p-0.5 text-gray-500 hover:text-red-400 ml-1" title="Remove">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>}
            </div>
          );
        })}
        {!isMain && <div className="relative flex items-center py-1.5 min-h-7">
          <div className="ml-4">
            <button onClick={handleAddReturnType} className="text-gray-500 hover:text-white" title="Add return value">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
            </button>
          </div>
        </div>}
      </div>
    </div>
  );
});

export default FunctionReturnNode;
