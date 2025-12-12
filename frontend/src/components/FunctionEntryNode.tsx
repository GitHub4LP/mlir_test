/**
 * FunctionEntryNode 组件
 * 
 * 函数入口节点（UE5 风格）：右侧显示 exec-out + 参数输出
 */

import { memo, useCallback, useMemo, useEffect } from 'react';
import { Handle, Position, type NodeProps, type Node, useEdges, useNodes, useReactFlow } from '@xyflow/react';
import type { FunctionEntryData, DataPin, FunctionTrait } from '../types';
import { getTypeColor } from '../services/typeSystem';
import { useProjectStore } from '../stores/projectStore';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { FunctionTraitsEditor } from './FunctionTraitsEditor';
import { computeTypeSelectorState, type TypeSelectorRenderParams } from '../services/typeSelectorRenderer';
import { EditableName, execPinStyle, dataPinStyle } from './shared';
import { dataOutHandle } from '../services/port';
import { useCurrentFunction, useTypeChangeHandler } from '../hooks';

export type FunctionEntryNodeType = Node<FunctionEntryData, 'function-entry'>;
export type FunctionEntryNodeProps = NodeProps<FunctionEntryNodeType>;

export const FunctionEntryNode = memo(function FunctionEntryNode({ id, data, selected }: FunctionEntryNodeProps) {
  const { functionId, functionName, execOut, isMain, pinnedTypes = {}, outputTypes = {} } = data;
  const edges = useEdges();
  const nodes = useNodes();
  const { setNodes } = useReactFlow();
  const addParameter = useProjectStore(state => state.addParameter);
  const removeParameter = useProjectStore(state => state.removeParameter);
  const updateParameter = useProjectStore(state => state.updateParameter);
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);
  const setFunctionTraits = useProjectStore(state => state.setFunctionTraits);
  const getConcreteTypes = useTypeConstraintStore(state => state.getConcreteTypes);

  // 使用统一的 hooks
  const currentFunction = useCurrentFunction();
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });
  
  const traits = currentFunction?.traits || [];
  const returnTypes = currentFunction?.returnTypes || [];
  const parameters = useMemo(() => currentFunction?.parameters || [], [currentFunction?.parameters]);

  const handleTraitsChange = useCallback((newTraits: FunctionTrait[]) => {
    setFunctionTraits(functionId, newTraits);
  }, [functionId, setFunctionTraits]);

  const handleAddParameter = useCallback(() => {
    const func = getCurrentFunction();
    const existingNames = func?.parameters.map(p => p.name) || [];
    let index = 0, newName = `arg${index}`;
    while (existingNames.includes(newName)) { index++; newName = `arg${index}`; }
    addParameter(functionId, { name: newName, constraint: 'AnyType' });
  }, [functionId, addParameter, getCurrentFunction]);

  const handleRemoveParameter = useCallback((paramName: string) => {
    removeParameter(functionId, paramName);
  }, [functionId, removeParameter]);

  const handleRenameParameter = useCallback((oldName: string, newName: string) => {
    const func = getCurrentFunction();
    const param = func?.parameters.find(p => p.name === oldName);
    if (param) updateParameter(functionId, oldName, { ...param, name: newName });
  }, [functionId, updateParameter, getCurrentFunction]);

  // 同步 FunctionDef.parameters 到 React Flow 节点的 data.outputs
  // 这是为了让类型传播函数能读取到正确的端口列表
  useEffect(() => {
    if (isMain) return;
    
    // 构建新的 outputs
    const newOutputs = parameters.map((param) => ({
      id: dataOutHandle(param.name),
      name: param.name,
      kind: 'output' as const,
      typeConstraint: param.constraint,
      color: getTypeColor(param.constraint),
    }));
    
    // 检查是否需要更新（避免无限循环）
    const currentNames = (data.outputs || []).map((o: { name: string }) => o.name).join(',');
    const newNames = newOutputs.map(o => o.name).join(',');
    
    if (currentNames !== newNames) {
      setNodes(ns => ns.map(n => 
        n.id === id ? { ...n, data: { ...n.data, outputs: newOutputs } } : n
      ));
    }
  }, [id, isMain, parameters, data.outputs, setNodes]);

  // 构建 DataPin 列表 - 直接从 FunctionDef.parameters 派生（单一数据源）
  const dataPins: DataPin[] = useMemo(() => {
    // main 函数没有参数，自定义函数从 FunctionDef 读取
    const params = isMain ? [] : parameters;
    return params.map((param) => {
      const portId = dataOutHandle(param.name);
      const constraint = param.constraint;
      return {
        id: portId,
        label: param.name,
        typeConstraint: constraint,
        displayName: constraint,
        color: getTypeColor(outputTypes[param.name] || pinnedTypes[portId] || constraint),
      };
    });
  }, [isMain, parameters, outputTypes, pinnedTypes]);

  const typeSelectorParams: TypeSelectorRenderParams = useMemo(() => ({
    nodeId: id,
    data,
    nodes,
    edges,
    currentFunction: currentFunction ?? undefined,
    getConcreteTypes,
    onTypeSelect: (portId: string, type: string, originalConstraint: string) => {
      handleTypeChange(portId, type, originalConstraint);
    },
  }), [id, data, nodes, edges, currentFunction, getConcreteTypes, handleTypeChange]);

  const headerColor = isMain ? '#f59e0b' : '#22c55e';

  return (
    <div className={`min-w-48 rounded-lg overflow-visible shadow-lg relative ${selected ? 'ring-2 ring-blue-400' : ''}`}
      style={{ backgroundColor: '#2d2d3d', border: `1px solid ${selected ? '#60a5fa' : '#3d3d4d'}` }}>
      <div className="px-3 py-2" style={{ backgroundColor: headerColor }}>
        <span className="text-sm font-semibold text-white">{functionName || 'Entry'}</span>
        {isMain && <span className="ml-1 text-xs text-white/70">(main)</span>}
      </div>
      <div className="px-1 py-1">
        <div className="relative flex items-center justify-end py-1.5 min-h-7">
          <div className="mr-4" />
          <Handle type="source" position={Position.Right} id={execOut.id} isConnectable={true}
            className="!absolute !right-0 !top-1/2 !-translate-y-1/2 !translate-x-1/2" style={execPinStyle} />
        </div>
        {dataPins.map((pin) => {
          const { displayType, options, canEdit, onSelect } = computeTypeSelectorState(pin, typeSelectorParams);
          return (
            <div key={pin.id} className="relative flex items-center justify-end py-1.5 min-h-7 group">
              {!isMain && <button onClick={() => handleRemoveParameter(pin.label)}
                className="opacity-0 group-hover:opacity-100 p-0.5 text-gray-500 hover:text-red-400 mr-1" title="Remove">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>}
              <div className="mr-4 flex flex-col items-end">
                {!isMain ? <EditableName value={pin.label} onChange={(n) => handleRenameParameter(pin.label, n)} />
                  : <span className="text-xs text-gray-300">{pin.label}</span>}
                <UnifiedTypeSelector 
                  selectedType={displayType}
                  onTypeSelect={onSelect} 
                  constraint={pin.typeConstraint}
                  allowedTypes={options.length > 0 ? options : undefined}
                  disabled={!canEdit} />
              </div>
              <Handle type="source" position={Position.Right} id={pin.id} isConnectable={true}
                className="!absolute !right-0 !top-1/2 !-translate-y-1/2 !translate-x-1/2"
                style={dataPinStyle(pin.color || getTypeColor(pin.typeConstraint))} />
            </div>
          );
        })}
        {!isMain && <div className="relative flex items-center justify-end py-1.5 min-h-7">
          <button onClick={handleAddParameter} className="mr-4 text-gray-500 hover:text-white" title="Add parameter">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
          </button>
        </div>}
        {/* Function Traits Editor - only for non-main functions */}
        {!isMain && (
          <div className="px-2">
            <FunctionTraitsEditor
              parameters={parameters}
              returnTypes={returnTypes}
              traits={traits}
              onChange={handleTraitsChange}
            />
          </div>
        )}
      </div>
    </div>
  );
});

export default FunctionEntryNode;
