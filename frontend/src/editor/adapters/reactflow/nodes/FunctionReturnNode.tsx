/**
 * FunctionReturnNode 组件
 * 
 * 函数返回节点（UE5 风格）：左侧显示 exec-in + 返回值输入
 */

import { memo, useCallback, useMemo, useEffect } from 'react';
import { Handle, Position, type NodeProps, type Node, useEdges, useNodes } from '@xyflow/react';
import type { FunctionReturnData } from '../../../../types';
import { getTypeColor } from '../../../../services/typeSystem';
import { useReactStore, projectStore, typeConstraintStore } from '../../../../stores';
import { UnifiedTypeSelector } from '../../../../components/UnifiedTypeSelector';
import { computeTypeSelectorState, type TypeSelectorRenderParams } from '../../../../services/typeSelectorRenderer';
import { EditableName, execPinStyle, dataPinStyle, getNodeContainerStyle, getNodeHeaderStyle } from '../../../../components/shared';
import { dataInHandle } from '../../../../services/port';
import { useCurrentFunction, useTypeChangeHandler } from '../../../../hooks';
import { StyleSystem } from '../../../core/StyleSystem';
import { toEditorNodes, toEditorEdges } from '../typeConversions';
import { generateReturnTypeName } from '../../../../services/parameterService';
import { buildReturnDataPins } from '../../../../services/pinUtils';
import { useEditorStoreUpdate } from '../useEditorStoreUpdate';

export type FunctionReturnNodeType = Node<FunctionReturnData, 'function-return'>;
export type FunctionReturnNodeProps = NodeProps<FunctionReturnNodeType>;

export const FunctionReturnNode = memo(function FunctionReturnNode({ id, data, selected }: FunctionReturnNodeProps) {
  const { branchName, execIn, isMain, pinnedTypes = {}, inputTypes = {} } = data;
  const edges = useEdges();
  const nodes = useNodes();
  
  // 直接更新 editorStore（数据一份，订阅更新）
  const { updateNodeData } = useEditorStoreUpdate<FunctionReturnData>(id);
  
  const addReturnType = useReactStore(projectStore, state => state.addReturnType);
  const removeReturnType = useReactStore(projectStore, state => state.removeReturnType);
  const updateReturnType = useReactStore(projectStore, state => state.updateReturnType);
  const getFunctionById = useReactStore(projectStore, state => state.getFunctionById);
  const getConstraintElements = useReactStore(typeConstraintStore, state => state.getConstraintElements);

  const currentFunction = useCurrentFunction();
  const { handleTypeChange } = useTypeChangeHandler({ nodeId: id });

  const functionId = currentFunction?.id || '';

  const handleAddReturnType = useCallback(() => {
    const func = getFunctionById(functionId);
    const existingNames = func?.returnTypes.map(r => r.name || '') || [];
    const newName = generateReturnTypeName(existingNames);
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

  // Sync FunctionDef.returnTypes to editorStore node data.inputs
  useEffect(() => {
    if (isMain) return;
    
    const returnTypes = currentFunction?.returnTypes || [];
    
    const newInputs = returnTypes.map((ret, idx) => ({
      id: dataInHandle(ret.name || `result_${idx}`),
      name: ret.name || `result_${idx}`,
      kind: 'input' as const,
      typeConstraint: ret.constraint,
      color: getTypeColor(ret.constraint),
    }));
    
    const currentNames = (data.inputs || []).map((i: { name: string }) => i.name).join(',');
    const newNames = newInputs.map(i => i.name).join(',');
    
    if (currentNames !== newNames) {
      updateNodeData(nodeData => ({ ...nodeData, inputs: newInputs }));
    }
  }, [id, isMain, currentFunction?.returnTypes, data.inputs, updateNodeData]);

  // Build DataPin list from FunctionDef.returnTypes (使用公用服务)
  const dataPins = useMemo(() => {
    const returnTypes = currentFunction?.returnTypes || [];
    return buildReturnDataPins(returnTypes, { pinnedTypes, inputTypes }, isMain);
  }, [isMain, currentFunction?.returnTypes, inputTypes, pinnedTypes]);

  const typeSelectorParams: TypeSelectorRenderParams = useMemo(() => ({
    nodeId: id,
    data,
    nodes: toEditorNodes(nodes),
    edges: toEditorEdges(edges),
    currentFunction: currentFunction ?? undefined,
    getConstraintElements,
    onTypeSelect: (portId: string, type: string, originalConstraint: string) => {
      handleTypeChange(portId, type, originalConstraint);
    },
  }), [id, data, nodes, edges, currentFunction, getConstraintElements, handleTypeChange]);

  const headerColor = isMain ? '#dc2626' : '#ef4444';
  const headerText = branchName ? `Return "${branchName}"` : 'Return';
  const nodeStyle = StyleSystem.getNodeStyle();

  return (
    <div className="overflow-visible shadow-lg relative"
      style={{
        ...getNodeContainerStyle(selected),
        minWidth: `${nodeStyle.minWidth}px`,
      }}>
      <div style={getNodeHeaderStyle(headerColor)}>
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
