/**
 * FunctionEntryNode 组件
 * 
 * 函数入口节点（UE5 风格）：右侧显示 exec-out + 参数输出
 */

import { memo, useCallback, useState, useMemo } from 'react';
import { Handle, Position, type NodeProps, type Node, useEdges, useNodes } from '@xyflow/react';
import type { FunctionEntryData, DataPin, FunctionTrait } from '../types';
import { getTypeColor } from '../services/typeSystem';
import { useProjectStore } from '../stores/projectStore';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { FunctionTraitsEditor } from './FunctionTraitsEditor';
import { 
  getInternalConnectedConstraints, 
  getExternalConnectedConstraints,
  computeSignaturePortOptions 
} from '../services/typeSystemService';

export type FunctionEntryNodeType = Node<FunctionEntryData, 'function-entry'>;
export type FunctionEntryNodeProps = NodeProps<FunctionEntryNodeType>;

const EditableName = memo(function EditableName({
  value, onChange, className = '',
}: { value: string; onChange: (newName: string) => void; className?: string; }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(value);

  const handleDoubleClick = useCallback(() => { setEditValue(value); setIsEditing(true); }, [value]);
  const handleBlur = useCallback(() => {
    setIsEditing(false);
    if (editValue.trim() && editValue !== value) onChange(editValue.trim());
  }, [editValue, value, onChange]);
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleBlur();
    else if (e.key === 'Escape') { setIsEditing(false); setEditValue(value); }
  }, [handleBlur, value]);

  if (isEditing) {
    return <input type="text" value={editValue} onChange={(e) => setEditValue(e.target.value)}
      onBlur={handleBlur} onKeyDown={handleKeyDown} onClick={(e) => e.stopPropagation()} autoFocus
      className={`text-xs bg-gray-700 text-white px-1 py-0.5 rounded border border-blue-500 outline-none w-16 ${className}`} />;
  }
  return <span className={`text-xs text-gray-300 cursor-text hover:text-white ${className}`}
    onDoubleClick={handleDoubleClick} title="Double-click to edit">{value}</span>;
});

const execPinStyle = {
  width: 0, height: 0, borderStyle: 'solid' as const, borderWidth: '5px 0 5px 8px',
  borderColor: 'transparent transparent transparent white', backgroundColor: 'transparent', borderRadius: 0
};
const dataPinStyle = (color: string) => ({
  width: 10, height: 10, backgroundColor: color,
  border: '2px solid #1a1a2e', borderRadius: '50%'
});

export const FunctionEntryNode = memo(function FunctionEntryNode({ id, data, selected }: FunctionEntryNodeProps) {
  const { functionId, functionName, outputs, execOut, isMain } = data;
  const edges = useEdges();
  const nodes = useNodes();
  const project = useProjectStore(state => state.project);
  const addParameter = useProjectStore(state => state.addParameter);
  const removeParameter = useProjectStore(state => state.removeParameter);
  const updateParameter = useProjectStore(state => state.updateParameter);
  const getCurrentFunction = useProjectStore(state => state.getCurrentFunction);
  const setFunctionTraits = useProjectStore(state => state.setFunctionTraits);

  // 获取当前函数的 traits 和 returnTypes
  const currentFunction = getCurrentFunction();
  const traits = currentFunction?.traits || [];
  const returnTypes = currentFunction?.returnTypes || [];
  const parameters = currentFunction?.parameters || [];

  const handleTraitsChange = useCallback((newTraits: FunctionTrait[]) => {
    setFunctionTraits(functionId, newTraits);
  }, [functionId, setFunctionTraits]);

  const handleParameterTypeChange = useCallback((paramName: string, newType: string) => {
    const func = getCurrentFunction();
    const param = func?.parameters.find(p => p.name === paramName);
    if (param) updateParameter(functionId, paramName, { ...param, type: newType });
  }, [functionId, updateParameter, getCurrentFunction]);

  const handleAddParameter = useCallback(() => {
    const func = getCurrentFunction();
    const existingNames = func?.parameters.map(p => p.name) || [];
    let index = 0, newName = `arg${index}`;
    while (existingNames.includes(newName)) { index++; newName = `arg${index}`; }
    addParameter(functionId, { name: newName, type: 'AnyType' });
  }, [functionId, addParameter, getCurrentFunction]);

  const handleRemoveParameter = useCallback((paramName: string) => {
    removeParameter(functionId, paramName);
  }, [functionId, removeParameter]);

  const handleRenameParameter = useCallback((oldName: string, newName: string) => {
    const func = getCurrentFunction();
    const param = func?.parameters.find(p => p.name === oldName);
    if (param) updateParameter(functionId, oldName, { ...param, name: newName });
  }, [functionId, updateParameter, getCurrentFunction]);

  // 计算端口的可选类型
  const getPortOptions = useCallback((portId: string, paramName: string): string[] | null => {
    if (isMain || !project) return null;
    
    // 获取内部约束（函数内连接的操作约束）
    const internalConstraints = getInternalConnectedConstraints(portId, id, nodes, edges);
    
    // 获取外部约束（调用处连接的类型/约束）
    const externalConstraints = getExternalConnectedConstraints(functionId, paramName, 'param', project);
    
    return computeSignaturePortOptions(internalConstraints, externalConstraints);
  }, [isMain, project, id, nodes, edges, functionId]);

  const dataPins: DataPin[] = useMemo(() => outputs.map((port) => ({
    id: port.id, label: port.name, typeConstraint: port.typeConstraint,
    displayName: port.typeConstraint,
    color: port.color || getTypeColor(port.typeConstraint),
  })), [outputs]);

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
          const options = getPortOptions(pin.id, pin.label);
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
                  selectedType={pin.typeConstraint}
                  onTypeSelect={(t) => handleParameterTypeChange(pin.label, t)} 
                  constraint="AnyType"
                  allowedTypes={options ?? undefined}
                  disabled={isMain} />
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
