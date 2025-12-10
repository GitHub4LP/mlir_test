/**
 * FunctionReturnNode 组件
 * 
 * 函数返回节点（UE5 风格）：左侧显示 exec-in + 返回值输入
 * 每个 Return 节点可有 branchName 用于多出口函数
 */

import { memo, useCallback, useState, useMemo } from 'react';
import { Handle, Position, type NodeProps, type Node, useEdges, useNodes } from '@xyflow/react';
import type { FunctionReturnData, DataPin } from '../types';
import { getTypeColor } from '../services/typeSystem';
import { useProjectStore } from '../stores/projectStore';
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import { 
  getInternalConnectedConstraints, 
  getExternalConnectedConstraints,
  computeSignaturePortOptions 
} from '../services/typeSystemService';

export type FunctionReturnNodeType = Node<FunctionReturnData, 'function-return'>;
export type FunctionReturnNodeProps = NodeProps<FunctionReturnNodeType>;

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

export const FunctionReturnNode = memo(function FunctionReturnNode({ id, data, selected }: FunctionReturnNodeProps) {
  const { functionId, branchName, inputs, execIn, isMain } = data;
  const edges = useEdges();
  const nodes = useNodes();
  const project = useProjectStore(state => state.project);
  const addReturnType = useProjectStore(state => state.addReturnType);
  const removeReturnType = useProjectStore(state => state.removeReturnType);
  const updateReturnType = useProjectStore(state => state.updateReturnType);
  const getFunctionById = useProjectStore(state => state.getFunctionById);

  const handleReturnTypeChange = useCallback((returnName: string, newType: string) => {
    const func = getFunctionById(functionId);
    const ret = func?.returnTypes.find(r => r.name === returnName);
    if (ret) updateReturnType(functionId, returnName, { ...ret, type: newType });
  }, [functionId, updateReturnType, getFunctionById]);

  const handleAddReturnType = useCallback(() => {
    const func = getFunctionById(functionId);
    const existingNames = func?.returnTypes.map(r => r.name || '') || [];
    let index = 0, newName = `ret${index}`;
    while (existingNames.includes(newName)) { index++; newName = `ret${index}`; }
    addReturnType(functionId, { name: newName, type: 'AnyType' });
  }, [functionId, addReturnType, getFunctionById]);

  const handleRemoveReturnType = useCallback((returnName: string) => {
    removeReturnType(functionId, returnName);
  }, [functionId, removeReturnType]);

  const handleRenameReturnType = useCallback((oldName: string, newName: string) => {
    const func = getFunctionById(functionId);
    const ret = func?.returnTypes.find(r => r.name === oldName);
    if (ret) updateReturnType(functionId, oldName, { ...ret, name: newName });
  }, [functionId, updateReturnType, getFunctionById]);

  // 计算端口的可选类型
  const getPortOptions = useCallback((portId: string, returnName: string): string[] | null => {
    if (isMain || !project) return null;
    
    // 获取内部约束（函数内连接的操作约束）
    const internalConstraints = getInternalConnectedConstraints(portId, id, nodes, edges);
    
    // 获取外部约束（调用处连接的类型/约束）
    const externalConstraints = getExternalConnectedConstraints(functionId, returnName, 'return', project);
    
    return computeSignaturePortOptions(internalConstraints, externalConstraints);
  }, [isMain, project, id, nodes, edges, functionId]);

  const dataPins: DataPin[] = useMemo(() => inputs.map((port) => ({
    id: port.id, label: port.name, typeConstraint: port.typeConstraint,
    displayName: port.typeConstraint,
    color: port.color || getTypeColor(port.typeConstraint),
  })), [inputs]);

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
          const options = getPortOptions(pin.id, pin.label);
          return (
            <div key={pin.id} className="relative flex items-center py-1.5 min-h-7 group">
              <Handle type="target" position={Position.Left} id={pin.id} isConnectable={true}
                className="!absolute !left-0 !top-1/2 !-translate-y-1/2 !-translate-x-1/2"
                style={dataPinStyle(pin.color || getTypeColor(pin.typeConstraint))} />
              <div className="ml-4 flex flex-col items-start flex-1">
                {!isMain ? <EditableName value={pin.label} onChange={(n) => handleRenameReturnType(pin.label, n)} />
                  : <span className="text-xs text-gray-300">{pin.label}</span>}
                <UnifiedTypeSelector 
                  selectedType={pin.typeConstraint}
                  onTypeSelect={(t) => handleReturnTypeChange(pin.label, t)} 
                  constraint="AnyType"
                  allowedTypes={options ?? undefined}
                  disabled={isMain} />
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
