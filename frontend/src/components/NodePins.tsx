/**
 * NodePins 组件
 * 
 * 统一的引脚渲染组件，处理执行引脚和数据引脚。
 * - 左侧：输入（exec-in + 数据输入）
 * - 右侧：输出（exec-out + 数据输出）
 * - Variadic 端口支持 +/- 按钮增删
 */

import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { DataPin, PinRow } from '../types';
import { getTypeColor } from '../services/typeSystem';
import {
  getExecHandleStyle,
  getExecHandleStyleRight,
  getDataHandleStyle,
} from '../editor/adapters/shared/styles';

interface PinRowProps {
  row: PinRow;
  nodeId: string;
  getPortType?: (pinId: string) => string | undefined;
  renderTypeSelector?: (pin: DataPin, selectedType?: string) => React.ReactNode;
}

/**
 * Single pin row component
 */
const PinRowComponent = memo(function PinRowComponent({
  row,
  renderTypeSelector,
  getPortType,
}: PinRowProps) {
  const leftPin = row.left;
  const rightPin = row.right;

  return (
    <div className="rf-pin-row">
      {/* Left pin (input) */}
      <div className="rf-pin-row-left">
        {leftPin && (
          <>
            <Handle
              type="target"
              position={Position.Left}
              id={leftPin.pin.id}
              isConnectable={true}
              className="rf-handle-left"
              style={leftPin.type === 'exec'
                ? getExecHandleStyle()
                : getDataHandleStyle((leftPin.pin as DataPin).color || getTypeColor((leftPin.pin as DataPin).typeConstraint))
              }
            />
            <div className="rf-pin-content rf-pin-content-left">
              {/* Show label for data pins, or exec pins with non-empty label */}
              {(leftPin.type === 'data' || leftPin.pin.label) && (
                <span className="rf-pin-label">{leftPin.pin.label}</span>
              )}
              {/* Type selector for data pins */}
              {leftPin.type === 'data' && renderTypeSelector && (
                renderTypeSelector(
                  leftPin.pin as DataPin,
                  getPortType?.(leftPin.pin.id)
                )
              )}
            </div>
          </>
        )}
      </div>

      {/* Right pin (output) */}
      <div className="rf-pin-row-right">
        {rightPin && (
          <>
            <div className="rf-pin-content rf-pin-content-right">
              {/* Show label for data pins, or exec pins with non-empty label */}
              {(rightPin.type === 'data' || rightPin.pin.label) && (
                <span className="rf-pin-label">{rightPin.pin.label}</span>
              )}
              {/* Type selector for data pins */}
              {rightPin.type === 'data' && renderTypeSelector && (
                renderTypeSelector(
                  rightPin.pin as DataPin,
                  getPortType?.(rightPin.pin.id)
                )
              )}
            </div>
            <Handle
              type="source"
              position={Position.Right}
              id={rightPin.pin.id}
              isConnectable={true}
              className="rf-handle-right"
              style={rightPin.type === 'exec'
                ? getExecHandleStyleRight()
                : getDataHandleStyle((rightPin.pin as DataPin).color || getTypeColor((rightPin.pin as DataPin).typeConstraint))
              }
            />
          </>
        )}
      </div>
    </div>
  );
});

/**
 * Variadic 控制按钮行
 */
interface VariadicControlRowProps {
  groupName: string;
  side: 'left' | 'right';
  onAdd: () => void;
  onRemove: () => void;
  canRemove: boolean;
}

const VariadicControlRow = memo(function VariadicControlRow({
  groupName,
  side,
  onAdd,
  onRemove,
  canRemove,
}: VariadicControlRowProps) {
  const controls = (
    <div className="rf-node-header-right">
      <button
        type="button"
        className="rf-variadic-btn rf-variadic-btn-add"
        onClick={onAdd}
        title={`添加 ${groupName}`}
      >
        +
      </button>
      {canRemove && (
        <button
          type="button"
          className="rf-variadic-btn rf-variadic-btn-remove"
          onClick={onRemove}
          title={`删除 ${groupName}`}
        >
          −
        </button>
      )}
    </div>
  );

  return (
    <div className="rf-pin-row" style={{ paddingLeft: 16, paddingRight: 16 }}>
      {side === 'left' ? controls : <div />}
      {side === 'right' ? controls : <div />}
    </div>
  );
});

interface NodePinsProps {
  rows: PinRow[];
  nodeId: string;
  getPortType?: (pinId: string) => string | undefined;
  renderTypeSelector?: (pin: DataPin, selectedType?: string) => React.ReactNode;
  /** Variadic 端口添加回调 */
  onVariadicAdd?: (groupName: string) => void;
  /** Variadic 端口删除回调 */
  onVariadicRemove?: (groupName: string) => void;
  /** Variadic 端口当前数量 */
  variadicCounts?: Record<string, number>;
}

/**
 * Renders all pin rows for a node
 */
export const NodePins = memo(function NodePins({
  rows,
  nodeId,
  getPortType,
  renderTypeSelector,
  onVariadicAdd,
  onVariadicRemove,
  variadicCounts = {},
}: NodePinsProps) {
  // 收集 variadic 组信息
  const variadicGroups = new Map<string, { side: 'left' | 'right'; lastIndex: number }>();

  rows.forEach((row, idx) => {
    const leftPin = row.left?.type === 'data' ? row.left.pin as DataPin : null;
    const rightPin = row.right?.type === 'data' ? row.right.pin as DataPin : null;

    if (leftPin?.variadicGroup) {
      const existing = variadicGroups.get(leftPin.variadicGroup);
      if (!existing || idx > existing.lastIndex) {
        variadicGroups.set(leftPin.variadicGroup, { side: 'left', lastIndex: idx });
      }
    }
    if (rightPin?.variadicGroup) {
      const existing = variadicGroups.get(rightPin.variadicGroup);
      if (!existing || idx > existing.lastIndex) {
        variadicGroups.set(rightPin.variadicGroup, { side: 'right', lastIndex: idx });
      }
    }
  });

  // 构建渲染列表（包含控制按钮行）
  const renderItems: Array<{ type: 'row'; row: PinRow; idx: number } | { type: 'control'; groupName: string; side: 'left' | 'right' }> = [];

  rows.forEach((row, idx) => {
    renderItems.push({ type: 'row', row, idx });

    // 检查是否需要在此行后添加控制按钮
    for (const [groupName, info] of variadicGroups) {
      if (info.lastIndex === idx && onVariadicAdd && onVariadicRemove) {
        renderItems.push({ type: 'control', groupName, side: info.side });
      }
    }
  });

  return (
    <div className="rf-func-body">
      {renderItems.map((item) => {
        if (item.type === 'row') {
          return (
            <PinRowComponent
              key={`row-${item.idx}`}
              row={item.row}
              nodeId={nodeId}
              getPortType={getPortType}
              renderTypeSelector={renderTypeSelector}
            />
          );
        } else {
          const count = variadicCounts[item.groupName] ?? 1;
          return (
            <VariadicControlRow
              key={`control-${item.groupName}`}
              groupName={item.groupName}
              side={item.side}
              onAdd={() => onVariadicAdd?.(item.groupName)}
              onRemove={() => onVariadicRemove?.(item.groupName)}
              canRemove={count > 0}
            />
          );
        }
      })}
    </div>
  );
});

export default NodePins;
