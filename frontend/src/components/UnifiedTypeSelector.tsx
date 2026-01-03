/**
 * UnifiedTypeSelector - 统一类型选择器 (React 版本)
 * 
 * 设计原则（来自 type-selector-design.md）：
 * 1. 构建面板始终可见：显示当前类型结构，支持嵌套可视化
 * 2. 选择面板复用：点击任意可编辑部分（▼）时显示同一个选择器
 * 3. 约束类型支持：AnyType、AnyFloat 等约束可作为类型使用
 * 4. 包装选项统一：在选择面板中提供 +tensor、+vector 等包装入口
 * 5. 方言分组：内置约束/类型 + 方言特有约束分组显示
 * 
 * 数据逻辑已抽取到 typeSelectorService.ts，本组件只负责渲染
 */

import { memo, useCallback, useMemo, useRef, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { getTypeColor } from '../services/typeSystem';
import { useTypeConstraintStore } from '../stores/typeConstraintStore';
import {
  type TypeNode,
  type WrapperInfo,
  WRAPPERS,
  serializeType,
  parseType,
  wrapWith,
} from '../services/typeNodeUtils';
import {
  type SearchFilter,
  computeTypeSelectorData,
  computeTypeGroups,
  hasConstraintLimit,
} from '../services/typeSelectorService';

// ============ 选择面板组件 ============

interface SelectionPanelProps {
  position: { top: number; left: number };
  currentValue: string;
  onSelect: (value: string) => void;
  onWrap: (wrapper: string) => void;
  onClose: () => void;
  panelRef: React.RefObject<HTMLDivElement | null>;
  /** 约束对应的具体类型列表，null 表示无约束 */
  constraintTypes: string[] | null;
  /** 允许的包装器列表 */
  allowedWrappers: readonly WrapperInfo[];
  /** 原始约束名（用于显示） */
  constraintName?: string;
}

const SelectionPanel = memo(function SelectionPanel({
  position,
  currentValue,
  onSelect,
  onWrap,
  onClose,
  panelRef,
  constraintTypes,
  allowedWrappers,
  constraintName,
}: SelectionPanelProps) {
  const [search, setSearch] = useState('');
  const [showConstraints, setShowConstraints] = useState(true);
  const [showTypes, setShowTypes] = useState(true);
  const [useRegex, setUseRegex] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const { buildableTypes, constraintDefs, getConstraintElements } = useTypeConstraintStore();

  // 是否有约束限制
  const hasConstraint = hasConstraintLimit(constraintTypes, constraintName);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // 使用 service 计算类型分组
  const typeGroups = useMemo(() => {
    const filter: SearchFilter = { searchText: search, showConstraints, showTypes, useRegex };
    const data = { 
      analysis: { 
        isUnconstrained: !hasConstraint, 
        isCompositeConstraint: false, 
        allowedWrappers: null, 
        scalarTypes: constraintTypes 
      }, 
      allowedWrappers, 
      constraintTypes 
    };
    return computeTypeGroups(
      data,
      filter,
      constraintName,
      buildableTypes,
      constraintDefs,
      getConstraintElements
    );
  }, [hasConstraint, constraintTypes, constraintName, constraintDefs, buildableTypes, 
      search, showConstraints, showTypes, useRegex, getConstraintElements, allowedWrappers]);

  const totalCount = useMemo(() =>
    typeGroups.reduce((sum, g) => sum + g.items.length, 0),
    [typeGroups]
  );

  return createPortal(
    <div
      ref={panelRef}
      className="rf-type-selector-panel"
      style={{ position: 'fixed', top: position.top, left: position.left, zIndex: 10000 }}
      onMouseDown={e => e.stopPropagation()}
    >
      {/* 搜索栏 */}
      <div className="rf-type-selector-search">
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, backgroundColor: 'var(--color-gray-700)', borderRadius: 'var(--radius-default)', padding: '4px 8px' }}>
          <input
            ref={inputRef}
            type="text"
            className="rf-type-selector-search-input"
            placeholder="搜索..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            onKeyDown={e => e.key === 'Escape' && onClose()}
          />
          {/* 只在无约束时显示过滤按钮 */}
          {!hasConstraint && (
            <div style={{ display: 'flex', gap: 2, borderLeft: '1px solid var(--color-gray-600)', paddingLeft: 4 }}>
              <button
                type="button"
                className="rf-filter-btn"
                style={{ backgroundColor: showConstraints ? 'var(--color-blue-600)' : 'transparent', color: showConstraints ? 'var(--color-white)' : 'var(--color-gray-500)' }}
                onClick={() => setShowConstraints(v => !v)}
                title="约束"
              >C</button>
              <button
                type="button"
                className="rf-filter-btn"
                style={{ backgroundColor: showTypes ? 'var(--color-blue-600)' : 'transparent', color: showTypes ? 'var(--color-white)' : 'var(--color-gray-500)' }}
                onClick={() => setShowTypes(v => !v)}
                title="类型"
              >T</button>
              <button
                type="button"
                className="rf-filter-btn"
                style={{ backgroundColor: useRegex ? 'var(--color-blue-600)' : 'transparent', color: useRegex ? 'var(--color-white)' : 'var(--color-gray-500)' }}
                onClick={() => setUseRegex(v => !v)}
                title="正则"
              >.*</button>
            </div>
          )}
        </div>
      </div>

      {/* 包装选项 - 只在有允许的包装器时显示 */}
      {allowedWrappers.length > 0 && (
        <div style={{ padding: '6px 8px', borderBottom: '1px solid var(--color-gray-700)' }}>
          <div style={{ fontSize: 'var(--text-subtitle-size)', color: 'var(--color-gray-500)', marginBottom: 4 }}>包装为:</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {allowedWrappers.map(w => (
              <button
                key={w.name}
                type="button"
                className="rf-wrapper-btn"
                onClick={() => onWrap(w.name)}
              >
                +{w.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 列表 */}
      <div className="rf-type-selector-list">
        {totalCount === 0 ? (
          <div style={{ padding: 12, fontSize: 'var(--text-label-size)', color: 'var(--color-gray-500)', textAlign: 'center' }}>无匹配结果</div>
        ) : (
          typeGroups.map((group) => (
            <div key={group.label}>
              {/* 有约束时只有一个分组，不显示标签 */}
              {!(hasConstraint && typeGroups.length === 1) && (
                <div className="rf-type-selector-group">
                  {group.label}
                </div>
              )}
              {group.items.map(item => {
                const color = getTypeColor(item);
                return (
                  <button
                    key={item}
                    type="button"
                    className="rf-type-selector-item"
                    style={{ color, backgroundColor: item === currentValue ? 'var(--color-gray-700)' : 'transparent' }}
                    onClick={() => onSelect(item)}
                  >
                    {item}
                  </button>
                );
              })}
            </div>
          ))
        )}
      </div>

      <div style={{ padding: '4px 8px', fontSize: 'var(--text-label-size)', color: 'var(--color-gray-500)', borderTop: '1px solid var(--color-gray-700)' }}>
        {totalCount} 个结果
      </div>
    </div>,
    document.body
  );
});

// ============ Shape 编辑器 ============

const ShapeEditor = memo(function ShapeEditor({
  shape,
  onChange,
}: {
  shape: (number | null)[];
  onChange: (shape: (number | null)[]) => void;
}) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2 }}>
      {shape.map((d, i) => (
        <span key={i} style={{ display: 'inline-flex', alignItems: 'center' }}>
          {i > 0 && <span style={{ color: 'var(--color-gray-500)' }}>×</span>}
          <span style={{ position: 'relative' }} className="rf-shape-dim">
            <input
              type="text"
              className="rf-shape-input"
              value={d === null ? '?' : d}
              onChange={e => {
                const val = e.target.value;
                const newShape = [...shape];
                newShape[i] = val === '?' || val === '' ? null : (parseInt(val, 10) || null);
                onChange(newShape);
              }}
              onMouseDown={e => e.stopPropagation()}
            />
            {shape.length > 1 && (
              <button
                type="button"
                className="rf-shape-remove-btn"
                onClick={() => onChange(shape.filter((_, j) => j !== i))}
              >×</button>
            )}
          </span>
        </span>
      ))}
      <button
        type="button"
        className="rf-shape-add-btn"
        onClick={() => onChange([...shape, 4])}
      >+</button>
      <span style={{ color: 'var(--color-gray-500)' }}>×</span>
    </span>
  );
});

// ============ 构建面板（递归渲染类型结构）============

interface BuildPanelProps {
  node: TypeNode;
  onChange: (node: TypeNode) => void;
  onOpenSelector: (target: 'leaf', rect: DOMRect, currentValue: string, element: Element) => void;
  path?: string;
}

const BuildPanel = memo(function BuildPanel({
  node,
  onChange,
  onOpenSelector,
  path = 'root',
}: BuildPanelProps) {
  const leafRef = useRef<HTMLButtonElement>(null);

  const handleLeafClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (leafRef.current) {
      const rect = leafRef.current.getBoundingClientRect();
      onOpenSelector('leaf', rect, node.kind === 'scalar' ? node.name : '', leafRef.current);
    }
  }, [node, onOpenSelector]);

  const handleUnwrap = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    if (node.kind === 'composite') {
      onChange(node.element);
    }
  }, [node, onChange]);

  const handleShapeChange = useCallback((shape: (number | null)[]) => {
    if (node.kind === 'composite') {
      onChange({ ...node, shape });
    }
  }, [node, onChange]);

  const handleElementChange = useCallback((element: TypeNode) => {
    if (node.kind === 'composite') {
      onChange({ ...node, element });
    }
  }, [node, onChange]);

  if (node.kind === 'scalar') {
    const color = getTypeColor(node.name);
    return (
      <button
        ref={leafRef}
        type="button"
        className="rf-type-leaf-btn"
        style={{ color }}
        onClick={handleLeafClick}
      >
        {node.name}
        <span className="rf-type-dropdown-icon">▼</span>
      </button>
    );
  }

  // 复合类型
  const w = WRAPPERS.find(x => x.name === node.wrapper);

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
      <span style={{ display: 'inline-flex', alignItems: 'center' }} className="rf-wrapper-group">
        <span style={{ color: 'var(--color-purple-500)' }}>{node.wrapper}</span>
        <span style={{ color: 'var(--color-gray-500)' }}>&lt;</span>
        <button
          type="button"
          className="rf-unwrap-btn"
          onClick={handleUnwrap}
          title="移除此层"
        >×</button>
      </span>

      {w?.hasShape && node.shape && (
        <ShapeEditor shape={node.shape} onChange={handleShapeChange} />
      )}

      <BuildPanel
        node={node.element}
        onChange={handleElementChange}
        onOpenSelector={onOpenSelector}
        path={`${path}.element`}
      />

      <span style={{ color: 'var(--color-gray-500)' }}>&gt;</span>
    </span>
  );
});

// ============ 主组件 ============

interface UnifiedTypeSelectorProps {
  selectedType: string;
  onTypeSelect: (type: string) => void;
  /** 
   * 可选约束，限制可选类型范围
   * - 无约束或 AnyType：显示所有约束和类型，允许构建任意复合类型
   * - 标量约束（如 SignlessIntegerLike）：只显示匹配的标量类型
   * - 复合类型约束（如 AnyTensor）：允许构建对应的复合类型
   */
  constraint?: string;
  /** 
   * 允许的具体类型列表（来自后端 AnyTypeOf 解析）
   * 如果提供，优先使用此列表而非从 constraintMap 查找
   */
  allowedTypes?: string[];
  /** 显示名称（用于 tooltip） */
  displayName?: string;
  /** 描述（用于 tooltip） */
  description?: string;
  disabled?: boolean;
  className?: string;
}

export const UnifiedTypeSelector = memo(function UnifiedTypeSelector({
  selectedType,
  onTypeSelect,
  constraint,
  allowedTypes: propAllowedTypes,
  disabled = false,
  className = '',
}: UnifiedTypeSelectorProps) {
  const { 
    buildableTypes, 
    constraintDefs, 
    getConstraintElements, 
    isShapedConstraint, 
    getAllowedContainers 
  } = useTypeConstraintStore();

  // 使用 service 计算数据
  const selectorData = useMemo(() => 
    computeTypeSelectorData({
      constraint,
      allowedTypes: propAllowedTypes,
      buildableTypes,
      constraintDefs,
      getConstraintElements,
      isShapedConstraint,
      getAllowedContainers,
    }),
    [constraint, propAllowedTypes, buildableTypes, constraintDefs, 
     getConstraintElements, isShapedConstraint, getAllowedContainers]
  );

  const { allowedWrappers, constraintTypes } = selectorData;

  const [isOpen, setIsOpen] = useState(false);
  const [selectorPos, setSelectorPos] = useState({ top: 0, left: 0 });
  // node 完全由 selectedType 派生（受控模式），无需本地 state
  const node = useMemo(() => parseType(selectedType || 'AnyType'), [selectedType]);
  const containerRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const clickedElementRef = useRef<Element | null>(null);

  // 更新位置（RAF 循环）- 跟随点击的元素
  useEffect(() => {
    if (!isOpen) return;

    let rafId: number;
    const update = () => {
      const el = clickedElementRef.current;
      if (el && document.body.contains(el)) {
        const rect = el.getBoundingClientRect();
        setSelectorPos({ top: rect.bottom + 4, left: rect.left });
      }
      rafId = requestAnimationFrame(update);
    };
    rafId = requestAnimationFrame(update);
    return () => cancelAnimationFrame(rafId);
  }, [isOpen]);

  // 点击外部关闭
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      // 检查是否点击在容器或面板内
      if (containerRef.current?.contains(target)) return;
      if (panelRef.current?.contains(target)) return;
      setIsOpen(false);
    };
    // 使用 capture 阶段，确保在其他处理器之前执行
    document.addEventListener('mousedown', handler, true);
    return () => document.removeEventListener('mousedown', handler, true);
  }, [isOpen]);

  const handleOpenSelector = useCallback((_target: 'leaf', rect: DOMRect, _currentValue: string, element: Element) => {
    clickedElementRef.current = element;
    setSelectorPos({ top: rect.bottom + 4, left: rect.left });
    setIsOpen(true);
  }, []);

  const handleSelect = useCallback((value: string) => {
    // 找到最内层的 scalar 并替换
    const updateLeaf = (n: TypeNode): TypeNode => {
      if (n.kind === 'scalar') {
        return { kind: 'scalar', name: value };
      }
      return { ...n, element: updateLeaf(n.element) };
    };
    const newNode = updateLeaf(node);
    onTypeSelect(serializeType(newNode));
    setIsOpen(false);
  }, [node, onTypeSelect]);

  const handleWrap = useCallback((wrapper: string) => {
    // 包装最内层的 scalar，而不是整个 node
    const wrapLeaf = (n: TypeNode): TypeNode => {
      if (n.kind === 'scalar') {
        return wrapWith(n, wrapper);
      }
      return { ...n, element: wrapLeaf(n.element) };
    };
    const newNode = wrapLeaf(node);
    onTypeSelect(serializeType(newNode));
    setIsOpen(false);
  }, [node, onTypeSelect]);

  const handleNodeChange = useCallback((newNode: TypeNode) => {
    onTypeSelect(serializeType(newNode));
  }, [onTypeSelect]);

  const preview = useMemo(() => serializeType(node), [node]);
  const color = getTypeColor(node.kind === 'scalar' ? node.name : 'tensor');

  // 禁用时显示为只读
  if (disabled) {
    return (
      <span
        className={`rf-type-readonly ${className}`}
        style={{ color, backgroundColor: `${color}20` }}
      >
        {preview}
      </span>
    );
  }

  return (
    <>
      <div
        ref={containerRef}
        className={`rf-type-container ${className}`}
        onMouseDown={e => e.stopPropagation()}
      >
        <BuildPanel
          node={node}
          onChange={handleNodeChange}
          onOpenSelector={handleOpenSelector}
        />
      </div>

      {isOpen && (
        <SelectionPanel
          position={selectorPos}
          currentValue={node.kind === 'scalar' ? node.name : ''}
          onSelect={handleSelect}
          onWrap={handleWrap}
          onClose={() => setIsOpen(false)}
          panelRef={panelRef}
          constraintTypes={constraintTypes}
          allowedWrappers={allowedWrappers}
          constraintName={constraint}
        />
      )}
    </>
  );
});

export default UnifiedTypeSelector;
