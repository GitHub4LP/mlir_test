/**
 * UnifiedTypeSelector - 统一类型选择器
 * 
 * 设计原则（来自 type-selector-design.md）：
 * 1. 构建面板始终可见：显示当前类型结构，支持嵌套可视化
 * 2. 选择面板复用：点击任意可编辑部分（▼）时显示同一个选择器
 * 3. 约束类型支持：AnyType、AnyFloat 等约束可作为类型使用
 * 4. 包装选项统一：在选择面板中提供 +tensor、+vector 等包装入口
 * 5. 方言分组：内置约束/类型 + 方言特有约束分组显示
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

// ============ 类型分组 ============

interface TypeGroup {
  label: string;
  items: string[];
}

function buildTypeGroups(
  constraintDefs: Map<string, { name: string; summary: string; rule: unknown }>,
  buildableTypes: string[],
  searchText: string,
  showConstraints: boolean,
  showTypes: boolean,
  useRegex: boolean
): TypeGroup[] {
  const buildableSet = new Set(buildableTypes);
  const groups: TypeGroup[] = [];

  // 构建匹配函数
  let matcher: (item: string) => boolean;
  if (!searchText) {
    matcher = () => true;
  } else if (useRegex) {
    try {
      const regex = new RegExp(searchText, 'i');
      matcher = (item) => regex.test(item);
    } catch {
      matcher = () => false;
    }
  } else {
    const lower = searchText.toLowerCase();
    matcher = (item) => item.toLowerCase().includes(lower);
  }

  // 1. 内置约束（排除具体类型）
  if (showConstraints) {
    const items = [...constraintDefs.keys()]
      .filter(key => !buildableSet.has(key))
      .filter(matcher)
      .sort();
    if (items.length > 0) {
      groups.push({ label: '约束', items });
    }
  }

  // 2. 内置类型
  if (showTypes) {
    const items = buildableTypes.filter(matcher).sort();
    if (items.length > 0) {
      groups.push({ label: '类型', items });
    }
  }

  return groups;
}

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

  // 是否有约束限制（AnyType 视为无约束，显示完整过滤功能）
  const hasConstraint = constraintTypes !== null && constraintName !== 'AnyType' && constraintName !== undefined;

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // 展开约束名到具体类型
  const expandedConstraintTypes = useMemo(() => {
    if (!constraintTypes) return null;

    const buildableSet = new Set(buildableTypes);
    const expanded = new Set<string>();

    for (const t of constraintTypes) {
      if (buildableSet.has(t)) {
        expanded.add(t);
        continue;
      }
      // 展开约束到集合元素
      const elements = getConstraintElements(t);
      if (elements.length > 0) {
        elements.forEach(ct => expanded.add(ct));
      } else {
        expanded.add(t);
      }
    }

    return [...expanded];
  }, [constraintTypes, getConstraintElements, buildableTypes]);

  const typeGroups = useMemo(() => {
    // 构建匹配函数
    let matcher: (item: string) => boolean;
    if (!search) {
      matcher = () => true;
    } else if (useRegex) {
      try {
        const regex = new RegExp(search, 'i');
        matcher = (item) => regex.test(item);
      } catch {
        matcher = () => false;
      }
    } else {
      const lower = search.toLowerCase();
      matcher = (item) => item.toLowerCase().includes(lower);
    }

    // 如果有约束限制，显示约束 + 展开后的具体类型
    if (hasConstraint && expandedConstraintTypes && expandedConstraintTypes.length > 0) {
      const allItems = new Set<string>();
      if (constraintName) allItems.add(constraintName);
      expandedConstraintTypes.forEach(t => allItems.add(t));

      const items = [...allItems].filter(matcher).sort((a, b) => {
        if (a === constraintName) return -1;
        if (b === constraintName) return 1;
        return a.localeCompare(b);
      });

      if (items.length > 0) {
        return [{ label: constraintName || '可选类型', items }];
      }
      return [];
    }

    // 无约束，显示所有
    return buildTypeGroups(
      constraintDefs, buildableTypes,
      search, showConstraints, showTypes, useRegex
    );
  }, [hasConstraint, expandedConstraintTypes, constraintName, constraintDefs, buildableTypes, search, showConstraints, showTypes, useRegex]);

  const totalCount = useMemo(() =>
    typeGroups.reduce((sum, g) => sum + g.items.length, 0),
    [typeGroups]
  );

  return createPortal(
    <div
      ref={panelRef}
      className="fixed w-72 bg-gray-800 border border-gray-600 rounded shadow-xl"
      style={{ top: position.top, left: position.left, zIndex: 10000 }}
      onMouseDown={e => e.stopPropagation()}
    >
      {/* 搜索栏 */}
      <div className="p-2 border-b border-gray-700">
        <div className="flex items-center gap-1 bg-gray-700 rounded px-2 py-1">
          <input
            ref={inputRef}
            type="text"
            className="flex-1 text-xs bg-transparent focus:outline-none text-gray-200"
            placeholder="搜索..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            onKeyDown={e => e.key === 'Escape' && onClose()}
          />
          {/* 只在无约束时显示过滤按钮 */}
          {!hasConstraint && (
            <div className="flex gap-0.5 border-l border-gray-600 pl-1">
              <button
                type="button"
                className={`w-5 h-5 text-xs rounded ${showConstraints ? 'bg-blue-600 text-white' : 'text-gray-500'}`}
                onClick={() => setShowConstraints(v => !v)}
                title="约束"
              >C</button>
              <button
                type="button"
                className={`w-5 h-5 text-xs rounded ${showTypes ? 'bg-blue-600 text-white' : 'text-gray-500'}`}
                onClick={() => setShowTypes(v => !v)}
                title="类型"
              >T</button>
              <button
                type="button"
                className={`w-5 h-5 text-xs rounded ${useRegex ? 'bg-blue-600 text-white' : 'text-gray-500'}`}
                onClick={() => setUseRegex(v => !v)}
                title="正则"
              >.*</button>
            </div>
          )}
        </div>
      </div>

      {/* 包装选项 - 只在有允许的包装器时显示 */}
      {allowedWrappers.length > 0 && (
        <div className="px-2 py-1.5 border-b border-gray-700">
          <div className="text-xs text-gray-500 mb-1">包装为:</div>
          <div className="flex flex-wrap gap-1">
            {allowedWrappers.map(w => (
              <button
                key={w.name}
                type="button"
                className="text-xs px-1.5 py-0.5 rounded bg-gray-700 text-purple-400 hover:bg-gray-600"
                onClick={() => onWrap(w.name)}
              >
                +{w.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 列表 */}
      <div className="max-h-52 overflow-y-auto">
        {totalCount === 0 ? (
          <div className="p-3 text-xs text-gray-500 text-center">无匹配结果</div>
        ) : (
          typeGroups.map((group) => (
            <div key={group.label}>
              {/* 有约束时只有一个分组，不显示标签 */}
              {!(hasConstraint && typeGroups.length === 1) && (
                <div className="px-2 py-1 text-xs text-gray-500 bg-gray-900 sticky top-0">
                  {group.label}
                </div>
              )}
              {group.items.map(item => {
                const color = getTypeColor(item);
                return (
                  <button
                    key={item}
                    type="button"
                    className={`w-full text-left px-3 py-1 text-xs hover:bg-gray-700
                      ${item === currentValue ? 'bg-gray-700' : ''}`}
                    style={{ color }}
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

      <div className="px-2 py-1 text-xs text-gray-500 border-t border-gray-700">
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
    <span className="inline-flex items-center gap-0.5">
      {shape.map((d, i) => (
        <span key={i} className="inline-flex items-center">
          {i > 0 && <span className="text-gray-500">×</span>}
          <span className="relative group">
            <input
              type="text"
              className="w-6 text-xs bg-gray-700 rounded px-0.5 py-0.5 border border-gray-600
                focus:outline-none focus:border-blue-500 text-gray-200 text-center"
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
                className="absolute -top-1 -right-1 w-3 h-3 bg-red-600 rounded-full text-white 
                  text-[8px] opacity-0 group-hover:opacity-100"
                onClick={() => onChange(shape.filter((_, j) => j !== i))}
              >×</button>
            )}
          </span>
        </span>
      ))}
      <button
        type="button"
        className="w-4 h-4 text-[10px] bg-gray-700 hover:bg-gray-600 rounded text-gray-400"
        onClick={() => onChange([...shape, 4])}
      >+</button>
      <span className="text-gray-500">×</span>
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
        className="text-xs bg-gray-700 hover:bg-gray-600 rounded px-1.5 py-0.5 
          border border-gray-600 inline-flex items-center gap-1"
        style={{ color }}
        onClick={handleLeafClick}
      >
        {node.name}
        <span className="text-gray-500 text-[10px]">▼</span>
      </button>
    );
  }

  // 复合类型
  const w = WRAPPERS.find(x => x.name === node.wrapper);

  return (
    <span className="inline-flex items-center flex-wrap gap-0.5">
      <span className="inline-flex items-center group">
        <span className="text-purple-400">{node.wrapper}</span>
        <span className="text-gray-500">&lt;</span>
        <button
          type="button"
          className="w-3 h-3 text-[8px] bg-red-600 rounded text-white ml-0.5
            opacity-0 group-hover:opacity-100"
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

      <span className="text-gray-500">&gt;</span>
    </span>
  );
});

// ============ 约束分析工具 ============

/**
 * 分析约束类型，返回约束的特性
 * 
 * 约束分类：
 * 1. 无约束/AnyType：允许所有类型（标量 + 复合），显示完整选择面板
 * 2. 标量约束（如 SignlessIntegerLike）：只允许特定标量类型，不显示包装选项
 * 3. 复合类型约束（如 AnyTensor）：允许构建特定复合类型
 * 4. 元素类型约束（如 AnyTensorOf<[F32]>）：允许构建复合类型，但限制元素类型
 */
interface ConstraintAnalysis {
  /** 是否无约束或 AnyType */
  isUnconstrained: boolean;
  /** 是否是复合类型约束 */
  isCompositeConstraint: boolean;
  /** 允许的包装器（null 表示所有，[] 表示不允许） */
  allowedWrappers: string[] | null;
  /** 约束对应的具体标量类型（null 表示无限制） */
  scalarTypes: string[] | null;
}

function analyzeConstraint(
  constraint: string | undefined,
  getConstraintElements: (name: string) => string[],
  isShapedConstraint: (name: string) => boolean,
  getAllowedContainers: (name: string) => string[]
): ConstraintAnalysis {
  // 无约束或 AnyType
  if (!constraint || constraint === 'AnyType') {
    return {
      isUnconstrained: true,
      isCompositeConstraint: false,
      allowedWrappers: null,
      scalarTypes: null,
    };
  }

  // Variadic<...> 类型：解析内部类型
  const variadicMatch = constraint.match(/^Variadic<(.+)>$/);
  if (variadicMatch) {
    const innerConstraint = variadicMatch[1];
    return analyzeConstraint(innerConstraint, getConstraintElements, isShapedConstraint, getAllowedContainers);
  }

  // AnyOf<...> 类型：合成约束
  if (constraint.startsWith('AnyOf<')) {
    return {
      isUnconstrained: false,
      isCompositeConstraint: false,
      allowedWrappers: [],
      scalarTypes: null,
    };
  }

  // 使用 store 的方法判断
  const isShaped = isShapedConstraint(constraint);
  if (isShaped) {
    const containers = getAllowedContainers(constraint);
    const scalarTypes = getConstraintElements(constraint);
    return {
      isUnconstrained: false,
      isCompositeConstraint: true,
      allowedWrappers: containers.length > 0 ? containers : ['tensor', 'memref', 'vector'],
      scalarTypes: scalarTypes.length > 0 ? scalarTypes : null,
    };
  }

  // 标量约束
  const scalarTypes = getConstraintElements(constraint);
  return {
    isUnconstrained: false,
    isCompositeConstraint: false,
    allowedWrappers: [],
    scalarTypes: scalarTypes.length > 0 ? scalarTypes : [constraint],
  };
}

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
  const { getConstraintElements, isShapedConstraint, getAllowedContainers } = useTypeConstraintStore();

  // 分析约束
  const analysis = useMemo(() =>
    analyzeConstraint(constraint, getConstraintElements, isShapedConstraint, getAllowedContainers),
    [constraint, getConstraintElements, isShapedConstraint, getAllowedContainers]
  );

  // 允许的包装器
  const allowedWrappers = useMemo(() => {
    if (analysis.allowedWrappers === null) return WRAPPERS;
    return WRAPPERS.filter(w => analysis.allowedWrappers!.includes(w.name));
  }, [analysis.allowedWrappers]);
  const [isOpen, setIsOpen] = useState(false);
  const [selectorPos, setSelectorPos] = useState({ top: 0, left: 0 });
  // node 完全由 selectedType 派生（受控模式），无需本地 state
  const node = useMemo(() => parseType(selectedType || 'AnyType'), [selectedType]);
  const containerRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const clickedElementRef = useRef<Element | null>(null);

  // 约束对应的标量类型（用于选择面板）
  // 优先使用 prop 传入的 allowedTypes（来自后端 AnyTypeOf 解析）
  const constraintTypes = propAllowedTypes || analysis.scalarTypes;

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
    // 例如：vector<4xAnyFloat> + tensor => vector<4xtensor<4xAnyFloat>>
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
        className={`text-xs px-1.5 py-0.5 rounded ${className}`}
        style={{ color, backgroundColor: `${color}20`, border: `1px solid ${color}40` }}
      >
        {preview}
      </span>
    );
  }

  return (
    <>
      <div
        ref={containerRef}
        className={`inline-flex items-center gap-1 bg-gray-800 rounded px-2 py-1 
          border border-gray-600 ${className}`}
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
