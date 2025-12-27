/**
 * 节点布局树构建
 * 将 GraphNode 转换为 LayoutNode 树，供布局引擎处理
 */

import type {
  LayoutNode,
  LayoutConfig,
} from './types';
import type {
  GraphNode,
  BlueprintNodeData,
  FunctionEntryData,
  FunctionReturnData,
  FunctionCallData,
  ArgumentDef,
} from '../../../types';
import { layoutConfig } from './LayoutConfig';

// ============================================================================
// 辅助类型
// ============================================================================

/** 引脚信息 */
interface PinInfo {
  handleId: string;
  label: string;
  typeConstraint: string;
  kind: 'exec' | 'data';
  isOutput: boolean;
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 获取节点的 header 背景色
 */
function getHeaderColor(node: GraphNode): string {
  switch (node.type) {
    case 'function-entry': {
      const data = node.data as FunctionEntryData;
      return data.isMain ? layoutConfig.nodeType.entryMain : layoutConfig.nodeType.entry;
    }
    case 'function-return': {
      const data = node.data as FunctionReturnData;
      return data.isMain ? layoutConfig.nodeType.returnMain : layoutConfig.nodeType.return;
    }
    case 'function-call':
      return layoutConfig.nodeType.call;
    case 'operation':
    default:
      return layoutConfig.nodeType.operation;
  }
}

/**
 * 获取节点标题
 */
function getNodeTitle(node: GraphNode): string {
  switch (node.type) {
    case 'operation': {
      const data = node.data as BlueprintNodeData;
      return data.operation.opName;
    }
    case 'function-entry': {
      const data = node.data as FunctionEntryData;
      return data.functionName || 'Entry';
    }
    case 'function-return': {
      const data = node.data as FunctionReturnData;
      return data.branchName ? `Return "${data.branchName}"` : 'Return';
    }
    case 'function-call': {
      const data = node.data as FunctionCallData;
      return data.functionName;
    }
    default:
      return 'Unknown';
  }
}

/**
 * 获取节点副标题
 */
function getNodeSubtitle(node: GraphNode): string | undefined {
  switch (node.type) {
    case 'operation': {
      const data = node.data as BlueprintNodeData;
      return data.operation.dialect;
    }
    case 'function-entry': {
      const data = node.data as FunctionEntryData;
      return data.isMain ? '(main)' : undefined;
    }
    case 'function-return': {
      const data = node.data as FunctionReturnData;
      return data.isMain ? '(main)' : undefined;
    }
    case 'function-call':
      return 'call';
    default:
      return undefined;
  }
}

/**
 * 获取显示的类型约束
 */
function getDisplayType(
  portName: string,
  originalConstraint: string,
  types: Record<string, string> | undefined
): string {
  return types?.[portName] || originalConstraint;
}

// ============================================================================
// 引脚收集
// ============================================================================

/**
 * 收集 Operation 节点的引脚
 */
function collectOperationPins(data: BlueprintNodeData): { inputs: PinInfo[]; outputs: PinInfo[] } {
  const inputs: PinInfo[] = [];
  const outputs: PinInfo[] = [];
  const op = data.operation;

  // 输入：execIn
  if (data.execIn) {
    inputs.push({
      handleId: 'exec-in',
      label: '',
      typeConstraint: 'exec',
      kind: 'exec',
      isOutput: false,
    });
  }

  // 输入：operands
  const operands = op.arguments.filter((a: ArgumentDef) => a.kind === 'operand');
  for (const operand of operands) {
    // 处理 variadic 端口
    if (operand.isVariadic) {
      const count = data.variadicCounts?.[operand.name] ?? 1;
      for (let i = 0; i < count; i++) {
        const handleId = `data-in-${operand.name}-${i}`;
        const portName = `${operand.name}[${i}]`;
        inputs.push({
          handleId,
          label: count > 1 ? `${operand.name}[${i}]` : operand.name,
          typeConstraint: getDisplayType(portName, operand.typeConstraint, data.inputTypes),
          kind: 'data',
          isOutput: false,
        });
      }
    } else {
      inputs.push({
        handleId: `data-in-${operand.name}`,
        label: operand.name,
        typeConstraint: getDisplayType(operand.name, operand.typeConstraint, data.inputTypes),
        kind: 'data',
        isOutput: false,
      });
    }
  }

  // 输出：execOuts
  for (const execOut of data.execOuts) {
    outputs.push({
      handleId: execOut.id,
      label: execOut.label || '',
      typeConstraint: 'exec',
      kind: 'exec',
      isOutput: true,
    });
  }

  // 输出：results
  for (let i = 0; i < op.results.length; i++) {
    const result = op.results[i];
    const resultName = result.name || `result_${i}`;
    
    // 处理 variadic results
    if (result.isVariadic) {
      const count = data.variadicCounts?.[resultName] ?? 1;
      for (let j = 0; j < count; j++) {
        const handleId = `data-out-${resultName}-${j}`;
        const portName = `${resultName}[${j}]`;
        outputs.push({
          handleId,
          label: count > 1 ? `${resultName}[${j}]` : resultName,
          typeConstraint: getDisplayType(portName, result.typeConstraint, data.outputTypes),
          kind: 'data',
          isOutput: true,
        });
      }
    } else {
      outputs.push({
        handleId: `data-out-${resultName}`,
        label: resultName,
        typeConstraint: getDisplayType(resultName, result.typeConstraint, data.outputTypes),
        kind: 'data',
        isOutput: true,
      });
    }
  }

  // 输出：regionPins (block args -> outputs)
  // 输入：regionPins (yield inputs)
  if (data.regionPins) {
    for (const regionPin of data.regionPins) {
      // Block args become output pins
      for (const blockArg of regionPin.blockArgOutputs) {
        outputs.push({
          handleId: blockArg.id,
          label: blockArg.label,
          typeConstraint: blockArg.typeConstraint,
          kind: 'data',
          isOutput: true,
        });
      }

      // Yield inputs become input pins
      if (regionPin.hasYieldInputs) {
        inputs.push({
          handleId: `region-${regionPin.regionName}-yield`,
          label: `${regionPin.regionName}_yield`,
          typeConstraint: 'inferred',
          kind: 'data',
          isOutput: false,
        });
      }
    }
  }

  return { inputs, outputs };
}

/**
 * 收集 FunctionEntry 节点的引脚
 */
function collectEntryPins(data: FunctionEntryData): { inputs: PinInfo[]; outputs: PinInfo[] } {
  const inputs: PinInfo[] = [];
  const outputs: PinInfo[] = [];

  // 输出：execOut
  outputs.push({
    handleId: 'exec-out',
    label: '',
    typeConstraint: 'exec',
    kind: 'exec',
    isOutput: true,
  });

  // 输出：parameters
  for (const param of data.outputs) {
    outputs.push({
      handleId: param.id || `data-out-${param.name}`,
      label: param.name,
      typeConstraint: getDisplayType(param.name, param.typeConstraint, data.outputTypes),
      kind: 'data',
      isOutput: true,
    });
  }

  return { inputs, outputs };
}

/**
 * 收集 FunctionReturn 节点的引脚
 */
function collectReturnPins(data: FunctionReturnData): { inputs: PinInfo[]; outputs: PinInfo[] } {
  const inputs: PinInfo[] = [];
  const outputs: PinInfo[] = [];

  // 输入：execIn
  inputs.push({
    handleId: 'exec-in',
    label: '',
    typeConstraint: 'exec',
    kind: 'exec',
    isOutput: false,
  });

  // 输入：return values
  for (const input of data.inputs) {
    inputs.push({
      handleId: input.id || `data-in-${input.name}`,
      label: input.name,
      typeConstraint: getDisplayType(input.name, input.typeConstraint, data.inputTypes),
      kind: 'data',
      isOutput: false,
    });
  }

  return { inputs, outputs };
}

/**
 * 收集 FunctionCall 节点的引脚
 */
function collectCallPins(data: FunctionCallData): { inputs: PinInfo[]; outputs: PinInfo[] } {
  const inputs: PinInfo[] = [];
  const outputs: PinInfo[] = [];

  // 输入：execIn
  inputs.push({
    handleId: 'exec-in',
    label: '',
    typeConstraint: 'exec',
    kind: 'exec',
    isOutput: false,
  });

  // 输入：parameters
  for (const input of data.inputs) {
    inputs.push({
      handleId: input.id || `data-in-${input.name}`,
      label: input.name,
      typeConstraint: getDisplayType(input.name, input.typeConstraint, data.inputTypes),
      kind: 'data',
      isOutput: false,
    });
  }

  // 输出：execOuts
  for (const execOut of data.execOuts) {
    outputs.push({
      handleId: execOut.id,
      label: execOut.label || '',
      typeConstraint: 'exec',
      kind: 'exec',
      isOutput: true,
    });
  }

  // 输出：return values
  for (const output of data.outputs) {
    outputs.push({
      handleId: output.id || `data-out-${output.name}`,
      label: output.name,
      typeConstraint: getDisplayType(output.name, output.typeConstraint, data.outputTypes),
      kind: 'data',
      isOutput: true,
    });
  }

  return { inputs, outputs };
}

/**
 * 收集节点的所有引脚
 */
function collectPins(node: GraphNode): { inputs: PinInfo[]; outputs: PinInfo[] } {
  switch (node.type) {
    case 'operation':
      return collectOperationPins(node.data as BlueprintNodeData);
    case 'function-entry':
      return collectEntryPins(node.data as FunctionEntryData);
    case 'function-return':
      return collectReturnPins(node.data as FunctionReturnData);
    case 'function-call':
      return collectCallPins(node.data as FunctionCallData);
    default:
      return { inputs: [], outputs: [] };
  }
}

// ============================================================================
// 布局树构建
// ============================================================================

/**
 * 构建文本节点
 */
function buildTextNode(
  type: string,
  content: string,
  style: { fontSize: number; fontWeight?: number; fill: string }
): LayoutNode {
  return {
    type,
    children: [],
    text: {
      content,
      fontSize: style.fontSize,
      fontWeight: style.fontWeight,
      fill: style.fill,
    },
  };
}

/**
 * 构建 Handle 节点
 */
function buildHandle(pin: PinInfo): LayoutNode {
  return {
    type: 'handle',
    children: [],
    interactive: {
      id: `handle-${pin.handleId}`,
      hitTestBehavior: 'opaque',
      cursor: 'crosshair',
    },
  };
}

/**
 * 构建 TypeLabel 节点
 * 
 * 结构：
 * typeLabel (有背景和 padding)
 * └── typeLabelText (文本节点，transparent 让点击穿透到父级)
 */
function buildTypeLabel(pin: PinInfo): LayoutNode {
  return {
    type: 'typeLabel',
    children: [
      {
        type: 'typeLabelText',
        children: [],
        text: {
          content: pin.typeConstraint,
          fontSize: 10,
          fill: '#ffffff',
        },
        // 让点击穿透到父级 typeLabel
        interactive: {
          id: '',
          hitTestBehavior: 'transparent',
        },
      },
    ],
    interactive: {
      id: `type-label-${pin.handleId}`,
      hitTestBehavior: 'opaque',
      cursor: 'pointer',
    },
  };
}

/**
 * 构建 PinContent 节点
 */
function buildPinContent(pin: PinInfo, side: 'left' | 'right', config: LayoutConfig): LayoutNode {
  const children: LayoutNode[] = [];

  // 标签
  if (pin.label) {
    children.push(buildTextNode('label', pin.label, config.text.label));
  }

  // 类型标签（仅 data 引脚）
  if (pin.kind === 'data') {
    children.push(buildTypeLabel(pin));
  }

  return {
    type: side === 'left' ? 'pinContent' : 'pinContentRight',
    children,
  };
}

/**
 * 构建左侧引脚组（Handle + Content）
 */
function buildLeftPinGroup(pin: PinInfo | null, config: LayoutConfig): LayoutNode {
  if (!pin) {
    return { type: 'leftPinGroup', children: [] };
  }
  return {
    type: 'leftPinGroup',
    children: [
      buildHandle(pin),
      buildPinContent(pin, 'left', config),
    ],
  };
}

/**
 * 构建右侧引脚组（Content + Handle）
 */
function buildRightPinGroup(pin: PinInfo | null, config: LayoutConfig): LayoutNode {
  if (!pin) {
    return { type: 'rightPinGroup', children: [] };
  }
  return {
    type: 'rightPinGroup',
    children: [
      buildPinContent(pin, 'right', config),
      buildHandle(pin),
    ],
  };
}

/**
 * 构建引脚行
 * 
 * 结构：
 * pinRow (horizontal, 透明, fill-parent)
 * ├── pinRowLeftSpacer (6px，透明)
 * ├── pinRowContent (fill-parent，有背景色)
 * │   ├── leftPinGroup [handle, content]
 * │   ├── pinRowSpacer (fill-parent)
 * │   └── rightPinGroup [content, handle]
 * └── pinRowRightSpacer (6px，透明)
 */
function buildPinRow(
  leftPin: PinInfo | null, 
  rightPin: PinInfo | null, 
  config: LayoutConfig,
  hasBottomRadius: boolean = false
): LayoutNode {
  // pinRowContent 的子节点
  const contentChildren: LayoutNode[] = [];
  
  // 左侧引脚组
  contentChildren.push(buildLeftPinGroup(leftPin, config));
  
  // 中间 spacer（fill-parent）
  contentChildren.push({
    type: 'pinRowSpacer',
    children: [],
  });
  
  // 右侧引脚组
  contentChildren.push(buildRightPinGroup(rightPin, config));
  
  // pinRowContent（有背景色）
  const pinRowContent: LayoutNode = {
    type: 'pinRowContent',
    children: contentChildren,
    // 动态设置圆角
    style: hasBottomRadius ? { cornerRadius: [0, 0, 8, 8] } : undefined,
  };
  
  // pinRow（透明，包含左右 spacer）
  return {
    type: 'pinRow',
    children: [
      { type: 'pinRowLeftSpacer', children: [] },
      pinRowContent,
      { type: 'pinRowRightSpacer', children: [] },
    ],
  };
}

/**
 * 构建 PinArea 节点
 * 
 * 结构：
 * pinArea (vertical)
 * ├── pinRow [leftPinGroup, rightPinGroup]  // 第一行
 * ├── pinRow [leftPinGroup, rightPinGroup]  // 第二行
 * └── ...
 * 
 * 配对规则（与 React Flow 一致）：
 * 1. 先配对 exec pins（execIn 与 execOuts）
 * 2. 再配对 data pins（operands 与 results）
 * 
 * @param hasBottomRadius - 如果为 true，最后一行的 pinRowContent 有下圆角
 */
function buildPinArea(node: GraphNode, config: LayoutConfig, hasBottomRadius: boolean = false): LayoutNode {
  const { inputs, outputs } = collectPins(node);

  // 分离 exec 和 data pins
  const execInputs = inputs.filter(p => p.kind === 'exec');
  const execOutputs = outputs.filter(p => p.kind === 'exec');
  const dataInputs = inputs.filter(p => p.kind === 'data');
  const dataOutputs = outputs.filter(p => p.kind === 'data');

  const rows: LayoutNode[] = [];
  const maxExec = Math.max(execInputs.length, execOutputs.length);
  const maxData = Math.max(dataInputs.length, dataOutputs.length);

  // 1. Exec pin rows
  // 如果没有 data pins，最后一个 exec row 需要 hasBottomRadius
  const execNeedsBottomRadius = hasBottomRadius && maxData === 0;
  for (let i = 0; i < maxExec; i++) {
    const leftPin = execInputs[i] || null;
    const rightPin = execOutputs[i] || null;
    const isLastExecRow = i === maxExec - 1;
    rows.push(buildPinRow(leftPin, rightPin, config, isLastExecRow && execNeedsBottomRadius));
  }

  // 2. Data pin rows
  for (let i = 0; i < maxData; i++) {
    const leftPin = dataInputs[i] || null;
    const rightPin = dataOutputs[i] || null;
    const isLastRow = i === maxData - 1;
    rows.push(buildPinRow(leftPin, rightPin, config, isLastRow && hasBottomRadius));
  }

  return {
    type: 'pinArea',
    children: rows,
  };
}

/**
 * 构建 Header 节点
 * 
 * 结构：
 * headerWrapper (horizontal, 透明, fill-parent)
 * ├── headerLeftSpacer (6px 宽, 透明)
 * ├── headerContent (fill-parent, 有背景色)
 * │   ├── titleGroup
 * │   ├── headerSpacer (fill-parent)
 * │   └── badgesGroup
 * └── headerRightSpacer (6px 宽, 透明)
 */
function buildHeader(node: GraphNode, config: LayoutConfig): LayoutNode {
  const title = getNodeTitle(node);
  const subtitle = getNodeSubtitle(node);
  const headerColor = getHeaderColor(node);

  // TitleGroup
  const titleGroupChildren: LayoutNode[] = [];
  if (subtitle) {
    titleGroupChildren.push(
      buildTextNode('subtitle', subtitle.toUpperCase(), config.text.subtitle)
    );
  }
  titleGroupChildren.push(buildTextNode('title', title, config.text.title));

  const titleGroup: LayoutNode = {
    type: 'titleGroup',
    children: titleGroupChildren,
  };

  // Spacer
  const spacer: LayoutNode = {
    type: 'headerSpacer',
    children: [],
  };

  // BadgesGroup（暂时为空）
  const badgesGroup: LayoutNode = {
    type: 'badgesGroup',
    children: [],
  };

  // headerContent（有背景色，颜色在这里设置）
  const headerContent: LayoutNode = {
    type: 'headerContent',
    children: [titleGroup, spacer, badgesGroup],
    // 设置样式，包含动态的 headerColor
    style: {
      fill: headerColor,
      cornerRadius: config.headerContent.cornerRadius,
    },
  };

  // headerWrapper（透明，包含左右 spacer）
  return {
    type: 'headerWrapper',
    children: [
      { type: 'headerLeftSpacer', children: [] },
      headerContent,
      { type: 'headerRightSpacer', children: [] },
    ],
  };
}

/**
 * 构建 AttrWrapper 节点（Operation 节点的属性区域）
 * 
 * 结构：
 * attrWrapper (horizontal, 透明)
 * ├── attrLeftSpacer (6px，透明)
 * ├── attrContent (fill-parent，有背景色)
 * │   ├── labelColumn (vertical, right-aligned)
 * │   └── valueColumn (vertical, fill-parent)
 * └── attrRightSpacer (6px，透明)
 */
function buildAttrWrapper(node: GraphNode, config: LayoutConfig, hasBottomRadius: boolean): LayoutNode | null {
  if (node.type !== 'operation') return null;

  const data = node.data as BlueprintNodeData;
  const attrs = data.operation.arguments.filter((a: ArgumentDef) => a.kind === 'attribute');

  if (attrs.length === 0) return null;

  // 构建 label 列
  const labelChildren: LayoutNode[] = attrs.map((attr: ArgumentDef) => 
    buildTextNode('attrLabel', attr.name, config.text.label)
  );

  // 构建 value 列
  const valueChildren: LayoutNode[] = attrs.map((attr: ArgumentDef) => {
    const value = data.attributes[attr.name] || '';
    return buildTextNode('attrValue', value || '(empty)', config.text.muted);
  });

  // attrContent（有背景色）
  const attrContent: LayoutNode = {
    type: 'attrContent',
    children: [
      { type: 'labelColumn', children: labelChildren },
      { type: 'valueColumn', children: valueChildren },
    ],
    // 动态设置圆角
    style: hasBottomRadius ? { cornerRadius: [0, 0, 8, 8] } : undefined,
  };

  // attrWrapper（透明，包含左右 spacer）
  return {
    type: 'attrWrapper',
    children: [
      { type: 'attrLeftSpacer', children: [] },
      attrContent,
      { type: 'attrRightSpacer', children: [] },
    ],
  };
}

/**
 * 构建 SummaryWrapper 节点
 * 
 * 结构：
 * summaryWrapper (horizontal, 透明)
 * ├── summaryLeftSpacer (6px，透明)
 * ├── summaryContent (fill-parent，有背景色，下圆角)
 * │   └── summaryText
 * └── summaryRightSpacer (6px，透明)
 */
function buildSummaryWrapper(node: GraphNode, config: LayoutConfig): LayoutNode | null {
  if (node.type !== 'operation') return null;

  const data = node.data as BlueprintNodeData;
  const summary = data.operation.summary;

  if (!summary) return null;

  // summaryContent（有背景色，下圆角）
  const summaryContent: LayoutNode = {
    type: 'summaryContent',
    children: [
      {
        type: 'summaryText',
        children: [],
        text: {
          content: summary,
          fontSize: config.text.muted.fontSize,
          fill: config.text.muted.fill,
        },
      },
    ],
  };

  // summaryWrapper（透明，包含左右 spacer）
  return {
    type: 'summaryWrapper',
    children: [
      { type: 'summaryLeftSpacer', children: [] },
      summaryContent,
      { type: 'summaryRightSpacer', children: [] },
    ],
  };
}

// ============================================================================
// 主函数
// ============================================================================

/**
 * 构建节点布局树
 * @param node - GraphNode
 * @param config - 布局配置（可选，默认使用全局配置）
 * @returns LayoutNode 树
 * 
 * 结构：
 * node (透明容器，无背景)
 * ├── headerWrapper (透明)
 * │   ├── headerLeftSpacer (6px，透明)
 * │   ├── headerContent (fill-parent，有背景色，上圆角)
 * │   └── headerRightSpacer (6px，透明)
 * ├── pinArea
 * │   └── pinRow (透明)
 * │       ├── pinRowLeftSpacer (6px，透明)
 * │       ├── pinRowContent (fill-parent，有背景色)
 * │       │   ├── leftPinGroup [handle, content]
 * │       │   ├── pinRowSpacer (fill-parent)
 * │       │   └── rightPinGroup [content, handle]
 * │       └── pinRowRightSpacer (6px，透明)
 * ├── attrWrapper (可选，透明)
 * │   ├── attrLeftSpacer (6px，透明)
 * │   ├── attrContent (fill-parent，有背景色，可能有下圆角)
 * │   └── attrRightSpacer (6px，透明)
 * └── summaryWrapper (可选，透明)
 *     ├── summaryLeftSpacer (6px，透明)
 *     ├── summaryContent (fill-parent，有背景色，下圆角)
 *     └── summaryRightSpacer (6px，透明)
 * 
 * 关键设计：
 * - handle 在 pinRow 边缘，中心距离 node 边缘 6px
 * - headerContent 和 pinRowContent 有背景色，宽度 = node 宽度 - 12px
 * - handle 中心正好在背景边缘
 * - 最后一个有背景的区域有下圆角
 */
export function buildNodeLayoutTree(
  node: GraphNode,
  config: LayoutConfig = layoutConfig
): LayoutNode {
  const children: LayoutNode[] = [];

  // 先判断有哪些可选区域
  const hasSummary = node.type === 'operation' && !!(node.data as BlueprintNodeData).operation.summary;
  const hasAttrs = node.type === 'operation' && 
    (node.data as BlueprintNodeData).operation.arguments.some((a: ArgumentDef) => a.kind === 'attribute');

  // 1. Header
  children.push(buildHeader(node, config));

  // 2. PinArea - 如果没有 attr 和 summary，最后一个 pinRowContent 需要下圆角
  const needsPinAreaBottomRadius = !hasAttrs && !hasSummary;
  children.push(buildPinArea(node, config, needsPinAreaBottomRadius));

  // 3. AttrWrapper（仅 Operation）- 如果没有 summary，attrContent 需要下圆角
  if (hasAttrs) {
    const attrWrapper = buildAttrWrapper(node, config, !hasSummary);
    if (attrWrapper) {
      children.push(attrWrapper);
    }
  }

  // 4. SummaryWrapper（仅 Operation）- summaryContent 总是有下圆角
  if (hasSummary) {
    const summaryWrapper = buildSummaryWrapper(node, config);
    if (summaryWrapper) {
      children.push(summaryWrapper);
    }
  }

  // node 层：透明容器，无背景
  return {
    type: 'node',
    children,
    interactive: {
      id: `node-${node.id}`,
      hitTestBehavior: 'translucent',
    },
  };
}

/**
 * 检查节点是否为 Entry 类型
 */
export function isEntryNode(node: GraphNode): boolean {
  return node.type === 'function-entry';
}

/**
 * 检查节点是否为 Return 类型
 */
export function isReturnNode(node: GraphNode): boolean {
  return node.type === 'function-return';
}

/**
 * 检查节点是否支持参数添加/删除
 */
export function supportsParamEdit(node: GraphNode): boolean {
  if (node.type === 'function-entry') {
    const data = node.data as FunctionEntryData;
    return !data.isMain;
  }
  return false;
}

/**
 * 检查节点是否支持返回值添加/删除
 */
export function supportsReturnEdit(node: GraphNode): boolean {
  if (node.type === 'function-return') {
    const data = node.data as FunctionReturnData;
    return !data.isMain;
  }
  return false;
}
