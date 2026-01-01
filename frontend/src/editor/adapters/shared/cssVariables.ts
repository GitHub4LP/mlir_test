/**
 * CSS 变量生成
 * 从 LayoutConfig 生成 CSS 变量，供 ReactFlow/VueFlow 使用
 */

import type { LayoutConfig, Padding, CornerRadius } from '../../core/layout/types';

/**
 * 格式化内边距为 CSS 字符串
 */
export function formatPadding(padding: Padding | undefined): string {
  if (padding === undefined) return '0';
  if (typeof padding === 'number') return `${padding}px`;
  return padding.map((p) => `${p}px`).join(' ');
}

/**
 * 格式化圆角为 CSS 字符串
 */
export function formatCornerRadius(radius: CornerRadius | undefined): string {
  if (radius === undefined) return '0';
  if (typeof radius === 'number') return `${radius}px`;
  return radius.map((r) => `${r}px`).join(' ');
}

/**
 * 从 LayoutConfig 生成 CSS 变量
 * @param config - 布局配置
 * @returns CSS 变量对象
 */
export function toCSSVariables(config: LayoutConfig): Record<string, string> {
  const vars: Record<string, string> = {};

  // ============================================================
  // Node
  // ============================================================
  vars['--node-min-width'] = `${config.node.minWidth ?? 200}px`;
  vars['--node-padding'] = formatPadding(config.node.padding);
  vars['--node-bg'] = config.node.fill ?? '#2d2d3d';
  vars['--node-border-color'] = config.node.stroke ?? '#3d3d4d';
  vars['--node-border-width'] = `${config.node.strokeWidth ?? 1}px`;
  vars['--node-border-radius'] = formatCornerRadius(config.node.cornerRadius);

  // Node selected state
  if (config.node.selected) {
    vars['--node-selected-border-color'] = config.node.selected.stroke ?? '#60a5fa';
    vars['--node-selected-border-width'] = `${config.node.selected.strokeWidth ?? 2}px`;
  }

  // ============================================================
  // Header
  // ============================================================
  vars['--header-height'] = typeof config.headerWrapper.minHeight === 'number' 
    ? `${config.headerWrapper.minHeight}px` 
    : 'auto';
  vars['--header-padding'] = formatPadding(config.headerWrapper.padding);

  // ============================================================
  // Title Group
  // ============================================================
  vars['--title-group-spacing'] = typeof config.titleGroup.spacing === 'number'
    ? `${config.titleGroup.spacing}px`
    : '0';

  // ============================================================
  // Pin Area
  // ============================================================
  vars['--pin-area-padding'] = formatPadding(config.pinArea.padding);
  vars['--pin-area-spacing'] = typeof config.pinArea.spacing === 'number'
    ? `${config.pinArea.spacing}px`
    : '0';

  // ============================================================
  // Pin Row
  // ============================================================
  vars['--pin-row-min-height'] = `${config.pinRow.minHeight ?? 28}px`;
  vars['--pin-row-padding'] = formatPadding(config.pinRow.padding);

  // ============================================================
  // Pin Content
  // ============================================================
  vars['--pin-content-spacing'] = typeof config.pinContent.spacing === 'number'
    ? `${config.pinContent.spacing}px`
    : '0';
  vars['--pin-content-margin-left'] = formatPadding(config.pinContent.padding);
  vars['--pin-content-margin-right'] = formatPadding(config.pinContentRight.padding);

  // ============================================================
  // Handle
  // ============================================================
  vars['--handle-size'] = `${config.handle.width ?? 12}px`;
  vars['--handle-stroke-width'] = `${config.handle.strokeWidth ?? 2}px`;

  // ============================================================
  // Type Label
  // ============================================================
  vars['--type-label-padding'] = formatPadding(config.typeLabel.padding);
  vars['--type-label-min-width'] = `${config.typeLabel.minWidth ?? 40}px`;
  vars['--type-label-border-radius'] = formatCornerRadius(config.typeLabel.cornerRadius);
  vars['--type-label-bg'] = config.typeLabel.fill ?? 'rgba(55, 65, 81, 1)';

  // ============================================================
  // Attr Area
  // ============================================================
  vars['--attr-area-padding'] = formatPadding(config.attrArea.padding);
  vars['--attr-area-spacing'] = typeof config.attrArea.spacing === 'number'
    ? `${config.attrArea.spacing}px`
    : '0';

  // ============================================================
  // Attr Row
  // ============================================================
  vars['--attr-row-height'] = typeof config.attrWrapper.minHeight === 'number'
    ? `${config.attrWrapper.minHeight}px`
    : 'auto';
  vars['--attr-row-spacing'] = typeof config.attrWrapper.spacing === 'number'
    ? `${config.attrWrapper.spacing}px`
    : '0';

  // ============================================================
  // Attr Label
  // ============================================================
  vars['--attr-label-width'] = typeof config.attrLabel.width === 'number'
    ? `${config.attrLabel.width}px`
    : 'auto';

  // ============================================================
  // Text
  // ============================================================
  vars['--font-family'] = config.text.fontFamily;
  vars['--title-font-size'] = `${config.text.title.fontSize}px`;
  vars['--title-font-weight'] = `${config.text.title.fontWeight ?? 600}`;
  vars['--title-color'] = config.text.title.fill;
  vars['--subtitle-font-size'] = `${config.text.subtitle.fontSize}px`;
  vars['--subtitle-font-weight'] = `${config.text.subtitle.fontWeight ?? 500}`;
  vars['--subtitle-color'] = config.text.subtitle.fill;
  vars['--label-font-size'] = `${config.text.label.fontSize}px`;
  vars['--label-color'] = config.text.label.fill;
  vars['--muted-color'] = config.text.muted.fill;

  // ============================================================
  // Edge
  // ============================================================
  vars['--edge-exec-color'] = config.edge.exec.stroke ?? '#ffffff';
  vars['--edge-exec-width'] = `${config.edge.exec.strokeWidth}px`;
  vars['--edge-data-width'] = `${config.edge.data.strokeWidth}px`;
  vars['--edge-data-default-color'] = config.edge.data.defaultStroke ?? '#888888';
  vars['--edge-selected-width'] = `${config.edge.selected.strokeWidth}px`;
  vars['--edge-bezier-offset'] = `${config.edge.bezierOffset}`;

  // ============================================================
  // Node Types
  // ============================================================
  vars['--node-type-entry'] = config.nodeType.entry;
  vars['--node-type-return'] = config.nodeType.return;
  vars['--node-type-call'] = config.nodeType.call;
  vars['--node-type-operation'] = config.nodeType.operation;

  return vars;
}

/**
 * 将 CSS 变量对象转换为 CSS 字符串
 * @param vars - CSS 变量对象
 * @returns CSS 字符串
 */
export function cssVariablesToString(vars: Record<string, string>): string {
  return Object.entries(vars)
    .map(([key, value]) => `${key}: ${value};`)
    .join('\n');
}

/**
 * 将 CSS 变量应用到元素
 * @param element - DOM 元素
 * @param vars - CSS 变量对象
 */
export function applyCSSVariables(
  element: HTMLElement,
  vars: Record<string, string>
): void {
  for (const [key, value] of Object.entries(vars)) {
    element.style.setProperty(key, value);
  }
}

/**
 * 生成 CSS 变量声明块
 * @param config - 布局配置
 * @param selector - CSS 选择器（默认 :root）
 * @returns CSS 字符串
 */
export function generateCSSVariablesBlock(
  config: LayoutConfig,
  selector: string = ':root'
): string {
  const vars = toCSSVariables(config);
  const content = cssVariablesToString(vars);
  return `${selector} {\n${content.split('\n').map(line => `  ${line}`).join('\n')}\n}`;
}
