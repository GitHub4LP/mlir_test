/**
 * UI 组件配置
 * overlay, ui, canvas, minimap, buttonStyle
 */

import type { OverlayConfig, UIConfig, CanvasConfig, MinimapConfig, ButtonConfig } from './types';

/** 弹出层样式 */
export const overlay = {
  bg: '#1f2937',
  borderColor: '#4b5563',
  borderWidth: 1,
  borderRadius: 8,
  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
  padding: 8,
} as const satisfies OverlayConfig;

/** UI 组件尺寸 */
export const ui = {
  listItemHeight: 28,
  searchHeight: 28,
  smallButtonHeight: 24,
  rowHeight: 28,
  labelWidth: 80,
  gap: 8,
  smallGap: 6,
  scrollbarWidth: 6,
  panelWidthNarrow: 240,
  panelWidthMedium: 280,
  panelMaxHeight: 320,
  shadowBlur: 16,
  shadowColor: 'rgba(0, 0, 0, 0.5)',
  darkBg: '#111827',
  buttonBg: '#374151',
  buttonHoverBg: '#4b5563',
  successColor: '#22c55e',
  successHoverColor: '#16a34a',
  cursorBlinkInterval: 530,
  minScrollbarHeight: 20,
  closeButtonOffset: 24,
  closeButtonSize: 10,
  titleLeftPadding: 12,
  colorDotRadius: 4,
  colorDotGap: 16,
} as const satisfies UIConfig;

/** 画布样式 */
export const canvas = {
  bg: '#030712',
} as const satisfies CanvasConfig;

/** 小地图样式 */
export const minimap = {
  nodeColor: '#4a5568',
  selectedNodeColor: '#4299e1',
  viewportColor: 'rgba(66, 153, 225, 0.3)',
  viewportBorderColor: 'rgba(66, 153, 225, 0.8)',
  bg: '#1a1a1a',
} as const satisfies MinimapConfig;

/** 按钮样式 */
export const buttonStyle = {
  size: 16,
  iconSize: 12,
  borderRadius: 4,
  bg: 'rgba(255, 255, 255, 0.1)',
  hoverBg: 'rgba(255, 255, 255, 0.2)',
  borderColor: 'rgba(255, 255, 255, 0.3)',
  borderWidth: 1,
  textColor: '#ffffff',
  fontSize: 12,
  danger: {
    color: '#ef4444',
    hoverColor: '#f87171',
  },
} as const satisfies ButtonConfig;
