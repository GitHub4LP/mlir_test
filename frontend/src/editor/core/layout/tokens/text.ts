/**
 * 文本样式配置
 */

import type { TextConfig } from './types';

/** 文本配置 */
export const text = {
  fontFamily: 'system-ui, -apple-system, sans-serif',
  title: {
    fontSize: 14,
    fontWeight: 600,
    fill: '#ffffff',
  },
  subtitle: {
    fontSize: 10,
    fontWeight: 500,
    fill: 'rgba(255,255,255,0.7)',
  },
  label: {
    fontSize: 12,
    fill: '#d1d5db',
  },
  muted: {
    fontSize: 12,
    fill: '#6b7280',
  },
} as const satisfies TextConfig;
