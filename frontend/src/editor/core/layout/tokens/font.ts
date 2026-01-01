/**
 * 字体配置
 */

import type { FontConfig } from './types';

/** 字体配置 */
export const font = {
  family: {
    sans: 'system-ui, -apple-system, sans-serif',
    mono: 'ui-monospace, monospace',
  },
  size: {
    xs: 10,
    sm: 12,
    base: 14,
    lg: 16,
    xl: 18,
  },
  weight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  lineHeight: {
    tight: '1.25',
    normal: '1.5',
    relaxed: '1.75',
  },
} as const satisfies FontConfig;
