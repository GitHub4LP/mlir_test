/**
 * 边配置
 */

import type { EdgeConfig } from './types';

/** 边配置 */
export const edge = {
  exec: {
    stroke: '#ffffff',
    strokeWidth: 2,
  },
  data: {
    strokeWidth: 2,
    defaultStroke: '#888888',
  },
  selected: {
    strokeWidth: 3,
  },
  bezierOffset: 100,
} as const satisfies EdgeConfig;
