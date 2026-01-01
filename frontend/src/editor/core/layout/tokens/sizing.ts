/**
 * 尺寸配置
 * size, radius, border
 */

import type { SizeConfig, RadiusConfig, BorderConfig } from './types';

/** 通用尺寸 */
export const size = {
  '0': '0',
  '1': 4,
  '2': 8,
  '3': 12,
  '4': 16,
  '5': 20,
  '6': 24,
  '7': 28,
  '8': 32,
  '10': 40,
  '12': 48,
  '16': 64,
  '20': 80,
  '24': 96,
  xs: 12,
  sm: 14,
  base: 16,
  lg: 18,
  xl: 20,
} as const satisfies SizeConfig;

/** 圆角 */
export const radius = {
  none: '0',
  sm: 2,
  default: 4,
  md: 6,
  lg: 8,
  xl: 12,
  full: 9999,
} as const satisfies RadiusConfig;

/** 边框宽度 */
export const border = {
  thin: 1,
  medium: 2,
  thick: 3,
} as const satisfies BorderConfig;
