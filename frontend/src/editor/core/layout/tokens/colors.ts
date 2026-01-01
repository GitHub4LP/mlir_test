/**
 * 颜色配置
 * colors, dialect, type, nodeType
 */

import type { ColorsConfig, DialectConfig, TypeConfig, NodeTypeConfig } from './types';

/** 颜色调色板 */
export const colors = {
  gray: {
    '50': '#f9fafb',
    '100': '#f3f4f6',
    '200': '#e5e7eb',
    '300': '#d1d5db',
    '400': '#9ca3af',
    '500': '#6b7280',
    '600': '#4b5563',
    '700': '#374151',
    '800': '#1f2937',
    '900': '#111827',
    '950': '#030712',
  },
  blue: {
    '400': '#60a5fa',
    '500': '#3b82f6',
    '600': '#2563eb',
  },
  green: {
    '400': '#4ade80',
    '500': '#22c55e',
    '600': '#16a34a',
  },
  red: {
    '400': '#f87171',
    '500': '#ef4444',
    '600': '#dc2626',
  },
  amber: {
    '500': '#f59e0b',
  },
  purple: {
    '500': '#a855f7',
  },
  white: '#ffffff',
  black: '#000000',
  transparent: 'transparent',
} as const satisfies ColorsConfig;

/** 方言颜色 */
export const dialect = {
  arith: '#4A90D9',
  func: '#50C878',
  scf: '#9B59B6',
  memref: '#E74C3C',
  tensor: '#1ABC9C',
  linalg: '#F39C12',
  vector: '#F1C40F',
  affine: '#E67E22',
  gpu: '#2ECC71',
  math: '#3498DB',
  cf: '#8E44AD',
  builtin: '#7F8C8D',
  default: '#3b82f6',
} as const satisfies DialectConfig;

/** 类型颜色 */
export const type = {
  I1: '#E74C3C',
  Index: '#50C878',
  BF16: '#3498DB',
  AnyType: '#F5F5F5',
  unsignedInteger: '#50C878',
  signlessInteger: '#52C878',
  signedInteger: '#2D8659',
  float: '#4A90D9',
  tensorFloat: '#5BA3E8',
  default: '#95A5A6',
} as const satisfies TypeConfig;

/** 节点类型颜色 */
export const nodeType = {
  entry: '#22c55e',
  return: '#ef4444',
  call: '#a855f7',
  operation: '#3b82f6',
} as const satisfies NodeTypeConfig;
