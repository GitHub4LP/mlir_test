/**
 * 类型颜色映射服务
 * 
 * 设计原则：
 * 1. 使用规则匹配（正则表达式）而不是逐个配置，自动覆盖所有类型
 * 2. 复合类型（约束）通过展开到基本类型集合，然后对颜色进行平均
 * 
 * 颜色方案：
 * - Bool (I1): 红色
 * - SignlessInteger (I*): 绿色系（中间色调）
 * - UnsignedInteger (UI*) + Index: 纯绿（Index 看作无符号整型）
 * - SignedInteger (SI*): 暗绿
 * - Float (F*, BF*, TF*): 蓝色系
 * - AnyType: 比纯白稍灰（#F5F5F5），表示任意类型
 * 
 * 注意：
 * - NoneType 不应该出现在类型约束系统中（没有可展开的集合元素）
 * - 执行线使用纯白色（#FFFFFF），在 CustomEdge.tsx 中处理
 */

// ============================================================================
// 颜色配置（基于规则）
// ============================================================================

/**
 * 类型颜色规则
 * 按优先级顺序匹配，第一个匹配的规则生效
 */
interface ColorRule {
  /** 匹配规则（正则表达式或精确匹配） */
  pattern: RegExp | string;
  /** 颜色值 */
  color: string;
  /** 规则描述 */
  description: string;
}

const COLOR_RULES: ColorRule[] = [
  // ===== Boolean =====
  {
    pattern: /^I1$/,
    color: '#E74C3C',  // 红色
    description: 'Boolean (I1)',
  },

  // ===== UnsignedInteger (UI*) + Index =====
  // Index 看作无符号整型，使用纯绿
  {
    pattern: /^UI\d+$/,
    color: '#50C878',  // 纯绿
    description: 'UnsignedInteger (UI*)',
  },
  {
    pattern: /^Index$/,
    color: '#50C878',  // 纯绿（与无符号整型相同）
    description: 'Index (看作无符号整型)',
  },

  // ===== SignlessInteger (I*) =====
  // 绿色系，中间色调（介于纯绿和暗绿之间）
  {
    pattern: /^I\d+$/,
    color: '#52C878',  // 中等绿色
    description: 'SignlessInteger (I*)',
  },

  // ===== SignedInteger (SI*) =====
  // 暗绿（偏暗）
  {
    pattern: /^SI\d+$/,
    color: '#2D8659',  // 暗绿
    description: 'SignedInteger (SI*)',
  },

  // ===== Float (F*) =====
  // 蓝色系，中间色调
  {
    pattern: /^F\d+$/,
    color: '#4A90D9',  // 中等蓝色
    description: 'Float (F*)',
  },

  // ===== BFloat16 =====
  // 蓝色系，稍暗
  {
    pattern: /^BF16$/,
    color: '#3498DB',  // 稍暗的蓝色
    description: 'BFloat16',
  },

  // ===== TensorFloat (TF*) =====
  // 蓝色系，稍亮
  {
    pattern: /^TF\d+$/,
    color: '#5BA3E8',  // 稍亮的蓝色
    description: 'TensorFloat (TF*)',
  },

  // ===== AnyType =====
  // 比纯白稍灰，表示任意类型（包含所有类型）
  {
    pattern: /^AnyType$/,
    color: '#F5F5F5',  // 比纯白稍灰
    description: 'AnyType (任意类型)',
  },
];

// ============================================================================
// 颜色工具函数
// ============================================================================

/**
 * 将 hex 颜色转换为 RGB
 */
function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) {
    throw new Error(`Invalid hex color: ${hex}`);
  }
  return {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16),
  };
}

/**
 * 将 RGB 转换为 hex 颜色
 */
function rgbToHex(rgb: { r: number; g: number; b: number }): string {
  const toHex = (n: number) => {
    const hex = Math.round(Math.max(0, Math.min(255, n))).toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  };
  return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`.toUpperCase();
}

/**
 * 对多个颜色进行 RGB 平均
 */
function averageColors(colors: string[]): string {
  if (colors.length === 0) {
    return '#95A5A6'; // 默认灰色
  }
  if (colors.length === 1) {
    return colors[0];
  }

  const rgbs = colors.map(hexToRgb);
  const avg = rgbs.reduce(
    (acc, rgb) => ({
      r: acc.r + rgb.r,
      g: acc.g + rgb.g,
      b: acc.b + rgb.b,
    }),
    { r: 0, g: 0, b: 0 }
  );

  const count = colors.length;
  return rgbToHex({
    r: avg.r / count,
    g: avg.g / count,
    b: avg.b / count,
  });
}

// ============================================================================
// 主要颜色获取函数
// ============================================================================

/**
 * 获取 BuildableType 的基础颜色
 * 
 * 使用规则匹配，自动覆盖所有类型
 * 
 * @param buildableType - BuildableType 名称（如 "I32", "F32", "Index"）
 * @returns hex 颜色字符串，如果未匹配则返回 null
 */
export function getBuildableTypeColor(buildableType: string): string | null {
  for (const rule of COLOR_RULES) {
    if (typeof rule.pattern === 'string') {
      if (buildableType === rule.pattern) {
        return rule.color;
      }
    } else {
      if (rule.pattern.test(buildableType)) {
        return rule.color;
      }
    }
  }
  return null;
}

/**
 * 获取类型约束的颜色
 * 
 * 策略：
 * 1. 如果是 BuildableType，直接返回配置的颜色
 * 2. 如果是约束，展开到 BuildableType 集合
 * 3. 单一类型 → 直接返回该类型颜色
 * 4. 多个类型 → RGB 平均
 * 
 * @param displayType - 显示类型（可能是 BuildableType 或约束）
 * @param getConstraintElements - 获取约束的元素列表函数
 * @returns hex 颜色字符串
 */
export function getTypeColor(
  displayType: string,
  getConstraintElements: (constraint: string) => string[]
): string {
  if (!displayType) {
    return '#95A5A6'; // 默认灰色
  }

  // 1. 如果是 BuildableType，直接返回配置的颜色
  const directColor = getBuildableTypeColor(displayType);
  if (directColor) {
    return directColor;
  }

  // 2. 展开约束到类型约束集合元素
  const elements = getConstraintElements(displayType);

  // 3. 单一元素 → 直接返回该元素颜色
  if (elements.length === 1) {
    const color = getBuildableTypeColor(elements[0]);
    return color || '#95A5A6';
  }

  // 4. 多个元素 → 对颜色进行 RGB 平均
  if (elements.length > 1) {
    const colors = elements
      .map(t => getBuildableTypeColor(t))
      .filter((c): c is string => c !== null); // 过滤未定义的颜色

    if (colors.length === 0) {
      return '#95A5A6'; // 默认灰色
    }
    if (colors.length === 1) {
      return colors[0];
    }

    return averageColors(colors);
  }

  // 5. 无法展开或未知类型 → 默认灰色
  return '#95A5A6';
}

/**
 * 获取所有颜色规则
 */
export function getColorRules(): ReadonlyArray<ColorRule> {
  return COLOR_RULES;
}

