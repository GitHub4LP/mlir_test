/**
 * 约束规则解析器
 * 
 * 后端返回结构化规则，前端按需递归展开
 */

// ============ 规则类型定义 ============

export type ConstraintRule =
  | { kind: 'type'; name: string }                    // 具体类型: I32
  | { kind: 'oneOf'; types: string[] }                // 类型枚举: [I1, I8, ...]
  | { kind: 'or'; children: ConstraintRule[] }        // 并集
  | { kind: 'and'; children: ConstraintRule[] }       // 交集
  | { kind: 'ref'; name: string }                     // 引用其他约束
  | { kind: 'shaped'; container: string; element?: ConstraintRule; ranked?: boolean }  // 容器
  | { kind: 'like'; element: ConstraintRule }         // 标量或其容器
  | { kind: 'any' };                                  // 任意类型

export interface ConstraintDef {
  name: string;
  summary: string;
  rule: ConstraintRule | null;
}

// ============ 规则展开 ============

/**
 * 展开规则到具体标量类型列表
 * 
 * @param rule 规则节点
 * @param defs 所有约束定义（用于解析 ref）
 * @param buildableTypes 所有可构建类型
 * @param visited 已访问的 ref（防止循环）
 */
export function expandRule(
  rule: ConstraintRule,
  defs: Map<string, ConstraintDef>,
  buildableTypes: string[],
  visited: Set<string> = new Set()
): string[] {
  switch (rule.kind) {
    case 'type':
      return [rule.name];
    
    case 'oneOf':
      return [...rule.types];  // 返回副本
    
    case 'or': {
      const results = rule.children.flatMap(c => expandRule(c, defs, buildableTypes, visited));
      return [...new Set(results)];
    }
    
    case 'and': {
      const childResults = rule.children.map(c => expandRule(c, defs, buildableTypes, visited));
      if (childResults.length === 0) return [];
      return childResults.reduce((acc, curr) => acc.filter(t => curr.includes(t)));
    }
    
    case 'ref': {
      if (visited.has(rule.name)) return [];
      visited.add(rule.name);
      const def = defs.get(rule.name);
      if (!def?.rule) return [];
      return expandRule(def.rule, defs, buildableTypes, visited);
    }
    
    case 'shaped':
      return [];
    
    case 'like':
      return expandRule(rule.element, defs, buildableTypes, visited);
    
    case 'any':
      return [...buildableTypes];  // 返回副本
  }
}

/**
 * 获取约束映射到的类型约束集合元素
 */
export function getConstraintElements(
  constraintName: string,
  defs: Map<string, ConstraintDef>,
  buildableTypes: string[]
): string[] {
  const def = defs.get(constraintName);
  if (!def?.rule) {
    // 没有规则，检查是否是 BuildableType
    if (buildableTypes.includes(constraintName)) {
      return [constraintName];
    }
    return [];
  }
  return expandRule(def.rule, defs, buildableTypes);
}

/**
 * 检查约束是否是复合类型约束（shaped）
 */
export function isShapedConstraint(
  constraintName: string,
  defs: Map<string, ConstraintDef>
): boolean {
  const def = defs.get(constraintName);
  if (!def?.rule) return false;
  return checkRuleIsShaped(def.rule, defs, new Set());
}

function checkRuleIsShaped(
  rule: ConstraintRule,
  defs: Map<string, ConstraintDef>,
  visited: Set<string>
): boolean {
  switch (rule.kind) {
    case 'shaped':
      return true;
    case 'like':
      return true; // like 允许 shaped
    case 'or':
      return rule.children.some(c => checkRuleIsShaped(c, defs, visited));
    case 'and':
      return rule.children.some(c => checkRuleIsShaped(c, defs, visited));
    case 'ref': {
      if (visited.has(rule.name)) return false;
      visited.add(rule.name);
      const def = defs.get(rule.name);
      if (!def?.rule) return false;
      return checkRuleIsShaped(def.rule, defs, visited);
    }
    default:
      return false;
  }
}

/**
 * 获取约束允许的容器类型
 */
export function getAllowedContainers(
  constraintName: string,
  defs: Map<string, ConstraintDef>
): string[] {
  const def = defs.get(constraintName);
  if (!def?.rule) return [];
  return collectContainers(def.rule, defs, new Set());
}

function collectContainers(
  rule: ConstraintRule,
  defs: Map<string, ConstraintDef>,
  visited: Set<string>
): string[] {
  switch (rule.kind) {
    case 'shaped':
      if (rule.container === 'shaped') {
        // shaped 表示任意容器
        return ['tensor', 'memref', 'vector'];
      }
      return [rule.container];
    case 'like':
      // like 允许任意容器
      return ['tensor', 'memref', 'vector'];
    case 'or':
      return [...new Set(rule.children.flatMap(c => collectContainers(c, defs, visited)))];
    case 'and': {
      // 交集：取公共容器
      const childContainers = rule.children.map(c => collectContainers(c, defs, visited));
      if (childContainers.length === 0) return [];
      return childContainers.reduce((acc, curr) => acc.filter(c => curr.includes(c)));
    }
    case 'ref': {
      if (visited.has(rule.name)) return [];
      visited.add(rule.name);
      const def = defs.get(rule.name);
      if (!def?.rule) return [];
      return collectContainers(def.rule, defs, visited);
    }
    default:
      return [];
  }
}

/**
 * 获取约束的元素类型约束（用于 shaped 类型）
 */
export function getElementConstraint(
  constraintName: string,
  defs: Map<string, ConstraintDef>
): ConstraintRule | null {
  const def = defs.get(constraintName);
  if (!def?.rule) return null;
  return findElementConstraint(def.rule, defs, new Set());
}

function findElementConstraint(
  rule: ConstraintRule,
  defs: Map<string, ConstraintDef>,
  visited: Set<string>
): ConstraintRule | null {
  switch (rule.kind) {
    case 'shaped':
      return rule.element ?? null;
    case 'like':
      return rule.element;
    case 'and':
      // 在 and 中查找 shaped 的 element
      for (const child of rule.children) {
        const elem = findElementConstraint(child, defs, visited);
        if (elem) return elem;
      }
      return null;
    case 'or':
      // 在 or 中，所有分支应该有相同的 element（或者取并集）
      for (const child of rule.children) {
        const elem = findElementConstraint(child, defs, visited);
        if (elem) return elem;
      }
      return null;
    case 'ref': {
      if (visited.has(rule.name)) return null;
      visited.add(rule.name);
      const def = defs.get(rule.name);
      if (!def?.rule) return null;
      return findElementConstraint(def.rule, defs, visited);
    }
    default:
      return null;
  }
}
