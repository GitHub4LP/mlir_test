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
  | { kind: 'like'; element: ConstraintRule; containers?: string[] };  // 标量或其容器

// 默认 like 容器（ValueSemantics 容器）
const DEFAULT_LIKE_CONTAINERS = ['tensor', 'vector'];

// 所有容器类型
const ALL_CONTAINERS = ['tensor', 'vector', 'memref', 'complex', 'unranked_tensor', 'unranked_memref'];

export interface ConstraintDef {
  name: string;
  summary: string;
  rule: ConstraintRule | null;
  dialect?: string | null;  // 来源方言，null/undefined 表示内置约束
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
      // shaped 规则不展开为标量类型
      // 如果有 element，可以展开 element 的标量部分
      if (rule.element) {
        return expandRule(rule.element, defs, buildableTypes, visited);
      }
      return [];
    
    case 'like':
      // like 展开为元素的标量类型
      return expandRule(rule.element, defs, buildableTypes, visited);
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
      // like 允许容器（根据 containers 字段）
      return true;
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
    case 'type':
    case 'oneOf':
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
        return [...ALL_CONTAINERS];
      }
      return [rule.container];
    case 'like':
      // like 使用 containers 字段，默认为 ValueSemantics 容器
      return rule.containers ?? DEFAULT_LIKE_CONTAINERS;
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
    case 'type':
    case 'oneOf':
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

// ============ 元素约束展开 ============

/**
 * 元素约束展开结果
 */
export interface ElementConstraintExpansion {
  scalarTypes: string[];      // 允许的标量类型
  allowedContainers: string[]; // 允许的容器类型（从 shaped 规则提取）
}

/**
 * 展开元素约束，分离标量类型和允许的容器
 * 
 * 用于类型选择器：根据当前容器的元素约束，决定内层可选的类型和包装选项
 * 
 * @param constraintName 元素约束名（如 VectorElementType）
 * @param defs 所有约束定义
 * @param buildableTypes 所有可构建类型
 * @returns 标量类型列表和允许的容器列表
 */
export function expandElementConstraint(
  constraintName: string,
  defs: Map<string, ConstraintDef>,
  buildableTypes: string[]
): ElementConstraintExpansion {
  const scalarTypes: string[] = [];
  const allowedContainers: string[] = [];
  
  const def = defs.get(constraintName);
  if (!def?.rule) {
    return { scalarTypes: [], allowedContainers: [] };
  }
  
  walkElementRule(def.rule, defs, buildableTypes, scalarTypes, allowedContainers, new Set());
  
  return {
    scalarTypes: [...new Set(scalarTypes)],
    allowedContainers: [...new Set(allowedContainers)],
  };
}

/**
 * 递归遍历规则，收集标量类型和容器
 */
function walkElementRule(
  rule: ConstraintRule,
  defs: Map<string, ConstraintDef>,
  buildableTypes: string[],
  scalarTypes: string[],
  allowedContainers: string[],
  visited: Set<string>
): void {
  switch (rule.kind) {
    case 'type':
      scalarTypes.push(rule.name);
      break;
    
    case 'oneOf':
      scalarTypes.push(...rule.types);
      break;
    
    case 'or':
      for (const child of rule.children) {
        walkElementRule(child, defs, buildableTypes, scalarTypes, allowedContainers, visited);
      }
      break;
    
    case 'and': {
      // 交集：分别收集，然后取交集
      const childScalars: string[][] = [];
      const childContainers: string[][] = [];
      for (const child of rule.children) {
        const s: string[] = [];
        const c: string[] = [];
        walkElementRule(child, defs, buildableTypes, s, c, new Set(visited));
        childScalars.push(s);
        childContainers.push(c);
      }
      // 取交集
      if (childScalars.length > 0) {
        const intersection = childScalars.reduce((acc, curr) => 
          acc.filter(t => curr.includes(t))
        );
        scalarTypes.push(...intersection);
      }
      if (childContainers.length > 0) {
        const intersection = childContainers.reduce((acc, curr) => 
          acc.filter(c => curr.includes(c))
        );
        allowedContainers.push(...intersection);
      }
      break;
    }
    
    case 'ref': {
      if (visited.has(rule.name)) return;
      visited.add(rule.name);
      const refDef = defs.get(rule.name);
      if (refDef?.rule) {
        walkElementRule(refDef.rule, defs, buildableTypes, scalarTypes, allowedContainers, visited);
      }
      break;
    }
    
    case 'shaped':
      // shaped 规则表示允许该容器作为元素
      if (rule.container === 'shaped') {
        // shaped 表示任意容器
        allowedContainers.push(...ALL_CONTAINERS);
      } else {
        allowedContainers.push(rule.container);
      }
      // 如果有 element，递归处理
      if (rule.element) {
        walkElementRule(rule.element, defs, buildableTypes, scalarTypes, allowedContainers, visited);
      }
      break;
    
    case 'like':
      // like 允许标量或其容器，展开标量部分
      walkElementRule(rule.element, defs, buildableTypes, scalarTypes, allowedContainers, visited);
      // like 使用 containers 字段
      allowedContainers.push(...(rule.containers ?? DEFAULT_LIKE_CONTAINERS));
      break;
  }
}


// ============ 三维度约束描述符 ============

/**
 * 约束的三维度描述符
 * 
 * 用于统一处理标量和容器约束的子集匹配。
 * 标量视为容器的特例：container = null, shape = []
 */
export interface ConstraintDescriptor {
  /** 允许的容器类型，null 表示标量（无包装） */
  containers: (string | null)[];
  /** 允许的元素类型（标量类型列表） */
  elements: string[];
  /** 形状约束：'any' | 'ranked' | 'unranked' */
  shapeConstraint: 'any' | 'ranked' | 'unranked';
}

/**
 * 从约束规则提取三维度描述符
 * 
 * @param rule 约束规则
 * @param defs 所有约束定义
 * @param buildableTypes 所有可构建类型
 * @returns 三维度描述符
 */
export function extractConstraintDescriptor(
  rule: ConstraintRule | null,
  defs: Map<string, ConstraintDef>,
  buildableTypes: string[]
): ConstraintDescriptor {
  if (!rule) {
    return { containers: [], elements: [], shapeConstraint: 'any' };
  }
  
  const containers: (string | null)[] = [];
  const elements: string[] = [];
  let shapeConstraint: 'any' | 'ranked' | 'unranked' = 'any';
  
  extractFromRule(rule, defs, buildableTypes, containers, elements, new Set());
  
  // 提取形状约束
  shapeConstraint = extractShapeConstraint(rule, defs, new Set());
  
  return {
    containers: [...new Set(containers)],
    elements: [...new Set(elements)],
    shapeConstraint,
  };
}

/**
 * 递归提取容器和元素信息
 */
function extractFromRule(
  rule: ConstraintRule,
  defs: Map<string, ConstraintDef>,
  buildableTypes: string[],
  containers: (string | null)[],
  elements: string[],
  visited: Set<string>
): void {
  switch (rule.kind) {
    case 'type':
      // 具体标量类型：容器 = null，元素 = 自身
      containers.push(null);
      elements.push(rule.name);
      break;
    
    case 'oneOf':
      // 标量类型枚举：容器 = null，元素 = 列表
      containers.push(null);
      elements.push(...rule.types);
      break;
    
    case 'or':
      // 并集：合并所有子规则
      for (const child of rule.children) {
        extractFromRule(child, defs, buildableTypes, containers, elements, visited);
      }
      break;
    
    case 'and':
      // 交集：取公共部分（简化处理，取第一个子规则）
      if (rule.children.length > 0) {
        extractFromRule(rule.children[0], defs, buildableTypes, containers, elements, visited);
      }
      break;
    
    case 'ref': {
      if (visited.has(rule.name)) return;
      visited.add(rule.name);
      const def = defs.get(rule.name);
      if (def?.rule) {
        extractFromRule(def.rule, defs, buildableTypes, containers, elements, visited);
      }
      break;
    }
    
    case 'shaped':
      // 容器约束：容器 = 指定容器，元素 = element 的展开
      if (rule.container === 'shaped') {
        containers.push('tensor', 'vector', 'memref');
      } else {
        containers.push(rule.container);
      }
      if (rule.element) {
        // 递归提取元素约束的标量类型
        const elemTypes = expandRule(rule.element, defs, buildableTypes);
        elements.push(...elemTypes);
      } else {
        // 无元素约束，允许所有标量
        elements.push(...buildableTypes);
      }
      break;
    
    case 'like': {
      // Like 约束：容器 = null + containers，元素 = element 的展开
      containers.push(null);  // 允许标量
      containers.push(...(rule.containers ?? DEFAULT_LIKE_CONTAINERS));
      const likeElements = expandRule(rule.element, defs, buildableTypes);
      elements.push(...likeElements);
      break;
    }
  }
}

/**
 * 提取形状约束
 */
function extractShapeConstraint(
  rule: ConstraintRule,
  defs: Map<string, ConstraintDef>,
  visited: Set<string>
): 'any' | 'ranked' | 'unranked' {
  switch (rule.kind) {
    case 'shaped':
      if (rule.ranked === true) return 'ranked';
      if (rule.ranked === false) return 'unranked';
      return 'any';
    
    case 'ref': {
      if (visited.has(rule.name)) return 'any';
      visited.add(rule.name);
      const def = defs.get(rule.name);
      if (def?.rule) {
        return extractShapeConstraint(def.rule, defs, visited);
      }
      return 'any';
    }
    
    case 'or':
    case 'and':
      // 简化：取第一个子规则的形状约束
      if (rule.children.length > 0) {
        return extractShapeConstraint(rule.children[0], defs, visited);
      }
      return 'any';
    
    default:
      return 'any';
  }
}

/**
 * 检查约束 A 是否是约束 B 的子集（三维度匹配）
 * 
 * A ⊆ B 当且仅当：
 * 1. A.containers ⊆ B.containers
 * 2. A.elements ⊆ B.elements
 * 3. A.shapeConstraint ⊆ B.shapeConstraint
 */
export function isConstraintSubset(
  sub: ConstraintDescriptor,
  sup: ConstraintDescriptor
): boolean {
  // 1. 容器子集检查
  const supContainers = new Set(sup.containers);
  for (const c of sub.containers) {
    if (!supContainers.has(c)) return false;
  }
  
  // 2. 元素子集检查
  const supElements = new Set(sup.elements);
  for (const e of sub.elements) {
    if (!supElements.has(e)) return false;
  }
  
  // 3. 形状约束子集检查
  if (!isShapeSubset(sub.shapeConstraint, sup.shapeConstraint)) {
    return false;
  }
  
  return true;
}

/**
 * 检查形状约束是否是子集
 */
function isShapeSubset(
  sub: 'any' | 'ranked' | 'unranked',
  sup: 'any' | 'ranked' | 'unranked'
): boolean {
  if (sup === 'any') return true;
  return sub === sup;
}

/**
 * 从约束名获取三元素描述符
 * 
 * @param constraintName 约束名（如 'AnyTensor', 'SignlessIntegerLike', 'I32'）
 * @param defs 所有约束定义
 * @param buildableTypes 所有可构建类型
 * @returns 三元素描述符
 */
export function getConstraintDescriptor(
  constraintName: string,
  defs: Map<string, ConstraintDef>,
  buildableTypes: string[]
): ConstraintDescriptor {
  const def = defs.get(constraintName);
  
  if (!def?.rule) {
    // 没有规则，检查是否是 BuildableType（具体标量类型）
    if (buildableTypes.includes(constraintName)) {
      return {
        containers: [null],  // 标量：容器 = null
        elements: [constraintName],
        shapeConstraint: 'any',
      };
    }
    // 未知约束
    return { containers: [], elements: [], shapeConstraint: 'any' };
  }
  
  return extractConstraintDescriptor(def.rule, defs, buildableTypes);
}

/**
 * 计算两个三元素描述符的交集
 * 
 * @returns 交集描述符，如果不兼容返回 null
 */
export function intersectDescriptors(
  d1: ConstraintDescriptor,
  d2: ConstraintDescriptor
): ConstraintDescriptor | null {
  // 1. 容器交集
  const containers = d1.containers.filter(c => d2.containers.includes(c));
  if (containers.length === 0) return null;
  
  // 2. 元素交集
  const elemSet2 = new Set(d2.elements);
  const elements = d1.elements.filter(e => elemSet2.has(e));
  if (elements.length === 0) return null;
  
  // 3. 形状约束交集
  const shapeConstraint = intersectShapeConstraints(d1.shapeConstraint, d2.shapeConstraint);
  if (shapeConstraint === null) return null;
  
  return { containers, elements, shapeConstraint };
}

/**
 * 计算形状约束的交集
 */
function intersectShapeConstraints(
  s1: 'any' | 'ranked' | 'unranked',
  s2: 'any' | 'ranked' | 'unranked'
): 'any' | 'ranked' | 'unranked' | null {
  if (s1 === 'any') return s2;
  if (s2 === 'any') return s1;
  if (s1 === s2) return s1;
  return null;  // ranked 和 unranked 不兼容
}
