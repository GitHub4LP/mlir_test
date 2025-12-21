/**
 * 参数/返回值管理服务
 * 
 * 提供参数和返回值的命名、验证等公用逻辑，供所有渲染器复用。
 */

/**
 * 生成唯一名称
 * 
 * @param existingNames 已存在的名称列表
 * @param prefix 名称前缀（如 'arg', 'ret'）
 * @returns 唯一的新名称
 */
export function generateUniqueName(existingNames: string[], prefix: string): string {
  let index = 0;
  let newName = `${prefix}${index}`;
  while (existingNames.includes(newName)) {
    index++;
    newName = `${prefix}${index}`;
  }
  return newName;
}

/**
 * 验证名称是否有效
 * 
 * @param name 要验证的名称
 * @param existingNames 已存在的名称列表
 * @param currentName 当前名称（重命名时排除自己）
 * @returns 验证结果
 */
export function validateName(
  name: string,
  existingNames: string[],
  currentName?: string
): { valid: boolean; error?: string } {
  // 空名称
  if (!name || name.trim() === '') {
    return { valid: false, error: '名称不能为空' };
  }

  // 名称格式（只允许字母、数字、下划线，不能以数字开头）
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
    return { valid: false, error: '名称只能包含字母、数字和下划线，且不能以数字开头' };
  }

  // 重复检查（排除当前名称）
  const otherNames = currentName 
    ? existingNames.filter(n => n !== currentName)
    : existingNames;
  
  if (otherNames.includes(name)) {
    return { valid: false, error: '名称已存在' };
  }

  return { valid: true };
}

/**
 * 生成参数的唯一名称
 */
export function generateParameterName(existingNames: string[]): string {
  return generateUniqueName(existingNames, 'arg');
}

/**
 * 生成返回值的唯一名称
 */
export function generateReturnTypeName(existingNames: string[]): string {
  return generateUniqueName(existingNames, 'ret');
}
