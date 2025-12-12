/**
 * 引脚样式定义
 * 
 * 统一的执行引脚和数据引脚样式
 */

/**
 * 执行引脚样式 - 白色三角形
 */
export const execPinStyle = {
  width: 0,
  height: 0,
  borderStyle: 'solid' as const,
  borderWidth: '5px 0 5px 8px',
  borderColor: 'transparent transparent transparent white',
  backgroundColor: 'transparent',
  borderRadius: 0,
};

/**
 * 数据引脚样式 - 彩色圆形
 */
export function dataPinStyle(color: string) {
  return {
    width: 10,
    height: 10,
    backgroundColor: color,
    border: '2px solid #1a1a2e',
    borderRadius: '50%',
  };
}
