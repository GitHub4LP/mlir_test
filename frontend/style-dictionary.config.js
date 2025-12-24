/**
 * Style Dictionary 配置
 * 
 * 将 Design Tokens (JSON) 转换为：
 * - CSS Variables (供 DOM 组件使用)
 * - TypeScript 常量 (供 Canvas/GPU 渲染器使用)
 */

import StyleDictionary from 'style-dictionary';

// 自定义 transform: 将 "32px" 转换为数字 32
StyleDictionary.registerTransform({
  name: 'size/pxToNumber',
  type: 'value',
  filter: (token) => {
    return typeof token.value === 'string' && token.value.endsWith('px');
  },
  transform: (token) => {
    return parseFloat(token.value);
  },
});

// 自定义 transform: 将 font-weight 字符串转换为数字
StyleDictionary.registerTransform({
  name: 'fontWeight/toNumber',
  type: 'value',
  filter: (token) => {
    return token.path.includes('weight') && typeof token.value === 'string' && /^\d+$/.test(token.value);
  },
  transform: (token) => {
    return parseInt(token.value, 10);
  },
});

// 自定义 format: 生成 TypeScript 常量文件
StyleDictionary.registerFormat({
  name: 'typescript/nested',
  format: ({ dictionary }) => {
    const tokens = {};
    
    dictionary.allTokens.forEach(token => {
      let current = tokens;
      const path = token.path;
      
      for (let i = 0; i < path.length - 1; i++) {
        const key = path[i];
        if (!current[key]) {
          current[key] = {};
        }
        current = current[key];
      }
      
      const lastKey = path[path.length - 1];
      current[lastKey] = token.value;
    });
    
    return `/**
 * Design Tokens - 自动生成，请勿手动修改
 * 
 * 生成命令: npm run build:tokens
 * 源文件: frontend/tokens/
 */

export const tokens = ${JSON.stringify(tokens, null, 2)} as const;

export type Tokens = typeof tokens;
`;
  },
});

// 自定义 format: 生成 CSS Variables
StyleDictionary.registerFormat({
  name: 'css/variables-flat',
  format: ({ dictionary }) => {
    const lines = dictionary.allTokens.map(token => {
      const name = token.path.join('-');
      return `  --${name}: ${token.value};`;
    });
    
    return `/**
 * Design Tokens CSS Variables - 自动生成，请勿手动修改
 * 
 * 生成命令: npm run build:tokens
 * 源文件: frontend/tokens/
 */

:root {
${lines.join('\n')}
}
`;
  },
});

export default {
  source: ['tokens/**/*.json'],
  platforms: {
    css: {
      transformGroup: 'css',
      buildPath: 'src/styles/',
      files: [{
        destination: 'tokens.css',
        format: 'css/variables-flat',
      }],
    },
    ts: {
      transforms: ['attribute/cti', 'name/camel', 'size/pxToNumber', 'fontWeight/toNumber'],
      buildPath: 'src/generated/',
      files: [{
        destination: 'tokens.ts',
        format: 'typescript/nested',
      }],
    },
  },
};
