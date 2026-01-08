/**
 * VS Code 模式构建配置
 * 
 * 构建用于 VS Code Webview 的前端资源：
 * - editor.js - 节点编辑器入口
 * - properties.js - 属性面板入口
 * 
 * 支持动态懒加载，与 Web 模式一致
 * 
 * 注意：VS Code Webview 中动态 import 的路径解析依赖于 <base> 标签，
 * 因此 base 设置为 './'，由 webview.ts 中的 <base href> 提供正确的基础 URL。
 */

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import vue from '@vitejs/plugin-vue';
import tailwindcss from '@tailwindcss/vite';
import { resolve } from 'path';

export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
    vue(),
    tailwindcss(),
  ],
  define: {
    'import.meta.env.VITE_PLATFORM': JSON.stringify('vscode'),
  },
  // 设置 base 为 './'，让动态 import 使用相对路径
  // 在 VS Code Webview 中，<base href> 会提供正确的基础 URL
  base: './',
  build: {
    outDir: '../vscode-extension/media',
    emptyOutDir: false,
    sourcemap: true,
    rollupOptions: {
      input: {
        editor: resolve(__dirname, 'src/app/EditorApp.tsx'),
        properties: resolve(__dirname, 'src/app/PropertiesApp.tsx'),
      },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name].[ext]',
      },
    },
    cssCodeSplit: false,
  },
  server: {
    hmr: false,
  },
});
