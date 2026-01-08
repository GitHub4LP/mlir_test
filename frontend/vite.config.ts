import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  // 使用相对路径，支持任意子路径部署
  // 构建后的资源引用为 ./assets/xxx.js 而不是 /assets/xxx.js
  base: './',
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
    vue(),  // Vue 支持（用于 Vue Flow 适配器）
    tailwindcss(),
  ],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: false,
    allowedHosts: true,  // 允许所有域名
    proxy: {
      // 代理 API 请求到后端
      // 支持动态端口：从环境变量 BACKEND_PORT 读取，默认 8000
      // 支持任意子路径：/api, /mlir-editor/api, /tools/blueprint/api
      '/api': {
        target: `http://localhost:${process.env.BACKEND_PORT || '8000'}`,
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
  },
})
