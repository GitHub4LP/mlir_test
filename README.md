# MLIR Blueprint Editor

可视化 MLIR 操作编排系统，类似 UE5 蓝图。

## 功能

- 拖拽方言操作节点构建计算图
- 自动类型推导与约束检查
- 执行引脚控制流
- 函数管理：入口/返回/调用节点
- JSON 格式项目持久化
- 生成可执行 MLIR IR

## 使用方式

### 方式一：独立 Web 部署

适合团队共享或远程访问。

```bash
# 后端
uv sync
uv run uvicorn backend.main:app --reload --port 8000

# 前端（新终端）
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173

### 方式二：VS Code / Code Server 扩展

适合本地开发，集成到现有工作流。

```bash
# 构建
cd frontend && npm install && npm run build:vscode
cd ../vscode-extension && npm install && npm run compile

# 启动后端（扩展需要后端服务）
uv run uvicorn backend.main:app --reload --port 8000
```

在 VS Code 中按 F5 启动扩展开发主机，或打包 `.vsix` 安装。

扩展功能：
- 侧边栏操作浏览器（支持拖放）
- 双击 `.mlir.json` 文件打开节点编辑器
- 工作区自动检测项目

## 项目结构

```
├── backend/              # FastAPI 后端
├── frontend/             # React + TypeScript 前端
├── vscode-extension/     # VS Code 扩展
├── mlir_data/            # 方言 JSON（自动生成）
└── projects/             # 用户项目
```

## 许可证

MIT
