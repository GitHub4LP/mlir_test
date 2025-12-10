# MLIR Blueprint Editor

可视化 MLIR 操作编排系统，类似 UE5 蓝图。

## 功能

- 拖拽方言操作节点构建计算图
- 自动类型推导与约束检查
- 执行引脚控制流（支持 scf.for、scf.while 等控制流操作）
- 生成可执行 MLIR IR

## 快速开始

```bash
# 安装依赖
uv sync
cd frontend && npm install

# 启动后端 (端口 8000)
uv run uvicorn backend.main:app --reload --port 8000

# 启动前端 (端口 5173)
cd frontend && npm run dev
```

访问 http://localhost:5173

## 快捷键

- `Ctrl+C/V` - 复制/粘贴节点
- `Delete` - 删除选中项
- 双击连线 - 删除连线
- 框选 - 多选节点

## 项目结构

```
├── backend/          # FastAPI 后端
│   ├── api/          # API 路由 (dialects, projects, execution)
│   └── mlir_utils/   # MLIR 工具 (CLI, JSON 生成)
├── frontend/         # React + TypeScript 前端
│   └── src/
│       ├── components/   # React 组件
│       ├── services/     # 业务逻辑
│       └── stores/       # Zustand 状态
└── mlir_data/        # 方言 JSON 数据
```

## CLI

```bash
python -m backend.mlir_utils.cli --list          # 列出方言
python -m backend.mlir_utils.cli arith -o out    # 生成单个方言
python -m backend.mlir_utils.cli all -o out      # 生成所有方言
```

## 技术栈

- **前端**: React 18, TypeScript, React Flow, Zustand, Tailwind CSS, Vite
- **后端**: Python, FastAPI, Pydantic
- **数据**: llvm-tblgen --dump-json 解析 TableGen 定义

## 许可证

MIT
