# MLIR 方言文档生成器

为构建 MLIR 可视化节点编排系统（类似 UE5 蓝图）提供基础数据。

## 项目目标

构建可视化的 MLIR 操作编排系统：
- 拖拽不同方言的操作节点
- 通过连线组合操作流程
- 自动处理类型转换
- 生成可执行的 MLIR IR

## 方言范围

通过分析 `Passes.td` 中的 `dependentDialects` 自动发现可 lower 到 LLVM 的方言，排除纯后端方言（`llvm`, `rocdl`, `emitc` 等）。

运行 `python -m mlir_utils.cli --list` 查看完整列表。

## 安装

```bash
uv sync
```

## 使用

```bash
# 列出可用方言
python -m mlir_utils.cli --list

# 生成单个方言 JSON
python -m mlir_utils.cli arith -o output

# 生成所有可 lower 方言的 JSON
python -m mlir_utils.cli all -o output

# 查看 lowering 关系图
python -m mlir_utils.lowering
```

## 项目结构

```
├── mlir_utils/              # 主模块
│   ├── cli.py               # 命令行接口
│   ├── generator.py         # JSON 生成器
│   ├── tblgen.py            # TableGen 工具封装
│   └── lowering.py          # 方言 lowering 关系发现
└── pyproject.toml
```

## 技术实现

### 数据源

使用 `llvm-tblgen --dump-json` 解析 TableGen 定义文件，直接输出原始 JSON 数据。

JSON 包含完整的操作信息：
- 操作名称、描述、assemblyFormat
- arguments（operands + attributes）
- results
- traits 和 interfaces
- hasFolder、hasCanonicalizer 等元信息

### Lowering 关系发现

通过解析所有 `Passes.td` 文件的 `dependentDialects` 字段，自动发现方言间的 lowering 关系：
- 扫描 `mlir/Conversion/Passes.td` 和各方言的 `Transforms/Passes.td`
- 从 pass 的 `dependentDialects` 提取目标方言（最准确的官方数据源）
- 反向 BFS 找出所有能到达 LLVM 的方言

## 其他有用的 MLIR 工具

### mlir-opt --view-op-graph

生成 Graphviz DOT 格式的操作图，可用于可视化：

```bash
mlir-opt test.mlir --view-op-graph > graph.dot
dot -Tpng graph.dot -o graph.png
```

选项：
- `--print-data-flow-edges` - 数据流边（默认 true）
- `--print-control-flow-edges` - 控制流边
- `--print-attrs` - 操作属性
- `--print-result-types` - 结果类型

### mlir-opt --print-op-stats

统计 IR 中各操作的数量。

### mlir-opt --mlir-print-op-generic

输出 generic 格式（显示所有属性），便于解析。

### mlir-translate --mlir-to-llvmir

将 LLVM dialect 转换为 LLVM IR。

## 许可证

MIT
