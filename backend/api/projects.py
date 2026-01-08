"""
项目管理 API 路由

新文件格式：
- main.mlir.json: main 函数 + 项目元数据
- {name}.mlir.json: 自定义函数

设计原则：项目路径放在请求体中，避免 URL 编码问题
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any

router = APIRouter()


# --- Pydantic Models ---

class ParameterDef(BaseModel):
    """函数参数"""
    name: str
    constraint: str


class TypeDef(BaseModel):
    """返回类型"""
    name: str
    constraint: str


class FunctionTrait(BaseModel):
    """函数级别的 Trait"""
    kind: str


class GraphNode(BaseModel):
    """图节点"""
    id: str
    type: str
    position: dict[str, float]
    data: dict[str, Any]


class GraphEdge(BaseModel):
    """图边"""
    source: str
    sourceHandle: str
    target: str
    targetHandle: str
    data: Optional[dict[str, Any]] = None


class GraphState(BaseModel):
    """图状态"""
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class StoredFunctionDef(BaseModel):
    """存储格式的函数定义"""
    name: str
    parameters: list[ParameterDef]
    returnTypes: list[TypeDef]
    traits: list[FunctionTrait] = []
    directDialects: list[str] = []
    graph: GraphState


class FunctionFile(BaseModel):
    """函数文件格式"""
    project: Optional[dict[str, str]] = None
    function: StoredFunctionDef


# --- Request/Response Models ---

class ProjectCreate(BaseModel):
    """创建项目请求"""
    name: str
    path: str


class ProjectResponse(BaseModel):
    """项目响应"""
    name: str
    path: str


class ProjectPathRequest(BaseModel):
    """包含项目路径的请求"""
    projectPath: str


class LoadProjectResponse(BaseModel):
    """加载项目响应"""
    projectName: str
    mainFunction: StoredFunctionDef
    functionNames: list[str]  # 所有函数名（包括 main）


# --- Helper Functions ---

def get_main_file_path(project_path: str) -> Path:
    """获取 main.mlir.json 路径"""
    return Path(project_path) / "main.mlir.json"


def create_default_main_function() -> StoredFunctionDef:
    """创建默认的 main 函数"""
    return StoredFunctionDef(
        name="main",
        parameters=[],
        returnTypes=[{"name": "result", "constraint": "I32"}],
        traits=[],
        directDialects=[],
        graph=GraphState(
            nodes=[
                GraphNode(
                    id="entry",
                    type="function-entry",
                    position={"x": 100, "y": 200},
                    data={"execOut": {"id": "exec-out", "label": ""}}
                ),
                GraphNode(
                    id="return-0",
                    type="function-return",
                    position={"x": 500, "y": 200},
                    data={
                        "branchName": "",
                        "execIn": {"id": "exec-in", "label": ""}
                    }
                )
            ],
            edges=[]
        )
    )


def load_function_file(file_path: Path) -> FunctionFile:
    """从文件加载函数"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FunctionFile(**data)


def save_function_file(file_path: Path, func_file: FunctionFile) -> None:
    """保存函数到文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(func_file.model_dump(exclude_none=True), f, indent=2, ensure_ascii=False)


def list_function_names(project_path: str) -> list[str]:
    """列出项目中的所有函数名"""
    project_dir = Path(project_path)
    function_names = []
    
    for file_path in project_dir.glob("*.mlir.json"):
        # 文件名格式: {name}.mlir.json，需要去掉 .mlir.json
        name = file_path.name.replace(".mlir.json", "")
        function_names.append(name)
    
    # 确保 main 在最前面
    if "main" in function_names:
        function_names.remove("main")
        function_names.insert(0, "main")
    
    return function_names


# --- API Routes ---

@router.post("/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """
    创建新的 MLIR Blueprint 项目
    
    创建 main.mlir.json 文件，包含项目元数据和默认 main 函数
    """
    try:
        project_dir = Path(project.path)
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建 build 目录
        (project_dir / "build").mkdir(exist_ok=True)
        
        # 创建 main.mlir.json
        main_func = create_default_main_function()
        main_file = FunctionFile(
            project={"name": project.name},
            function=main_func
        )
        
        main_path = get_main_file_path(project.path)
        save_function_file(main_path, main_file)
        
        return ProjectResponse(
            name=project.name,
            path=project.path
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create project: {str(e)}"
        )


@router.post("/load", response_model=LoadProjectResponse)
async def load_project(request: ProjectPathRequest):
    """
    加载现有项目
    
    只加载 main.mlir.json 和函数名列表，其他函数按需加载
    """
    project_path = request.projectPath
    main_path = get_main_file_path(project_path)
    
    if not main_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Project not found at {project_path} (missing main.mlir.json)"
        )
    
    try:
        # 加载 main.mlir.json
        main_file = load_function_file(main_path)
        
        # 确保函数名为 main
        main_file.function.name = "main"
        
        # 获取项目名
        project_name = main_file.project.get("name", "Untitled") if main_file.project else "Untitled"
        
        # 列出所有函数名
        function_names = list_function_names(project_path)
        
        return LoadProjectResponse(
            projectName=project_name,
            mainFunction=main_file.function,
            functionNames=function_names
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid project file format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load project: {str(e)}"
        )


@router.post("/delete")
async def delete_project(request: ProjectPathRequest):
    """
    删除项目
    
    删除整个项目目录
    """
    import shutil
    
    project_path = request.projectPath
    project_dir = Path(project_path)
    
    if not project_dir.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Project not found at {project_path}"
        )
    
    try:
        shutil.rmtree(project_dir)
        return {"status": "deleted", "path": project_path}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete project: {str(e)}"
        )
