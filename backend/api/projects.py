"""
项目管理 API 路由

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
    type: str


class TypeDef(BaseModel):
    """返回类型"""
    name: str
    type: str


class PortConfig(BaseModel):
    """端口配置"""
    id: str
    name: str
    kind: str  # 'input' | 'output'
    typeConstraint: str
    concreteType: Optional[str] = None
    color: str


class ArgumentDef(BaseModel):
    """操作参数"""
    name: str
    kind: str  # 'operand' | 'attribute'
    typeConstraint: str
    isOptional: bool


class ResultDef(BaseModel):
    """操作结果"""
    name: str
    typeConstraint: str


class OperationDef(BaseModel):
    """MLIR 操作"""
    dialect: str
    opName: str
    fullName: str
    summary: str
    description: str
    arguments: list[ArgumentDef]
    results: list[ResultDef]
    traits: list[str]
    assemblyFormat: str


class BlueprintNodeData(BaseModel):
    """Blueprint node data (stored format).
    
    Only stores fullName reference, not the full operation definition.
    The operation definition is loaded from dialect data at runtime.
    """
    fullName: str
    attributes: dict[str, Any]
    inputTypes: dict[str, str]
    outputTypes: dict[str, str]


class FunctionEntryData(BaseModel):
    """函数入口节点"""
    functionId: str
    outputs: list[PortConfig]


class FunctionReturnData(BaseModel):
    """函数返回节点"""
    functionId: str
    inputs: list[PortConfig]


class FunctionCallData(BaseModel):
    """函数调用节点"""
    functionId: str
    functionName: str
    inputs: list[PortConfig]
    outputs: list[PortConfig]


class GraphNode(BaseModel):
    """图节点"""
    id: str
    type: str  # 'operation' | 'function-entry' | 'function-return' | 'function-call'
    position: dict[str, float]
    data: dict[str, Any]


class GraphEdge(BaseModel):
    """图边"""
    id: str
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class GraphState(BaseModel):
    """Graph state containing nodes and edges."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class FunctionDef(BaseModel):
    """函数定义"""
    id: str
    name: str
    parameters: list[ParameterDef]
    returnTypes: list[TypeDef]
    graph: GraphState
    isMain: bool


class Project(BaseModel):
    """完整项目"""
    name: str
    path: str
    mainFunction: FunctionDef
    customFunctions: list[FunctionDef]
    dialects: list[str]


class ProjectCreate(BaseModel):
    """创建项目请求"""
    name: str
    path: str


class ProjectResponse(BaseModel):
    """项目响应"""
    name: str
    path: str
    dialects: list[str] = []


class SaveProjectRequest(BaseModel):
    """保存项目请求"""
    project: Project


class SaveProjectResponse(BaseModel):
    """保存响应"""
    status: str
    path: str


class LoadProjectResponse(BaseModel):
    """加载响应"""
    project: Project


class ProjectPathRequest(BaseModel):
    """包含项目路径的请求"""
    projectPath: str


# --- Helper Functions ---

def get_project_file_path(project_path: str) -> Path:
    """获取 project.json 路径"""
    return Path(project_path) / "project.json"


def get_functions_dir(project_path: str) -> Path:
    """获取函数目录路径"""
    return Path(project_path) / "functions"


def ensure_project_directories(project_path: str) -> None:
    """确保项目目录存在"""
    path = Path(project_path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "functions").mkdir(exist_ok=True)
    (path / "generated").mkdir(exist_ok=True)


def save_function_to_file(functions_dir: Path, func: FunctionDef) -> None:
    """Save a function definition to a JSON file."""
    file_path = functions_dir / f"{func.id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(func.model_dump(), f, indent=2, ensure_ascii=False)


def load_function_from_file(file_path: Path) -> FunctionDef:
    """Load a function definition from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FunctionDef(**data)


def extract_dialects_from_project(project: Project) -> list[str]:
    """
    从项目的所有节点中提取使用的方言列表。
    
    遍历所有函数的图节点，从 operation 类型节点的 fullName 中提取方言名。
    """
    dialects: set[str] = set()
    
    def extract_from_graph(graph: GraphState) -> None:
        for node in graph.nodes:
            if node.type == "operation":
                full_name = node.data.get("fullName", "")
                if "." in full_name:
                    dialect = full_name.split(".")[0]
                    dialects.add(dialect)
    
    # 提取主函数的方言
    extract_from_graph(project.mainFunction.graph)
    
    # 提取自定义函数的方言
    for func in project.customFunctions:
        extract_from_graph(func.graph)
    
    return sorted(dialects)


# --- API Routes ---

@router.post("/", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """
    Create a new MLIR Blueprint project.
    
    Creates a project with a default main function and stores project metadata.
    Requirements: 1.1
    """
    try:
        ensure_project_directories(project.path)
        
        # Create default main function
        main_function = FunctionDef(
            id="main",
            name="main",
            parameters=[],
            returnTypes=[],
            graph=GraphState(nodes=[], edges=[]),
            isMain=True
        )
        
        # Create project
        new_project = Project(
            name=project.name,
            path=project.path,
            mainFunction=main_function,
            customFunctions=[],
            dialects=[]
        )
        
        # Save project metadata
        project_file = get_project_file_path(project.path)
        project_data = {
            "name": new_project.name,
            "path": new_project.path,
            "dialects": new_project.dialects,
            "mainFunctionId": main_function.id,
            "customFunctionIds": []
        }
        with open(project_file, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        
        # Save main function
        functions_dir = get_functions_dir(project.path)
        save_function_to_file(functions_dir, main_function)
        
        return ProjectResponse(
            name=new_project.name,
            path=new_project.path,
            dialects=new_project.dialects
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")


@router.post("/load", response_model=LoadProjectResponse)
async def load_project(request: ProjectPathRequest):
    """
    Load an existing project from disk.
    
    Loads all functions and their node graphs from the project directory.
    Requirements: 1.3
    
    注意：使用 POST + 请求体传递路径，避免 URL 编码问题
    """
    project_path = request.projectPath
    project_file = get_project_file_path(project_path)
    
    if not project_file.exists():
        raise HTTPException(status_code=404, detail=f"Project not found at {project_path}")
    
    try:
        # Load project metadata
        with open(project_file, "r", encoding="utf-8") as f:
            project_data = json.load(f)
        
        functions_dir = get_functions_dir(project_path)
        
        # Load main function
        main_function_id = project_data.get("mainFunctionId", "main")
        main_function_file = functions_dir / f"{main_function_id}.json"
        
        if not main_function_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Main function file not found: {main_function_file}"
            )
        
        main_function = load_function_from_file(main_function_file)
        
        # Load custom functions
        custom_functions: list[FunctionDef] = []
        custom_function_ids = project_data.get("customFunctionIds", [])
        
        for func_id in custom_function_ids:
            func_file = functions_dir / f"{func_id}.json"
            if func_file.exists():
                custom_functions.append(load_function_from_file(func_file))
        
        # Construct project
        project = Project(
            name=project_data.get("name", "Untitled"),
            path=project_path,
            mainFunction=main_function,
            customFunctions=custom_functions,
            dialects=project_data.get("dialects", [])
        )
        
        return LoadProjectResponse(project=project)
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid project file format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load project: {str(e)}")


@router.post("/save", response_model=SaveProjectResponse)
async def save_project(request: SaveProjectRequest):
    """
    Save project to disk.
    
    Saves all project files to the specified directory.
    Requirements: 1.2
    
    注意：使用 POST + 请求体传递路径，避免 URL 编码问题
    路径从 project.path 获取
    """
    try:
        project = request.project
        project_path = project.path
        
        # Ensure directories exist
        ensure_project_directories(project_path)
        
        functions_dir = get_functions_dir(project_path)
        
        # Save main function
        save_function_to_file(functions_dir, project.mainFunction)
        
        # Save custom functions
        custom_function_ids: list[str] = []
        for func in project.customFunctions:
            save_function_to_file(functions_dir, func)
            custom_function_ids.append(func.id)
        
        # 从节点自动计算使用的方言列表（不再手动指定）
        dialects = extract_dialects_from_project(project)
        
        # Save project metadata
        project_file = get_project_file_path(project_path)
        project_data = {
            "name": project.name,
            "path": project_path,
            "dialects": dialects,
            "mainFunctionId": project.mainFunction.id,
            "customFunctionIds": custom_function_ids
        }
        with open(project_file, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        
        return SaveProjectResponse(status="saved", path=project_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save project: {str(e)}")


@router.post("/delete")
async def delete_project(request: ProjectPathRequest):
    """
    Delete a project from disk.
    
    Removes all project files from the specified directory.
    
    注意：使用 POST + 请求体传递路径，避免 URL 编码问题
    """
    import shutil
    
    project_path = request.projectPath
    project_dir = Path(project_path)
    
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project not found at {project_path}")
    
    try:
        shutil.rmtree(project_dir)
        return {"status": "deleted", "path": project_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")
