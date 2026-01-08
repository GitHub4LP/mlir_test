"""
函数管理 API 路由

基于文件的函数存储：每个函数一个 .mlir.json 文件
- main.mlir.json: main 函数 + 项目元数据
- {name}.mlir.json: 自定义函数
"""

import json
import re
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
    project: Optional[dict[str, str]] = None  # 仅 main.mlir.json 有
    function: StoredFunctionDef


# --- Request/Response Models ---

class LoadFunctionRequest(BaseModel):
    """加载函数请求"""
    projectPath: str
    functionName: str


class LoadFunctionResponse(BaseModel):
    """加载函数响应"""
    function: StoredFunctionDef


class SaveFunctionRequest(BaseModel):
    """保存函数请求"""
    projectPath: str
    function: StoredFunctionDef
    projectName: Optional[str] = None  # 仅保存 main 时需要


class SaveFunctionResponse(BaseModel):
    """保存函数响应"""
    status: str


class CreateFunctionRequest(BaseModel):
    """创建函数请求"""
    projectPath: str
    functionName: str


class CreateFunctionResponse(BaseModel):
    """创建函数响应"""
    function: StoredFunctionDef


class RenameFunctionRequest(BaseModel):
    """重命名函数请求"""
    projectPath: str
    oldName: str
    newName: str


class RenameFunctionResponse(BaseModel):
    """重命名函数响应"""
    status: str
    updatedFiles: list[str]  # 被更新的文件列表


class DeleteFunctionRequest(BaseModel):
    """删除函数请求"""
    projectPath: str
    functionName: str
    force: bool = False  # 强制删除（即使有引用）


class DeleteFunctionResponse(BaseModel):
    """删除函数响应"""
    status: str
    references: list[str] = []  # 引用该函数的其他函数


class ListFunctionsRequest(BaseModel):
    """列出函数请求"""
    projectPath: str


class ListFunctionsResponse(BaseModel):
    """列出函数响应"""
    functionNames: list[str]


# --- Helper Functions ---

def is_valid_function_name(name: str) -> bool:
    """检查函数名是否有效（字母/下划线开头，字母数字下划线组成）"""
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def get_function_file_path(project_path: str, function_name: str) -> Path:
    """获取函数文件路径"""
    return Path(project_path) / f"{function_name}.mlir.json"


def load_function_file(file_path: Path) -> FunctionFile:
    """从文件加载函数"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FunctionFile(**data)


def save_function_file(file_path: Path, func_file: FunctionFile) -> None:
    """保存函数到文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(func_file.model_dump(exclude_none=True), f, indent=2, ensure_ascii=False)


def create_empty_function(name: str) -> StoredFunctionDef:
    """创建空函数定义"""
    return StoredFunctionDef(
        name=name,
        parameters=[],
        returnTypes=[],
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


def find_function_references(project_path: str, function_name: str) -> list[str]:
    """查找引用指定函数的所有函数"""
    references = []
    project_dir = Path(project_path)
    
    for file_path in project_dir.glob("*.mlir.json"):
        if file_path.stem == function_name:
            continue  # 跳过自己
        
        try:
            func_file = load_function_file(file_path)
            for node in func_file.function.graph.nodes:
                if node.type == "function-call":
                    if node.data.get("functionName") == function_name:
                        references.append(file_path.stem)
                        break
        except Exception:
            continue
    
    return references


def update_function_references(
    project_path: str, 
    old_name: str, 
    new_name: str
) -> list[str]:
    """更新所有引用指定函数的 Call 节点"""
    updated_files = []
    project_dir = Path(project_path)
    
    for file_path in project_dir.glob("*.mlir.json"):
        if file_path.stem == new_name:
            continue  # 跳过目标函数自己
        
        try:
            func_file = load_function_file(file_path)
            modified = False
            
            for node in func_file.function.graph.nodes:
                if node.type == "function-call":
                    if node.data.get("functionName") == old_name:
                        node.data["functionName"] = new_name
                        modified = True
            
            if modified:
                save_function_file(file_path, func_file)
                updated_files.append(file_path.stem)
        except Exception:
            continue
    
    return updated_files


# --- API Routes ---

@router.post("/load", response_model=LoadFunctionResponse)
async def load_function(request: LoadFunctionRequest):
    """加载单个函数"""
    file_path = get_function_file_path(request.projectPath, request.functionName)
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Function '{request.functionName}' not found"
        )
    
    try:
        func_file = load_function_file(file_path)
        # 确保函数名与文件名一致
        func_file.function.name = request.functionName
        return LoadFunctionResponse(function=func_file.function)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid function file format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load function: {str(e)}"
        )


@router.post("/save", response_model=SaveFunctionResponse)
async def save_function(request: SaveFunctionRequest):
    """保存单个函数"""
    function_name = request.function.name
    
    if not is_valid_function_name(function_name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid function name: '{function_name}'"
        )
    
    file_path = get_function_file_path(request.projectPath, function_name)
    
    # 确保目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 构建文件内容
        func_file = FunctionFile(function=request.function)
        
        # 如果是 main 函数，添加项目元数据
        if function_name == "main" and request.projectName:
            func_file.project = {"name": request.projectName}
        elif function_name == "main":
            # 尝试保留现有的项目元数据
            if file_path.exists():
                existing = load_function_file(file_path)
                func_file.project = existing.project
        
        save_function_file(file_path, func_file)
        return SaveFunctionResponse(status="saved")
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to save function: {str(e)}"
        )


@router.post("/create", response_model=CreateFunctionResponse)
async def create_function(request: CreateFunctionRequest):
    """创建新函数"""
    function_name = request.functionName
    
    if not is_valid_function_name(function_name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid function name: '{function_name}'"
        )
    
    if function_name == "main":
        raise HTTPException(
            status_code=400,
            detail="Cannot create function named 'main'"
        )
    
    file_path = get_function_file_path(request.projectPath, function_name)
    
    if file_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Function '{function_name}' already exists"
        )
    
    try:
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建空函数
        func = create_empty_function(function_name)
        func_file = FunctionFile(function=func)
        save_function_file(file_path, func_file)
        
        return CreateFunctionResponse(function=func)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create function: {str(e)}"
        )


@router.post("/rename", response_model=RenameFunctionResponse)
async def rename_function(request: RenameFunctionRequest):
    """重命名函数"""
    old_name = request.oldName
    new_name = request.newName
    
    if not is_valid_function_name(new_name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid function name: '{new_name}'"
        )
    
    if old_name == "main":
        raise HTTPException(
            status_code=400,
            detail="Cannot rename main function"
        )
    
    if new_name == "main":
        raise HTTPException(
            status_code=400,
            detail="Cannot rename to 'main'"
        )
    
    old_path = get_function_file_path(request.projectPath, old_name)
    new_path = get_function_file_path(request.projectPath, new_name)
    
    if not old_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Function '{old_name}' not found"
        )
    
    if new_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Function '{new_name}' already exists"
        )
    
    try:
        # 1. 加载并更新函数名
        func_file = load_function_file(old_path)
        func_file.function.name = new_name
        
        # 2. 保存到新文件
        save_function_file(new_path, func_file)
        
        # 3. 删除旧文件
        old_path.unlink()
        
        # 4. 更新所有引用
        updated_files = update_function_references(
            request.projectPath, old_name, new_name
        )
        
        return RenameFunctionResponse(
            status="renamed",
            updatedFiles=updated_files
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to rename function: {str(e)}"
        )


@router.post("/delete", response_model=DeleteFunctionResponse)
async def delete_function(request: DeleteFunctionRequest):
    """删除函数"""
    function_name = request.functionName
    
    if function_name == "main":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete main function"
        )
    
    file_path = get_function_file_path(request.projectPath, function_name)
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Function '{function_name}' not found"
        )
    
    # 检查引用
    references = find_function_references(request.projectPath, function_name)
    
    if references and not request.force:
        return DeleteFunctionResponse(
            status="has_references",
            references=references
        )
    
    try:
        # 删除文件
        file_path.unlink()
        
        return DeleteFunctionResponse(
            status="deleted",
            references=references
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete function: {str(e)}"
        )


@router.post("/list", response_model=ListFunctionsResponse)
async def list_functions(request: ListFunctionsRequest):
    """列出项目中的所有函数名"""
    project_dir = Path(request.projectPath)
    
    if not project_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Project directory not found: {request.projectPath}"
        )
    
    function_names = []
    for file_path in project_dir.glob("*.mlir.json"):
        function_names.append(file_path.stem)
    
    # 确保 main 在最前面
    if "main" in function_names:
        function_names.remove("main")
        function_names.insert(0, "main")
    
    return ListFunctionsResponse(functionNames=function_names)
