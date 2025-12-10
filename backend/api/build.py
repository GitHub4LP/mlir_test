"""
Build API Routes

将项目构建为 MLIR 文件和 LLVM IR。

设计原则：
- 项目级构建，支持多函数和函数调用
- 利用 ProjectBuilder 统一处理所有节点类型
- 自动依赖分析和拓扑排序
- 项目路径放在请求体中，避免 URL 编码问题
- 生成 MLIR 和 LLVM IR，可选编译（需系统 clang）
"""

import json
import platform
import asyncio
import subprocess
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.project_builder import build_project_from_dict

router = APIRouter()


# ============== 工具函数 ==============

async def run_command(cmd: list[str], input_data: str | None = None, timeout: float = 30.0) -> tuple[int, str, str]:
    """运行命令并返回 (returncode, stdout, stderr)"""
    try:
        loop = asyncio.get_event_loop()
        
        def _run():
            result = subprocess.run(
                cmd,
                input=input_data.encode('utf-8') if input_data else None,
                capture_output=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout.decode('utf-8', errors='replace'), result.stderr.decode('utf-8', errors='replace')
        
        return await loop.run_in_executor(None, _run)
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError as e:
        return -1, "", f"Command not found: {e}"
    except Exception as e:
        return -1, "", f"Error running command: {e}"


# ============== 请求/响应模型 ==============

class ProjectRequest(BaseModel):
    """包含项目路径的基础请求"""
    projectPath: str


class BuildRequest(BaseModel):
    """构建请求"""
    projectPath: str
    generateLlvm: bool = True
    generateExecutable: bool = False  # 默认不编译，因为需要系统 clang


class ExecuteRequest(BaseModel):
    """执行请求"""
    projectPath: str


class PreviewResponse(BaseModel):
    """预览响应"""
    success: bool
    mlirCode: str
    verified: bool
    error: str | None = None


class ExecuteResponse(BaseModel):
    """执行响应"""
    success: bool
    mlirCode: str
    output: str
    error: str | None = None


class BuildResponse(BaseModel):
    """构建响应 - 只返回文件路径，不返回代码内容"""
    success: bool
    mlirPath: str | None = None
    llvmPath: str | None = None
    binPath: str | None = None
    error: str | None = None


class BuildInfoResponse(BaseModel):
    """构建信息响应"""
    projectName: str
    version: str
    platform: str
    arch: str


# ============== 工具函数 ==============

def get_platform_info() -> tuple[str, str]:
    """获取当前平台信息"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    os_name = {
        "windows": "windows",
        "linux": "linux",
        "darwin": "darwin",
    }.get(system, system)
    
    arch = {
        "x86_64": "x64",
        "amd64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(machine, machine)
    
    return os_name, arch


def to_relative_posix(file_path: Path, base_path: Path) -> str:
    """
    将绝对路径转换为相对于 base_path 的 POSIX 风格路径
    
    例如：
    - file_path: D:\\projects\\hello\\build\\mlir\\hello.mlir
    - base_path: D:\\projects\\hello
    - 返回: build/mlir/hello.mlir
    """
    return file_path.relative_to(base_path).as_posix()


def load_project_metadata(project_path: Path) -> dict:
    """加载项目元数据"""
    project_file = project_path / "project.json"
    if not project_file.exists():
        raise FileNotFoundError(f"Project file not found: {project_file}")
    
    with open(project_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_function_graph(functions_dir: Path, func_id: str) -> dict:
    """加载函数图"""
    func_file = functions_dir / f"{func_id}.json"
    if not func_file.exists():
        raise FileNotFoundError(f"Function file not found: {func_file}")
    
    with open(func_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_project_dict(project_path: str) -> dict:
    """
    加载完整的项目字典
    
    统一的项目加载逻辑，供所有 API 使用
    """
    project_dir = Path(project_path)
    
    if not project_dir.exists():
        raise FileNotFoundError(f"Project not found: {project_path}")
    
    metadata = load_project_metadata(project_dir)
    functions_dir = project_dir / "functions"
    
    # 加载主函数
    main_func_id = metadata.get("mainFunctionId", "main")
    main_func_data = load_function_graph(functions_dir, main_func_id)
    
    # 加载自定义函数
    custom_functions = []
    for func_id in metadata.get("customFunctionIds", []):
        func_data = load_function_graph(functions_dir, func_id)
        custom_functions.append(func_data)
    
    return {
        "name": metadata.get("name", project_dir.name),
        "version": metadata.get("version", "0.0.0"),
        "path": project_path,
        "mainFunction": main_func_data,
        "customFunctions": custom_functions,
        "dialects": metadata.get("dialects", []),
    }


# ============== API 端点 ==============

@router.post("/preview", response_model=PreviewResponse)
async def preview_project(request: ProjectRequest):
    """
    预览项目的 MLIR 代码（不保存文件）
    
    从磁盘加载已保存的项目，生成 MLIR 代码并验证。
    """
    try:
        project_dict = load_project_dict(request.projectPath)
        module = build_project_from_dict(project_dict)
        mlir_code = str(module)
        verified = module.operation.verify()
        
        return PreviewResponse(
            success=verified,
            mlirCode=mlir_code,
            verified=verified,
            error=None if verified else "MLIR verification failed",
        )
    except FileNotFoundError as e:
        return PreviewResponse(
            success=False,
            mlirCode="",
            verified=False,
            error=str(e),
        )
    except Exception as e:
        return PreviewResponse(
            success=False,
            mlirCode="",
            verified=False,
            error=str(e),
        )


@router.post("/execute", response_model=ExecuteResponse)
async def execute_project(request: ExecuteRequest):
    """
    执行项目（仅支持 JIT 模式）
    
    使用 MLIR Python bindings JIT 执行，用于编辑器内快速验证。
    """
    from backend.api.execution import _execute_jit
    
    try:
        project_dict = load_project_dict(request.projectPath)
        module = build_project_from_dict(project_dict)
        mlir_code = str(module)
        
        # 验证
        if not module.operation.verify():
            return ExecuteResponse(
                success=False,
                mlirCode=mlir_code,
                output="",
                error="MLIR verification failed",
            )
        
        # JIT 执行
        result = await _execute_jit(mlir_code)
        
        return ExecuteResponse(
            success=result.success,
            mlirCode=mlir_code,
            output=result.output,
            error=result.error,
        )
    except FileNotFoundError as e:
        return ExecuteResponse(
            success=False,
            mlirCode="",
            output="",
            error=str(e),
        )
    except Exception as e:
        return ExecuteResponse(
            success=False,
            mlirCode="",
            output="",
            error=str(e),
        )


@router.post("", response_model=BuildResponse)
async def build_project(request: BuildRequest):
    """
    构建项目，生成 MLIR 文件、LLVM IR 和可执行文件
    """
    try:
        project_dict = load_project_dict(request.projectPath)
        project_dir = Path(request.projectPath)
        project_name = project_dict["name"]
        version = project_dict["version"]
        
        # 生成 MLIR
        module = build_project_from_dict(project_dict)
        final_mlir = str(module)
        
        # 创建 build 目录
        build_dir = project_dir / "build"
        mlir_dir = build_dir / "mlir"
        mlir_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 MLIR 文件
        mlir_file = mlir_dir / f"{project_name}.mlir"
        with open(mlir_file, "w", encoding="utf-8") as f:
            f.write(final_mlir)
        
        response = BuildResponse(
            success=True,
            mlirPath=to_relative_posix(mlir_file, project_dir),
        )
        
        # 生成 LLVM IR
        if request.generateLlvm:
            llvm_dir = build_dir / "llvm"
            llvm_file = llvm_dir / f"{project_name}.ll"
            
            success, _ = await generate_llvm_ir(final_mlir, llvm_file)
            if success:
                response.llvmPath = to_relative_posix(llvm_file, project_dir)
        
        # 生成可执行文件
        if request.generateExecutable and response.llvmPath:
            bin_dir = build_dir / "bin"
            
            success, exe_path = await generate_executable(
                llvm_file,
                bin_dir,
                project_name,
                version,
            )
            if success:
                response.binPath = to_relative_posix(Path(exe_path), project_dir)
        
        return response
        
    except FileNotFoundError as e:
        return BuildResponse(success=False, error=str(e))
    except Exception as e:
        return BuildResponse(success=False, error=f"Build failed: {str(e)}")


@router.post("/info", response_model=BuildInfoResponse)
async def get_build_info(request: ProjectRequest):
    """获取构建信息"""
    project_dir = Path(request.projectPath)
    
    if not project_dir.exists():
        raise Exception(f"Project not found: {request.projectPath}")
    
    metadata = load_project_metadata(project_dir)
    os_name, arch = get_platform_info()
    
    return BuildInfoResponse(
        projectName=metadata.get("name", project_dir.name),
        version=metadata.get("version", "0.0.0"),
        platform=os_name,
        arch=arch,
    )


# ============== 内部函数 ==============

async def generate_llvm_ir(mlir_code: str, output_path: Path) -> tuple[bool, str]:
    """
    将 MLIR 代码转换为 LLVM IR
    
    使用 mlir-opt 和 mlir-translate（来自 mlir_wheel）
    """
    from backend.mlir_utils.paths import find_tool
    
    mlir_opt = find_tool("mlir-opt")
    mlir_translate = find_tool("mlir-translate")
    
    if not mlir_opt:
        return False, "mlir-opt not found"
    if not mlir_translate:
        return False, "mlir-translate not found"
    
    lowering_passes = [
        "--convert-func-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-cf-to-llvm",
        "--convert-scf-to-cf",
        "--convert-index-to-llvm",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    
    returncode, stdout, stderr = await run_command(
        [str(mlir_opt)] + lowering_passes,
        input_data=mlir_code
    )
    
    if returncode != 0:
        return False, f"mlir-opt lowering failed: {stderr}"
    
    lowered_mlir = stdout
    
    returncode, stdout, stderr = await run_command(
        [str(mlir_translate), "--mlir-to-llvmir"],
        input_data=lowered_mlir
    )
    
    if returncode != 0:
        return False, f"mlir-translate failed: {stderr}"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stdout)
    
    return True, ""


async def generate_executable(
    llvm_ir_path: Path,
    output_dir: Path,
    project_name: str,
    version: str,
) -> tuple[bool, str]:
    """
    尝试将 LLVM IR 编译为可执行文件（可选功能）
    
    需要系统安装 clang。如果不可用，返回友好提示。
    """
    import shutil
    
    # 检查系统是否有 clang
    clang = shutil.which("clang")
    if not clang:
        return False, "clang not found in system PATH. Please compile manually:\n  clang output.ll -o program"
    
    os_name, arch = get_platform_info()
    
    if os_name == "windows":
        exe_name = f"{project_name}-{version}-{os_name}-{arch}.exe"
    else:
        exe_name = f"{project_name}-{version}-{os_name}-{arch}"
    
    exe_path = output_dir / exe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用 clang 直接编译 LLVM IR
    returncode, _, stderr = await run_command(
        [clang, str(llvm_ir_path), "-o", str(exe_path)]
    )
    
    if returncode != 0:
        return False, f"clang compilation failed: {stderr}\nYou can try compiling manually:\n  clang {llvm_ir_path} -o {exe_name}"
    
    return True, str(exe_path)

