"""
ProjectBuilder 测试

测试项目级 MLIR 构建，包括：
- 单函数构建
- 多函数构建
- 函数调用
- 依赖排序
"""

from backend.services.project_builder import build_project_from_dict


def make_entry_node(func_id: str, func_name: str, outputs: list[dict], is_main: bool = False):
    """创建函数入口节点"""
    return {
        "id": f"{func_id}-entry",
        "type": "function-entry",
        "data": {
            "functionId": func_id,
            "functionName": func_name,
            "outputs": outputs,
            "execOut": {"id": "exec-out", "label": ""},
            "isMain": is_main,
        }
    }


def make_return_node(func_id: str, func_name: str, inputs: list[dict], is_main: bool = False):
    """创建函数返回节点"""
    return {
        "id": f"{func_id}-return",
        "type": "function-return",
        "data": {
            "functionId": func_id,
            "functionName": func_name,
            "branchName": "",
            "inputs": inputs,
            "execIn": {"id": "exec-in", "label": ""},
            "isMain": is_main,
        }
    }


def make_constant_node(node_id: str, value: int, type_str: str = "I32"):
    """创建常量节点（新格式：只有 fullName，没有完整 operation）"""
    return {
        "id": node_id,
        "type": "operation",
        "data": {
            "fullName": "arith.constant",
            "attributes": {"value": value},
            "inputTypes": {},
            "outputTypes": {"result": type_str},
        }
    }


def make_addi_node(node_id: str, type_str: str = "I32"):
    """创建加法节点（新格式：只有 fullName，没有完整 operation）"""
    return {
        "id": node_id,
        "type": "operation",
        "data": {
            "fullName": "arith.addi",
            "attributes": {},
            "inputTypes": {"lhs": type_str, "rhs": type_str},
            "outputTypes": {"result": type_str},
        }
    }


def make_function_call_node(node_id: str, func_id: str, func_name: str, inputs: list[dict], outputs: list[dict]):
    """创建函数调用节点"""
    return {
        "id": node_id,
        "type": "function-call",
        "data": {
            "functionId": func_id,
            "functionName": func_name,
            "inputs": inputs,
            "outputs": outputs,
            "execIn": {"id": "exec-in", "label": ""},
            "execOuts": [{"id": "exec-out-0", "label": ""}],
        }
    }


class TestProjectBuilder:
    """ProjectBuilder 测试"""
    
    def test_single_function_constant_return(self):
        """测试单函数：常量返回"""
        project = {
            "name": "test",
            "path": "./test",
            "mainFunction": {
                "id": "main",
                "name": "main",
                "parameters": [],
                "returnTypes": [{"name": "result", "type": "I32"}],
                "graph": {
                    "nodes": [
                        make_entry_node("main", "main", [], is_main=True),
                        make_return_node("main", "main", [
                            {"id": "return-result", "name": "result", "kind": "input", 
                             "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}
                        ], is_main=True),
                        make_constant_node("const1", 42),
                    ],
                    "edges": [
                        {"id": "e1", "source": "const1", "sourceHandle": "output-result",
                         "target": "main-return", "targetHandle": "return-result"},
                    ],
                },
                "isMain": True,
            },
            "customFunctions": [],
        }
        
        module = build_project_from_dict(project)
        mlir_code = str(module)
        
        assert "func.func @main" in mlir_code
        assert "arith.constant 42" in mlir_code
        assert "return" in mlir_code
        assert module.operation.verify()
    
    def test_single_function_with_params(self):
        """测试单函数：带参数"""
        project = {
            "name": "test",
            "path": "./test",
            "mainFunction": {
                "id": "add",
                "name": "add",
                "parameters": [
                    {"name": "a", "type": "I32"},
                    {"name": "b", "type": "I32"},
                ],
                "returnTypes": [{"name": "result", "type": "I32"}],
                "graph": {
                    "nodes": [
                        make_entry_node("add", "add", [
                            {"id": "param-a", "name": "a", "kind": "output",
                             "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"},
                            {"id": "param-b", "name": "b", "kind": "output",
                             "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"},
                        ]),
                        make_return_node("add", "add", [
                            {"id": "return-result", "name": "result", "kind": "input",
                             "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}
                        ]),
                        make_addi_node("addi1"),
                    ],
                    "edges": [
                        {"id": "e1", "source": "add-entry", "sourceHandle": "param-a",
                         "target": "addi1", "targetHandle": "input-lhs"},
                        {"id": "e2", "source": "add-entry", "sourceHandle": "param-b",
                         "target": "addi1", "targetHandle": "input-rhs"},
                        {"id": "e3", "source": "addi1", "sourceHandle": "output-result",
                         "target": "add-return", "targetHandle": "return-result"},
                    ],
                },
                "isMain": False,
            },
            "customFunctions": [],
        }
        
        module = build_project_from_dict(project)
        mlir_code = str(module)
        
        assert "func.func @add(%arg0: i32, %arg1: i32)" in mlir_code
        assert "arith.addi" in mlir_code
        assert module.operation.verify()
    
    def test_function_call(self):
        """测试函数调用"""
        # add1 函数：返回 x + 1
        add1_func = {
            "id": "add1",
            "name": "add1",
            "parameters": [{"name": "x", "type": "I32"}],
            "returnTypes": [{"name": "result", "type": "I32"}],
            "graph": {
                "nodes": [
                    make_entry_node("add1", "add1", [
                        {"id": "param-x", "name": "x", "kind": "output",
                         "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"},
                    ]),
                    make_return_node("add1", "add1", [
                        {"id": "return-result", "name": "result", "kind": "input",
                         "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}
                    ]),
                    make_constant_node("const1", 1),
                    make_addi_node("addi1"),
                ],
                "edges": [
                    {"id": "e1", "source": "add1-entry", "sourceHandle": "param-x",
                     "target": "addi1", "targetHandle": "input-lhs"},
                    {"id": "e2", "source": "const1", "sourceHandle": "output-result",
                     "target": "addi1", "targetHandle": "input-rhs"},
                    {"id": "e3", "source": "addi1", "sourceHandle": "output-result",
                     "target": "add1-return", "targetHandle": "return-result"},
                ],
            },
            "isMain": False,
        }
        
        # main 函数：调用 add1(10)
        main_func = {
            "id": "main",
            "name": "main",
            "parameters": [],
            "returnTypes": [{"name": "result", "type": "I32"}],
            "graph": {
                "nodes": [
                    make_entry_node("main", "main", [], is_main=True),
                    make_return_node("main", "main", [
                        {"id": "return-result", "name": "result", "kind": "input",
                         "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}
                    ], is_main=True),
                    make_constant_node("const10", 10),
                    make_function_call_node("call1", "add1", "add1",
                        inputs=[{"id": "add1_param_x", "name": "x", "kind": "input",
                                "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}],
                        outputs=[{"id": "add1_return_result", "name": "result", "kind": "output",
                                 "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}],
                    ),
                ],
                "edges": [
                    {"id": "e1", "source": "const10", "sourceHandle": "output-result",
                     "target": "call1", "targetHandle": "add1_param_x"},
                    {"id": "e2", "source": "call1", "sourceHandle": "add1_return_result",
                     "target": "main-return", "targetHandle": "return-result"},
                ],
            },
            "isMain": True,
        }
        
        project = {
            "name": "test",
            "path": "./test",
            "mainFunction": main_func,
            "customFunctions": [add1_func],
        }
        
        module = build_project_from_dict(project)
        mlir_code = str(module)
        
        # 验证生成的代码
        assert "func.func @add1" in mlir_code
        assert "func.func @main" in mlir_code
        assert "call @add1" in mlir_code
        
        # add1 应该在 main 之前（依赖排序）
        add1_pos = mlir_code.find("func.func @add1")
        main_pos = mlir_code.find("func.func @main")
        assert add1_pos < main_pos, "add1 should be defined before main"
        
        assert module.operation.verify()
    
    def test_dependency_ordering(self):
        """测试依赖排序：A 调用 B，B 调用 C"""
        # C 函数
        func_c = {
            "id": "c",
            "name": "c",
            "parameters": [],
            "returnTypes": [{"name": "result", "type": "I32"}],
            "graph": {
                "nodes": [
                    make_entry_node("c", "c", []),
                    make_return_node("c", "c", [
                        {"id": "return-result", "name": "result", "kind": "input",
                         "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}
                    ]),
                    make_constant_node("const1", 1),
                ],
                "edges": [
                    {"id": "e1", "source": "const1", "sourceHandle": "output-result",
                     "target": "c-return", "targetHandle": "return-result"},
                ],
            },
            "isMain": False,
        }
        
        # B 函数：调用 C
        func_b = {
            "id": "b",
            "name": "b",
            "parameters": [],
            "returnTypes": [{"name": "result", "type": "I32"}],
            "graph": {
                "nodes": [
                    make_entry_node("b", "b", []),
                    make_return_node("b", "b", [
                        {"id": "return-result", "name": "result", "kind": "input",
                         "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}
                    ]),
                    make_function_call_node("call_c", "c", "c",
                        inputs=[],
                        outputs=[{"id": "c_return_result", "name": "result", "kind": "output",
                                 "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}],
                    ),
                ],
                "edges": [
                    {"id": "e1", "source": "call_c", "sourceHandle": "c_return_result",
                     "target": "b-return", "targetHandle": "return-result"},
                ],
            },
            "isMain": False,
        }
        
        # A (main) 函数：调用 B
        func_a = {
            "id": "main",
            "name": "main",
            "parameters": [],
            "returnTypes": [{"name": "result", "type": "I32"}],
            "graph": {
                "nodes": [
                    make_entry_node("main", "main", [], is_main=True),
                    make_return_node("main", "main", [
                        {"id": "return-result", "name": "result", "kind": "input",
                         "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}
                    ], is_main=True),
                    make_function_call_node("call_b", "b", "b",
                        inputs=[],
                        outputs=[{"id": "b_return_result", "name": "result", "kind": "output",
                                 "typeConstraint": "I32", "concreteType": "I32", "color": "#4A90D9"}],
                    ),
                ],
                "edges": [
                    {"id": "e1", "source": "call_b", "sourceHandle": "b_return_result",
                     "target": "main-return", "targetHandle": "return-result"},
                ],
            },
            "isMain": True,
        }
        
        project = {
            "name": "test",
            "path": "./test",
            "mainFunction": func_a,
            "customFunctions": [func_b, func_c],  # 故意乱序
        }
        
        module = build_project_from_dict(project)
        mlir_code = str(module)
        
        # 验证顺序：C < B < main
        c_pos = mlir_code.find("func.func @c")
        b_pos = mlir_code.find("func.func @b")
        main_pos = mlir_code.find("func.func @main")
        
        assert c_pos < b_pos < main_pos, f"Expected c < b < main, got c={c_pos}, b={b_pos}, main={main_pos}"
        assert module.operation.verify()
