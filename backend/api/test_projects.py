"""
Project API Tests

Tests for project save/load functionality.
Requirements: 1.2, 1.3

注意：API 使用 POST + 请求体传递路径，避免 URL 编码问题
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for project tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def create_test_project(name: str = "Test Project", path: str = "/test"):
    """Create a minimal test project structure."""
    return {
        "name": name,
        "path": path,
        "mainFunction": {
            "id": "main",
            "name": "main",
            "parameters": [],
            "returnTypes": [],
            "graph": {"nodes": [], "edges": []},
            "isMain": True
        },
        "customFunctions": [],
        "dialects": ["arith", "func"]
    }


class TestProjectAPI:
    """Tests for project API endpoints."""

    def test_create_project(self, temp_project_dir):
        """Test creating a new project."""
        response = client.post(
            "/api/projects/",
            json={"name": "New Project", "path": temp_project_dir}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Project"
        assert data["path"] == temp_project_dir
        
        # Verify files were created
        project_file = Path(temp_project_dir) / "project.mlir.json"
        assert project_file.exists()
        
        functions_dir = Path(temp_project_dir) / "functions"
        assert functions_dir.exists()
        
        main_func_file = functions_dir / "main.json"
        assert main_func_file.exists()

    def test_save_project(self, temp_project_dir):
        """Test saving a project to disk."""
        project = create_test_project(path=temp_project_dir)
        
        # 使用新的 POST /save API
        response = client.post(
            "/api/projects/save",
            json={"project": project}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "saved"
        
        # Verify project.mlir.json was created
        project_file = Path(temp_project_dir) / "project.mlir.json"
        assert project_file.exists()
        
        with open(project_file, "r") as f:
            saved_data = json.load(f)
        
        assert saved_data["name"] == "Test Project"
        assert saved_data["mainFunctionId"] == "main"

    def test_load_project(self, temp_project_dir):
        """Test loading a project from disk."""
        # First save a project
        project = create_test_project(path=temp_project_dir)
        client.post(
            "/api/projects/save",
            json={"project": project}
        )
        
        # Then load it using POST /load
        response = client.post(
            "/api/projects/load",
            json={"projectPath": temp_project_dir}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "project" in data
        
        loaded_project = data["project"]
        assert loaded_project["name"] == "Test Project"
        assert loaded_project["mainFunction"]["id"] == "main"
        # dialects 从节点自动提取，空项目无节点则为�?        assert loaded_project["dialects"] == []

    def test_load_nonexistent_project(self, temp_project_dir):
        """Test loading a project that doesn't exist."""
        nonexistent_path = os.path.join(temp_project_dir, "nonexistent")
        
        response = client.post(
            "/api/projects/load",
            json={"projectPath": nonexistent_path}
        )
        
        assert response.status_code == 404

    def test_save_and_load_with_custom_functions(self, temp_project_dir):
        """Test saving and loading a project with custom functions."""
        project = create_test_project(path=temp_project_dir)
        project["customFunctions"] = [
            {
                "id": "func_helper",
                "name": "helper",
                "parameters": [{"name": "x", "constraint": "i32"}],
                "returnTypes": [{"name": "result", "constraint": "i32"}],
                "graph": {"nodes": [], "edges": []},
                "isMain": False
            }
        ]
        
        # Save
        response = client.post(
            "/api/projects/save",
            json={"project": project}
        )
        assert response.status_code == 200
        
        # Load
        response = client.post(
            "/api/projects/load",
            json={"projectPath": temp_project_dir}
        )
        assert response.status_code == 200
        
        loaded = response.json()["project"]
        assert len(loaded["customFunctions"]) == 1
        assert loaded["customFunctions"][0]["name"] == "helper"
        assert loaded["customFunctions"][0]["parameters"][0]["name"] == "x"

    def test_save_project_with_nodes(self, temp_project_dir):
        """Test saving a project with nodes in the graph."""
        project = create_test_project(path=temp_project_dir)
        project["mainFunction"]["graph"]["nodes"] = [
            {
                "id": "node_1",
                "type": "operation",
                "position": {"x": 100, "y": 200},
                "data": {
                    "fullName": "arith.addi",
                    "attributes": {},
                    "inputTypes": {"lhs": "I32", "rhs": "I32"},
                    "outputTypes": {"result": "I32"}
                }
            }
        ]
        
        # Save
        response = client.post(
            "/api/projects/save",
            json={"project": project}
        )
        assert response.status_code == 200
        
        # Load
        response = client.post(
            "/api/projects/load",
            json={"projectPath": temp_project_dir}
        )
        assert response.status_code == 200
        
        loaded = response.json()["project"]
        assert len(loaded["mainFunction"]["graph"]["nodes"]) == 1
        assert loaded["mainFunction"]["graph"]["nodes"][0]["id"] == "node_1"
        assert loaded["mainFunction"]["graph"]["nodes"][0]["position"]["x"] == 100

    def test_delete_project(self, temp_project_dir):
        """Test deleting a project."""
        # First create a project
        project = create_test_project(path=temp_project_dir)
        client.post(
            "/api/projects/save",
            json={"project": project}
        )
        
        # Verify it exists
        assert Path(temp_project_dir).exists()
        
        # Delete it using POST /delete
        response = client.post(
            "/api/projects/delete",
            json={"projectPath": temp_project_dir}
        )
        assert response.status_code == 200
        
        # Verify it's gone
        assert not Path(temp_project_dir).exists()

    def test_round_trip_preserves_data(self, temp_project_dir):
        """Test that save/load round trip preserves all project data."""
        project = create_test_project(path=temp_project_dir)
        # dialects 从节点自动提取，不再手动指定
        project["mainFunction"]["parameters"] = [
            {"name": "input", "type": "tensor<4x4xf32>"}
        ]
        project["mainFunction"]["returnTypes"] = [
            {"name": "output", "type": "tensor<4x4xf32>"}
        ]
        
        # Save
        client.post(
            "/api/projects/save",
            json={"project": project}
        )
        
        # Load
        response = client.post(
            "/api/projects/load",
            json={"projectPath": temp_project_dir}
        )
        loaded = response.json()["project"]
        
        # Verify all data is preserved
        assert loaded["name"] == project["name"]
        # dialects 从节点自动提取，空项目无节点则为�?        assert loaded["dialects"] == []
        assert loaded["mainFunction"]["parameters"] == project["mainFunction"]["parameters"]
        assert loaded["mainFunction"]["returnTypes"] == project["mainFunction"]["returnTypes"]
