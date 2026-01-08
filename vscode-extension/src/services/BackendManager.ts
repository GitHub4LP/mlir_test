/**
 * 后端管理服务
 * 
 * 负责后端服务的启动、停止和健康检查。
 * 支持自动选择可用端口，避免端口冲突。
 */

import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import { findAvailablePort, waitForHealth } from './port-utils';

export class BackendManager {
  private process: ChildProcess | null = null;
  private port: number | null = null;
  private outputChannel: vscode.OutputChannel;

  constructor(outputChannel: vscode.OutputChannel) {
    this.outputChannel = outputChannel;
  }

  /** 启动后端服务 */
  async start(): Promise<number> {
    const config = vscode.workspace.getConfiguration('mlirBlueprint');
    const autoStart = config.get<boolean>('autoStartBackend', false);
    
    if (!autoStart) {
      // 使用配置的 URL 提取端口
      const url = config.get<string>('backendUrl', 'http://localhost:8000');
      const match = url.match(/:(\d+)/);
      this.port = match ? parseInt(match[1]) : 8000;
      this.outputChannel.appendLine(`Using configured backend URL: ${url}`);
      return this.port;
    }

    // 自动启动后端
    this.outputChannel.appendLine('Finding available port...');
    this.port = await findAvailablePort(8000, 8100);
    this.outputChannel.appendLine(`Using port ${this.port}`);
    
    const pythonPath = config.get<string>('pythonPath', 'python');
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    
    if (!workspaceFolder) {
      throw new Error('No workspace folder found');
    }

    this.outputChannel.appendLine(`Starting backend with ${pythonPath}...`);
    
    this.process = spawn(pythonPath, [
      '-m', 'uvicorn',
      'backend.main:app',
      '--port', this.port.toString(),
      '--host', '127.0.0.1'
    ], {
      cwd: workspaceFolder,
      shell: true,
    });

    // 捕获输出
    this.process.stdout?.on('data', (data) => {
      this.outputChannel.appendLine(`[Backend] ${data.toString().trim()}`);
    });

    this.process.stderr?.on('data', (data) => {
      this.outputChannel.appendLine(`[Backend] ${data.toString().trim()}`);
    });

    this.process.on('error', (err) => {
      this.outputChannel.appendLine(`[Backend Error] ${err.message}`);
      vscode.window.showErrorMessage(`Failed to start backend: ${err.message}`);
    });

    this.process.on('exit', (code) => {
      this.outputChannel.appendLine(`[Backend] Process exited with code ${code}`);
      this.process = null;
    });

    // 等待就绪
    this.outputChannel.appendLine('Waiting for backend to be ready...');
    try {
      await waitForHealth(`http://localhost:${this.port}/api/health`, 30000);
      this.outputChannel.appendLine('Backend is ready');
    } catch (error) {
      this.outputChannel.appendLine(`Backend failed to start: ${error}`);
      this.stop();
      throw error;
    }
    
    return this.port;
  }

  /** 停止后端服务 */
  stop(): void {
    if (this.process) {
      this.outputChannel.appendLine('Stopping backend...');
      this.process.kill();
      this.process = null;
    }
  }

  /** 获取当前端口 */
  getPort(): number | null {
    return this.port;
  }

  /** 获取后端 URL */
  getUrl(): string {
    if (this.port) {
      return `http://localhost:${this.port}`;
    }
    const config = vscode.workspace.getConfiguration('mlirBlueprint');
    return config.get<string>('backendUrl', 'http://localhost:8000');
  }

  /** 检查后端是否运行中 */
  isRunning(): boolean {
    return this.process !== null;
  }
}
