/**
 * 开发模式启动脚本
 * 
 * 自动选择可用端口，启动后端和前端。
 */

import { spawn, ChildProcess } from 'child_process';
import { findAvailablePort, waitForHealth, isServiceRunning } from './port-utils';

const HEALTH_CHECK_PATH = '/api/health';

/** 子进程列表，用于清理 */
const processes: ChildProcess[] = [];

/** 清理所有子进程 */
function cleanup() {
  console.log('\n🛑 Shutting down...');
  for (const proc of processes) {
    if (!proc.killed) {
      proc.kill('SIGTERM');
    }
  }
  process.exit(0);
}

/** 启动后端服务 */
async function startBackend(port: number): Promise<ChildProcess> {
  console.log(`🚀 Starting backend on port ${port}...`);
  
  const backend = spawn('uv', [
    'run', 'uvicorn',
    'backend.main:app',
    '--port', port.toString(),
    '--reload'
  ], {
    stdio: 'inherit',
    shell: true,
  });

  processes.push(backend);

  backend.on('error', (err) => {
    console.error('❌ Failed to start backend:', err.message);
    cleanup();
  });

  backend.on('exit', (code) => {
    if (code !== 0 && code !== null) {
      console.error(`❌ Backend exited with code ${code}`);
    }
  });

  return backend;
}

/** 启动前端服务 */
function startFrontend(backendPort: number): ChildProcess {
  console.log('🎨 Starting frontend...');
  
  const frontend = spawn('npm', ['run', 'vite'], {
    stdio: 'inherit',
    shell: true,
    cwd: 'frontend',
    env: {
      ...process.env,
      BACKEND_PORT: backendPort.toString(),
    },
  });

  processes.push(frontend);

  frontend.on('error', (err) => {
    console.error('❌ Failed to start frontend:', err.message);
    cleanup();
  });

  return frontend;
}


/** 主函数 */
async function main() {
  console.log('🔍 Finding available port...');
  
  let port: number;
  try {
    port = await findAvailablePort();
    console.log(`✅ Using port ${port}`);
  } catch (err) {
    console.error('❌ No available port found:', (err as Error).message);
    process.exit(1);
  }

  // 检查是否已有服务运行
  const healthUrl = `http://localhost:${port}${HEALTH_CHECK_PATH}`;
  if (await isServiceRunning(healthUrl)) {
    console.log(`⚠️  Service already running on port ${port}`);
    // 直接启动前端
    startFrontend(port);
    return;
  }

  // 启动后端
  startBackend(port);

  // 等待后端就绪
  console.log('⏳ Waiting for backend to be ready...');
  try {
    await waitForHealth(healthUrl, 60000);
    console.log('✅ Backend is ready');
  } catch (err) {
    console.error('❌ Backend failed to start:', (err as Error).message);
    cleanup();
    return;
  }

  // 启动前端
  startFrontend(port);

  console.log(`
╔════════════════════════════════════════════════════════════╗
║  MLIR Blueprint Development Server                         ║
╠════════════════════════════════════════════════════════════╣
║  Backend:  http://localhost:${port.toString().padEnd(5)}                        ║
║  Frontend: http://localhost:5173                           ║
║                                                            ║
║  Press Ctrl+C to stop                                      ║
╚════════════════════════════════════════════════════════════╝
`);
}

// 处理退出信号
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

// 运行
main().catch((err) => {
  console.error('❌ Startup failed:', err);
  cleanup();
});
