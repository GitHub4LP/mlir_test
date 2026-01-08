/**
 * 端口工具函数
 * 
 * 用于动态端口选择和服务健康检查。
 */

import * as net from 'net';

/** 默认端口范围 */
export const DEFAULT_PORT_START = 8000;
export const DEFAULT_PORT_END = 8100;

/** 检查端口是否可用 */
export function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    
    server.once('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE') {
        resolve(false);
      } else {
        // 其他错误也视为不可用
        resolve(false);
      }
    });
    
    server.once('listening', () => {
      server.close();
      resolve(true);
    });
    
    server.listen(port, '127.0.0.1');
  });
}

/** 在范围内找到可用端口 */
export async function findAvailablePort(
  start: number = DEFAULT_PORT_START,
  end: number = DEFAULT_PORT_END
): Promise<number> {
  for (let port = start; port <= end; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error(`No available port in range ${start}-${end}`);
}

/** 等待服务就绪（健康检查） */
export async function waitForHealth(
  url: string,
  timeout: number = 30000,
  interval: number = 500
): Promise<void> {
  const start = Date.now();
  
  while (Date.now() - start < timeout) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // 服务未就绪，继续等待
    }
    
    await new Promise(resolve => setTimeout(resolve, interval));
  }
  
  throw new Error(`Service not ready after ${timeout}ms: ${url}`);
}

/** 检查服务是否运行 */
export async function isServiceRunning(url: string): Promise<boolean> {
  try {
    const response = await fetch(url);
    return response.ok;
  } catch {
    return false;
  }
}
