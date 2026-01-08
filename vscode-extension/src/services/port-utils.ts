/**
 * 端口工具函数
 * 
 * 提供端口可用性检查和服务就绪等待功能。
 */

import * as net from 'net';

/** 检查端口是否可用 */
export function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.once('listening', () => {
      server.close();
      resolve(true);
    });
    server.listen(port, '127.0.0.1');
  });
}

/** 在范围内找到可用端口 */
export async function findAvailablePort(start: number, end: number): Promise<number> {
  for (let port = start; port <= end; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error(`No available port in range ${start}-${end}`);
}

/** 等待服务就绪 */
export async function waitForHealth(url: string, timeout = 30000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // 继续等待
    }
    await new Promise(r => setTimeout(r, 500));
  }
  throw new Error(`Service not ready after ${timeout}ms`);
}
