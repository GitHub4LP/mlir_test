/**
 * Webview 工具函数
 * 
 * 提供 Webview HTML 生成和安全相关功能。
 */

import * as vscode from 'vscode';
import * as fs from 'fs';

/** 生成随机 nonce 用于 CSP */
export function getNonce(): string {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}

/** Webview 入口类型 */
export type WebviewEntry = 'editor' | 'properties';

/**
 * 扫描 chunks 目录，生成 Import Map
 * 
 * Import Map 让动态 import 的相对路径能正确解析到 webview URI
 */
function generateImportMap(
  webview: vscode.Webview,
  extensionUri: vscode.Uri
): string {
  const chunksDir = vscode.Uri.joinPath(extensionUri, 'media', 'chunks');
  const chunksPath = chunksDir.fsPath;
  
  const imports: Record<string, string> = {};
  
  try {
    if (fs.existsSync(chunksPath)) {
      const files = fs.readdirSync(chunksPath);
      for (const file of files) {
        if (file.endsWith('.js')) {
          const chunkUri = webview.asWebviewUri(
            vscode.Uri.joinPath(chunksDir, file)
          );
          // 映射相对路径到 webview URI
          imports[`./chunks/${file}`] = chunkUri.toString();
        }
      }
    }
  } catch (e) {
    console.error('Failed to scan chunks directory:', e);
  }
  
  return JSON.stringify({ imports }, null, 2);
}

/**
 * 生成 Webview HTML 内容
 * 
 * @param webview - Webview 实例
 * @param extensionUri - 扩展 URI
 * @param entry - 入口类型
 * @param backendPort - 后端端口（可选）
 */
export function getWebviewContent(
  webview: vscode.Webview,
  extensionUri: vscode.Uri,
  entry: WebviewEntry,
  backendPort?: number
): string {
  // 获取资源 URI
  const mediaUri = vscode.Uri.joinPath(extensionUri, 'media');
  const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(mediaUri, `${entry}.js`));
  const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(mediaUri, 'assets', 'style.css'));
  
  // 生成 Import Map 用于动态 import 路径解析
  const importMap = generateImportMap(webview, extensionUri);
  
  // CSP 配置
  const cspSource = webview.cspSource;
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="
    default-src 'none';
    style-src ${cspSource} 'unsafe-inline';
    script-src ${cspSource} 'unsafe-inline';
    img-src ${cspSource} data:;
    font-src ${cspSource};
    connect-src http://localhost:* https://localhost:* ${cspSource};
  ">
  <link href="${styleUri}" rel="stylesheet">
  <title>MLIR Blueprint</title>
  <script type="importmap">
${importMap}
  </script>
</head>
<body>
  <div id="root"></div>
  <script>
    // 传递后端端口给前端
    window.__MLIR_BLUEPRINT_CONFIG__ = {
      backendPort: ${backendPort || 8000},
      entry: '${entry}'
    };
  </script>
  <script type="module" src="${scriptUri}"></script>
</body>
</html>`;
}

/**
 * 生成简单的占位 HTML（用于资源未构建时）
 */
export function getPlaceholderContent(message: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      font-family: var(--vscode-font-family);
      color: var(--vscode-foreground);
      background-color: var(--vscode-editor-background);
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }
    .message {
      text-align: center;
      padding: 20px;
    }
    .message h2 {
      margin-bottom: 10px;
    }
    .message p {
      color: var(--vscode-descriptionForeground);
    }
  </style>
</head>
<body>
  <div class="message">
    <h2>MLIR Blueprint</h2>
    <p>${message}</p>
  </div>
</body>
</html>`;
}
