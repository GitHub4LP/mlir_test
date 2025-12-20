/// <reference types="vite/client" />

// GLSL 着色器文件类型声明
declare module '*.glsl?raw' {
  const content: string;
  export default content;
}

declare module '*.vert.glsl?raw' {
  const content: string;
  export default content;
}

declare module '*.frag.glsl?raw' {
  const content: string;
  export default content;
}

// WGSL 着色器文件类型声明
declare module '*.wgsl?raw' {
  const content: string;
  export default content;
}
