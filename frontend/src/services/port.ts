/**
 * 端口系统核心模块
 * 
 * 提供类型安全的端口标识符，统一所有节点类型的端口处理。
 * 
 * 设计原则：
 * 1. 单一数据源：所有端口 ID 生成和解析都通过此模块
 * 2. 类型安全：使用字符串字面量类型，编译时检查
 * 3. 序列化友好：toString/parse 支持 JSON 序列化
 * 
 * 端口 ID 格式统一为：kind-name
 * - data-in-lhs：数据输入端口
 * - data-out-result：数据输出端口
 * - exec-in-default：执行输入端口
 * - exec-out-then：执行输出端口
 */

/**
 * 端口类型常量
 */
export const PortKind = {
  /** 数据输入端口 */
  DataIn: 'data-in',
  /** 数据输出端口 */
  DataOut: 'data-out',
  /** 执行输入端口 */
  ExecIn: 'exec-in',
  /** 执行输出端口 */
  ExecOut: 'exec-out',
} as const;

export type PortKindType = typeof PortKind[keyof typeof PortKind];

/**
 * 端口类型辅助函数
 */
export const PortKindUtils = {
  isData(kind: PortKindType): boolean {
    return kind === PortKind.DataIn || kind === PortKind.DataOut;
  },
  
  isExec(kind: PortKindType): boolean {
    return kind === PortKind.ExecIn || kind === PortKind.ExecOut;
  },
  
  isInput(kind: PortKindType): boolean {
    return kind === PortKind.DataIn || kind === PortKind.ExecIn;
  },
  
  isOutput(kind: PortKindType): boolean {
    return kind === PortKind.DataOut || kind === PortKind.ExecOut;
  },
  
  /** 验证是否为有效的端口类型 */
  isValid(kind: string): kind is PortKindType {
    return kind === PortKind.DataIn || kind === PortKind.DataOut ||
           kind === PortKind.ExecIn || kind === PortKind.ExecOut;
  },
};

/**
 * 端口引用：类型安全的端口标识符
 * 
 * 用于唯一标识图中的任意端口，支持：
 * - 传播图构建
 * - 连接验证
 * - 类型状态管理
 * 
 * 完整格式（用于 Map 键）：nodeId:kind:name
 * Handle 格式（用于 React Flow）：kind-name
 */
export class PortRef {
  readonly nodeId: string;
  readonly kind: PortKindType;
  readonly name: string;
  
  constructor(nodeId: string, kind: PortKindType, name: string) {
    this.nodeId = nodeId;
    this.kind = kind;
    this.name = name;
  }
  
  /** 获取唯一键，用于 Map/Set */
  get key(): string {
    return `${this.nodeId}:${this.kind}:${this.name}`;
  }
  
  /** 获取用于 React Flow handle 的 ID */
  get handleId(): string {
    return `${this.kind}-${this.name}`;
  }
  
  /** 序列化为字符串 */
  toString(): string {
    return this.key;
  }
  
  /** 判断是否为数据端口 */
  isData(): boolean {
    return PortKindUtils.isData(this.kind);
  }
  
  /** 判断是否为执行端口 */
  isExec(): boolean {
    return PortKindUtils.isExec(this.kind);
  }
  
  /** 判断是否为输入端口 */
  isInput(): boolean {
    return PortKindUtils.isInput(this.kind);
  }
  
  /** 判断是否为输出端口 */
  isOutput(): boolean {
    return PortKindUtils.isOutput(this.kind);
  }
  
  /** 判断两个端口引用是否相等 */
  equals(other: PortRef): boolean {
    return this.key === other.key;
  }
  
  /**
   * 从完整键解析端口引用
   * 格式：nodeId:kind:name
   */
  static parse(key: string): PortRef | null {
    const parts = key.split(':');
    if (parts.length !== 3) return null;
    
    const [nodeId, kind, name] = parts;
    if (!nodeId || !name || !PortKindUtils.isValid(kind)) {
      return null;
    }
    
    return new PortRef(nodeId, kind, name);
  }
  
  /**
   * 从 nodeId 和 handleId 创建端口引用
   * handleId 格式：kind-name 或 kind（执行引脚可能没有名称）
   */
  static fromHandle(nodeId: string, handleId: string): PortRef | null {
    const parsed = PortRef.parseHandleId(handleId);
    if (!parsed) return null;
    return new PortRef(nodeId, parsed.kind, parsed.name);
  }
  
  /**
   * 解析 handleId 获取端口类型和名称
   * handleId 格式：kind-name 或 kind（执行引脚默认名称为 default）
   * - data-in-{name}, data-out-{name}：数据端口
   * - exec-in, exec-in-{name}, exec-out, exec-out-{name}：执行端口
   */
  static parseHandleId(handleId: string): { kind: PortKindType; name: string } | null {
    const kindPattern = `(${PortKind.DataIn}|${PortKind.DataOut}|${PortKind.ExecIn}|${PortKind.ExecOut})`;
    const regex = new RegExp(`^${kindPattern}(?:-(.+))?$`);
    const match = handleId.match(regex);
    
    if (match) {
      const kind = match[1] as PortKindType;
      const name = match[2] || 'default';
      return { kind, name };
    }
    
    return null;
  }
}

// ============ 便捷工厂函数 ============

/** 创建数据输入端口引用 */
export function dataIn(nodeId: string, name: string): PortRef {
  return new PortRef(nodeId, PortKind.DataIn, name);
}

/** 创建数据输出端口引用 */
export function dataOut(nodeId: string, name: string): PortRef {
  return new PortRef(nodeId, PortKind.DataOut, name);
}

/** 创建执行输入端口引用 */
export function execIn(nodeId: string, name: string): PortRef {
  return new PortRef(nodeId, PortKind.ExecIn, name);
}

/** 创建执行输出端口引用 */
export function execOut(nodeId: string, name: string): PortRef {
  return new PortRef(nodeId, PortKind.ExecOut, name);
}

// ============ Handle ID 生成函数（用于组件） ============

/** 生成数据输入端口的 handleId */
export function dataInHandle(name: string): string {
  return `${PortKind.DataIn}-${name}`;
}

/** 生成数据输出端口的 handleId */
export function dataOutHandle(name: string): string {
  return `${PortKind.DataOut}-${name}`;
}

/** 生成执行输入端口的 handleId */
export function execInHandle(name: string): string {
  return `${PortKind.ExecIn}-${name}`;
}

/** 生成执行输出端口的 handleId */
export function execOutHandle(name: string): string {
  return `${PortKind.ExecOut}-${name}`;
}
