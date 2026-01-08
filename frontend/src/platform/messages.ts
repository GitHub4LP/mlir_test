/**
 * 消息协议定义
 * 
 * 用于 Webview 与 Extension 之间的通讯。
 * Web 模式下不使用，但类型定义保持一致。
 */

/** 基础消息结构 */
export interface Message {
  /** 消息类型 */
  type: string;
  /** 请求 ID，用于请求-响应匹配 */
  requestId?: string;
  /** 消息载荷 */
  payload?: unknown;
}

/** 请求消息类型 */
export type RequestType =
  | 'callApi'
  | 'showMlirCode'
  | 'appendOutput'
  | 'clearOutput'
  | 'showOpenDialog'
  | 'showSaveDialog'
  | 'showNotification';

/** 事件消息类型 */
export type EventType =
  | 'selectionChanged'
  | 'propertyChanged'
  | 'projectLoaded'
  | 'backendReady'
  | 'connectionStatus';

/** API 请求消息 */
export interface CallApiMessage extends Message {
  type: 'callApi';
  payload: {
    endpoint: string;
    method: string;
    body?: unknown;
  };
}

/** 显示 MLIR 代码消息 */
export interface ShowMlirCodeMessage extends Message {
  type: 'showMlirCode';
  payload: {
    code: string;
    verified: boolean;
  };
}

/** 追加输出消息 */
export interface AppendOutputMessage extends Message {
  type: 'appendOutput';
  payload: {
    message: string;
    type: 'info' | 'success' | 'error' | 'output';
  };
}


/** 清空输出消息 */
export interface ClearOutputMessage extends Message {
  type: 'clearOutput';
}

/** 显示对话框消息 */
export interface ShowDialogMessage extends Message {
  type: 'showOpenDialog' | 'showSaveDialog';
  payload: {
    title?: string;
    filters?: Array<{ name: string; extensions: string[] }>;
    canSelectFolders?: boolean;
  };
}

/** 显示通知消息 */
export interface ShowNotificationMessage extends Message {
  type: 'showNotification';
  payload: {
    message: string;
    type: 'info' | 'warning' | 'error';
  };
}

/** 选中变化消息 */
export interface SelectionChangedMessage extends Message {
  type: 'selectionChanged';
  payload: {
    nodeIds: string[];
    nodeData: unknown;
  };
}

/** 属性变化消息 */
export interface PropertyChangedMessage extends Message {
  type: 'propertyChanged';
  payload: {
    nodeId: string;
    property: string;
    value: unknown;
  };
}

/** 后端就绪消息 */
export interface BackendReadyMessage extends Message {
  type: 'backendReady';
  payload: {
    port: number;
    url: string;
  };
}

/** 响应消息 */
export interface ResponseMessage extends Message {
  type: 'response';
  requestId: string;
  payload: unknown;
}

/** 错误消息 */
export interface ErrorMessage extends Message {
  type: 'error';
  requestId: string;
  payload: string;
}

/** 所有消息类型联合 */
export type AnyMessage =
  | CallApiMessage
  | ShowMlirCodeMessage
  | AppendOutputMessage
  | ClearOutputMessage
  | ShowDialogMessage
  | ShowNotificationMessage
  | SelectionChangedMessage
  | PropertyChangedMessage
  | BackendReadyMessage
  | ResponseMessage
  | ErrorMessage;
