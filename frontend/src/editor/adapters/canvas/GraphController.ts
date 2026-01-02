/**
 * 图控制器 - 统一业务逻辑层
 * 
 * 设计原则：
 * - 所有业务逻辑集中在此，渲染后端只负责显示
 * - 负责：计算渲染数据、命中测试、状态机、业务决策
 * - 不直接操作 DOM，通过 IRenderer 接口与渲染后端交互
 */

import type { IRenderer } from './IRenderer';
import type { RawInput } from './input';
import type {
  RenderData,
  Viewport,
  RenderRect,
  RenderText,
  RenderPath,
  RenderCircle,
  RenderTriangle,
  InteractionHint,
  OverlayInfo,
} from './types';
import { createEmptyRenderData, createDefaultViewport, createDefaultHint } from './types';
import type { GraphNode, GraphState } from '../../../types';
import {
  computeEdgePath,
  distanceToEdge,
  isPointInRect,
  isPointInCircle,
  NODE_LAYOUT,
  EDGE_LAYOUT,
} from './layout';
// 统一布局系统 - 所有尺寸和位置信息都从 LayoutBox 获取
import { computeNodeLayoutBox, extractHandlePositions, type LayoutBox, type HandlePosition } from '../../core/layout';
import { performanceMonitor } from './PerformanceMonitor';

// ============================================================
// 控制器状态类型
// ============================================================

/** 空闲状态 */
interface IdleState {
  kind: 'idle';
}

/** 拖拽节点状态 */
interface DraggingNodeState {
  kind: 'dragging-node';
  /** 主拖拽节点 ID（鼠标点击的节点） */
  nodeId: string;
  /** 拖拽开始时的画布坐标 */
  startX: number;
  startY: number;
  /** 所有被拖拽节点的初始位置 */
  nodeStartPositions: Map<string, { x: number; y: number }>;
}

/** 拖拽视口状态 */
interface DraggingViewportState {
  kind: 'dragging-viewport';
  /** 拖拽开始时的屏幕坐标 */
  startX: number;
  startY: number;
  /** 视口原始位置 */
  viewportStartX: number;
  viewportStartY: number;
}

/** 创建连接状态 */
interface ConnectingState {
  kind: 'connecting';
  sourceNodeId: string;
  sourceHandleId: string;
  /** 当前鼠标位置（画布坐标） */
  currentX: number;
  currentY: number;
}

/** 框选状态 */
interface BoxSelectingState {
  kind: 'box-selecting';
  /** 框选起点（画布坐标） */
  startX: number;
  startY: number;
  /** 当前鼠标位置（画布坐标） */
  currentX: number;
  currentY: number;
}

/** 控制器状态联合类型 */
export type ControllerState =
  | IdleState
  | DraggingNodeState
  | DraggingViewportState
  | ConnectingState
  | BoxSelectingState;

// ============================================================
// 命中测试结果类型
// ============================================================

/** 未命中 */
interface HitNone {
  kind: 'none';
}

/** 命中节点 */
interface HitNode {
  kind: 'node';
  nodeId: string;
}

/** 命中端口 */
interface HitHandle {
  kind: 'handle';
  nodeId: string;
  handleId: string;
  isOutput: boolean;
}

/** 命中边 */
interface HitEdge {
  kind: 'edge';
  edgeId: string;
}

/** 命中测试结果联合类型 */
export type HitResult = HitNone | HitNode | HitHandle | HitEdge;

// ============================================================
// GraphController 类
// ============================================================

/**
 * 图控制器
 * 
 * 统一的业务逻辑层，所有渲染后端共享同一实例。
 */
export class GraphController {
  // 渲染后端
  private renderer: IRenderer | null = null;
  
  // 视口状态
  private viewport: Viewport = createDefaultViewport();
  
  // 控制器状态机
  private state: ControllerState = { kind: 'idle' };
  
  // 选中的节点 ID 集合
  private selectedNodeIds: Set<string> = new Set();
  
  // 选中的边 ID 集合
  private selectedEdgeIds: Set<string> = new Set();
  
  // 渲染请求标志（用于合并多次渲染请求）
  private renderRequested: boolean = false;
  
  // 缓存的渲染数据
  private cachedRenderData: RenderData = createEmptyRenderData();

  // ============================================================
  // 外部回调（用于与应用层通信）
  // ============================================================

  /** 节点位置变化回调 */
  onNodePositionChange: ((nodeId: string, x: number, y: number) => void) | null = null;

  // 渲染扩展已移至 LayoutBox 系统，不再需要 onExtendRenderData 回调

  /** 连接尝试回调 */
  onConnectionAttempt: ((
    sourceNodeId: string,
    sourceHandleId: string,
    targetNodeId: string,
    targetHandleId: string
  ) => void) | null = null;

  /** 删除选中元素回调 */
  onDeleteSelected: (() => void) | null = null;

  /** 拖拽状态变化回调（用于渲染优化） */
  onDragStateChange: ((isDragging: boolean) => void) | null = null;

  /** 缩放状态变化回调（用于渲染优化） */
  onZoomStateChange: ((isZooming: boolean) => void) | null = null;

  /** 选择变化回调 */
  onSelectionChange: ((nodeIds: string[], edgeIds: string[]) => void) | null = null;

  /** 视口变化回调 */
  onViewportChange: ((viewport: Viewport) => void) | null = null;

  /** 双击边删除回调 */
  onEdgeDoubleClick: ((edgeId: string) => void) | null = null;

  /** 拖放回调 */
  onDrop: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null = null;

  /** 扩展命中测试回调（用于类型标签、按钮等交互区域）
   * 返回 true 表示已处理，不再进行默认处理
   */
  onExtendedHitTest: ((screenX: number, screenY: number) => boolean) | null = null;

  /** UI 事件预处理回调（用于 Canvas UI 组件优先处理事件）
   * 返回 true 表示事件已被 UI 组件处理，不再进行默认处理
   */
  onPreMouseDown: ((screenX: number, screenY: number, button: number) => boolean) | null = null;
  onPreMouseMove: ((screenX: number, screenY: number) => boolean) | null = null;
  onPreMouseUp: ((screenX: number, screenY: number) => boolean) | null = null;
  onPreWheel: ((screenX: number, screenY: number, deltaX: number, deltaY: number) => boolean) | null = null;
  onPreKeyDown: ((key: string, code: string, ctrlKey: boolean, shiftKey: boolean, altKey: boolean) => boolean) | null = null;

  /** Hover 状态变化回调（用于显示删除按钮等） */
  onHoverChange: ((screenX: number, screenY: number) => void) | null = null;

  // 双击检测
  private lastClickTime: number = 0;
  private lastClickTarget: HitResult = { kind: 'none' };

  // ============================================================
  // 渲染后端管理
  // ============================================================

  /**
   * 设置渲染后端
   */
  setRenderer(renderer: IRenderer): void {
    // 如果有旧后端且不是同一个，先卸载
    if (this.renderer && this.renderer !== renderer) {
      this.renderer.unmount();
    }
    
    this.renderer = renderer;
    
    // 注册输入回调
    renderer.onInput((input: RawInput) => this.handleInput(input));
  }

  /**
   * 获取当前渲染后端
   */
  getRenderer(): IRenderer | null {
    return this.renderer;
  }

  /**
   * 切换渲染后端
   * 保留视口状态
   */
  switchRenderer(newRenderer: IRenderer, container: HTMLElement): void {
    // 保存当前视口
    const savedViewport = { ...this.viewport };
    
    // 卸载旧后端
    if (this.renderer) {
      this.renderer.unmount();
    }
    
    // 设置新后端
    this.renderer = newRenderer;
    newRenderer.onInput((input: RawInput) => this.handleInput(input));
    newRenderer.mount(container);
    
    // 恢复视口
    this.viewport = savedViewport;
    
    // 触发渲染
    this.requestRender();
  }

  // ============================================================
  // 坐标转换
  // ============================================================

  /**
   * 屏幕坐标 → 画布坐标
   */
  screenToCanvas(screenX: number, screenY: number): { x: number; y: number } {
    return {
      x: (screenX - this.viewport.x) / this.viewport.zoom,
      y: (screenY - this.viewport.y) / this.viewport.zoom,
    };
  }

  /**
   * 画布坐标 → 屏幕坐标
   */
  canvasToScreen(canvasX: number, canvasY: number): { x: number; y: number } {
    return {
      x: canvasX * this.viewport.zoom + this.viewport.x,
      y: canvasY * this.viewport.zoom + this.viewport.y,
    };
  }

  // ============================================================
  // 视口控制
  // ============================================================

  /**
   * 获取当前视口
   */
  getViewport(): Viewport {
    return { ...this.viewport };
  }

  /**
   * 设置视口
   */
  setViewport(viewport: Viewport): void {
    this.viewport = { ...viewport };
    this.onViewportChange?.(this.viewport);
    this.requestRender();
  }

  /**
   * 设置视口（静默，不触发回调）
   * 用于外部同步视口状态，避免循环
   */
  setViewportSilent(viewport: Viewport): void {
    this.viewport = { ...viewport };
    this.requestRender();
  }

  /**
   * 平移视口
   */
  panViewport(deltaX: number, deltaY: number): void {
    this.viewport.x += deltaX;
    this.viewport.y += deltaY;
    this.onViewportChange?.(this.viewport);
    this.requestRender();
  }

  /**
   * 缩放视口（以指定世界坐标点为中心）
   */
  zoomViewport(factor: number, centerX: number, centerY: number): void {
    const oldZoom = this.viewport.zoom;
    const newZoom = Math.max(0.1, Math.min(4, oldZoom * factor));
    
    // 保持缩放中心点不变（世界坐标）
    this.viewport.x = centerX * oldZoom + this.viewport.x - centerX * newZoom;
    this.viewport.y = centerY * oldZoom + this.viewport.y - centerY * newZoom;
    this.viewport.zoom = newZoom;
    
    this.onViewportChange?.(this.viewport);
    this.requestRender();
  }

  /**
   * 缩放视口（以指定屏幕坐标点为中心）
   */
  zoomViewportAtScreen(factor: number, screenX: number, screenY: number): void {
    const oldZoom = this.viewport.zoom;
    const newZoom = Math.max(0.1, Math.min(4, oldZoom * factor));
    
    // 保持屏幕坐标点对应的世界坐标不变
    // 屏幕坐标 screenX = worldX * zoom + viewport.x
    // worldX = (screenX - viewport.x) / zoom
    // 缩放后：screenX = worldX * newZoom + newViewport.x
    // newViewport.x = screenX - worldX * newZoom
    
    const worldX = (screenX - this.viewport.x) / oldZoom;
    const worldY = (screenY - this.viewport.y) / oldZoom;
    
    this.viewport.x = screenX - worldX * newZoom;
    this.viewport.y = screenY - worldY * newZoom;
    this.viewport.zoom = newZoom;
    
    this.onViewportChange?.(this.viewport);
    this.requestRender();
  }

  // ============================================================
  // 状态机
  // ============================================================

  /**
   * 获取当前状态
   */
  getState(): ControllerState {
    return this.state;
  }

  /**
   * 处理原始输入
   * 这是状态机的入口点
   */
  handleInput(input: RawInput): void {
    switch (input.kind) {
      case 'pointer':
        this.handlePointerInput(input.data);
        break;
      case 'wheel':
        this.handleWheelInput(input.data);
        break;
      case 'key':
        this.handleKeyInput(input.data);
        break;
    }
  }

  /**
   * 处理指针输入
   * 注意：input.data.x/y 是屏幕坐标（相对于 canvas 左上角）
   */
  private handlePointerInput(data: import('./input').PointerInput): void {
    // UI 事件预处理（Canvas UI 组件优先）
    if (data.type === 'down' && this.onPreMouseDown?.(data.x, data.y, data.button)) {
      return; // UI 组件已处理
    }
    if (data.type === 'move' && this.onPreMouseMove?.(data.x, data.y)) {
      return; // UI 组件已处理
    }
    if (data.type === 'up' && this.onPreMouseUp?.(data.x, data.y)) {
      return; // UI 组件已处理
    }

    switch (this.state.kind) {
      case 'idle':
        this.handlePointerInIdle(data);
        break;
      case 'dragging-node':
        this.handlePointerInDraggingNode(data);
        break;
      case 'dragging-viewport':
        this.handlePointerInDraggingViewport(data);
        break;
      case 'connecting':
        this.handlePointerInConnecting(data);
        break;
      case 'box-selecting':
        this.handlePointerInBoxSelecting(data);
        break;
    }
  }

  /**
   * 空闲状态下处理指针输入
   * data.x/y 是屏幕坐标
   */
  private handlePointerInIdle(data: import('./input').PointerInput): void {
    // 处理 move 事件 - 更新 hover 状态
    if (data.type === 'move') {
      this.onHoverChange?.(data.x, data.y);
      return;
    }

    if (data.type !== 'down') return;

    // 先尝试扩展命中测试（类型标签、按钮等）
    // 只在左键点击时触发
    if (data.button === 0 && this.onExtendedHitTest?.(data.x, data.y)) {
      return; // 已处理，不再进行默认处理
    }

    // 转换为世界坐标用于命中测试
    const worldPos = this.screenToCanvas(data.x, data.y);
    const hit = this.hitTest(worldPos.x, worldPos.y);

    // 双击检测
    const now = Date.now();
    const isDoubleClick = now - this.lastClickTime < 300 && 
      this.lastClickTarget.kind === hit.kind &&
      ((hit.kind === 'edge' && this.lastClickTarget.kind === 'edge' && hit.edgeId === this.lastClickTarget.edgeId) ||
       (hit.kind === 'node' && this.lastClickTarget.kind === 'node' && hit.nodeId === this.lastClickTarget.nodeId));
    this.lastClickTime = now;
    this.lastClickTarget = hit;

    // 双击边 → 删除边
    if (isDoubleClick && hit.kind === 'edge') {
      this.onEdgeDoubleClick?.(hit.edgeId);
      return;
    }

    switch (hit.kind) {
      case 'handle':
        // 点击端口 → 开始创建连接（存储世界坐标）
        this.state = {
          kind: 'connecting',
          sourceNodeId: hit.nodeId,
          sourceHandleId: hit.handleId,
          currentX: worldPos.x,
          currentY: worldPos.y,
        };
        break;

      case 'node':
        // 点击节点
        if (data.modifiers.ctrl || data.modifiers.meta) {
          // Ctrl+点击（或 Mac 上的 Cmd+点击）→ 多选
          if (this.selectedNodeIds.has(hit.nodeId)) {
            this.deselectNode(hit.nodeId);
          } else {
            this.selectNode(hit.nodeId, true);
          }
        } else {
          // 普通点击 → 选择并开始拖拽
          if (!this.selectedNodeIds.has(hit.nodeId)) {
            this.selectNode(hit.nodeId, false);
          }
          
          // 收集所有选中节点的初始位置（直接使用节点的 position）
          const nodeStartPositions = new Map<string, { x: number; y: number }>();
          const graphData = this.getGraphData();
          if (graphData) {
            for (const nodeId of this.selectedNodeIds) {
              const node = graphData.nodes.find(n => n.id === nodeId);
              if (node) {
                // 直接使用节点的 position，不需要通过布局系统
                nodeStartPositions.set(nodeId, { x: node.position.x, y: node.position.y });
              }
            }
          }
          
          if (nodeStartPositions.size > 0) {
            this.state = {
              kind: 'dragging-node',
              nodeId: hit.nodeId,
              startX: worldPos.x,
              startY: worldPos.y,
              nodeStartPositions,
            };
            // 通知拖拽开始
            this.onDragStateChange?.(true);
          }
        }
        break;

      case 'edge':
        // 点击边 → 选择边
        this.selectedNodeIds.clear();
        this.selectedEdgeIds.clear();
        this.selectedEdgeIds.add(hit.edgeId);
        this.notifySelectionChange();
        this.requestRender();
        break;

      case 'none':
        // 点击空白区域
        // 匹配 ReactFlow 行为：左键拖拽 = 框选，中键拖拽 = 平移
        if (data.button === 0) {
          // 左键点击空白 → 清除选择并开始框选
          this.clearSelection();
          this.state = {
            kind: 'box-selecting',
            startX: worldPos.x,
            startY: worldPos.y,
            currentX: worldPos.x,
            currentY: worldPos.y,
          };
        } else if (data.button === 1 || data.button === 2) {
          // 中键或右键点击空白 → 开始拖拽视口
          // 存储屏幕坐标！这是关键
          this.state = {
            kind: 'dragging-viewport',
            startX: data.x,  // 屏幕坐标
            startY: data.y,  // 屏幕坐标
            viewportStartX: this.viewport.x,
            viewportStartY: this.viewport.y,
          };
          // 通知拖拽开始
          this.onDragStateChange?.(true);
        }
        break;
    }
  }

  /**
   * 拖拽节点状态下处理指针输入
   * data.x/y 是屏幕坐标，需要转换为世界坐标
   */
  private handlePointerInDraggingNode(data: import('./input').PointerInput): void {
    if (this.state.kind !== 'dragging-node') return;

    if (data.type === 'move') {
      // 转换为世界坐标
      const worldPos = this.screenToCanvas(data.x, data.y);
      
      // 计算世界坐标位移
      const deltaX = worldPos.x - this.state.startX;
      const deltaY = worldPos.y - this.state.startY;
      
      // 更新所有选中节点的位置
      for (const [nodeId, startPos] of this.state.nodeStartPositions) {
        this.onNodePositionChange?.(
          nodeId,
          startPos.x + deltaX,
          startPos.y + deltaY
        );
      }
      
      this.requestRender();
    } else if (data.type === 'up') {
      // 拖拽结束
      this.state = { kind: 'idle' };
      // 通知拖拽结束
      this.onDragStateChange?.(false);
    }
  }

  /**
   * 拖拽视口状态下处理指针输入
   * data.x/y 是屏幕坐标，直接使用
   */
  private handlePointerInDraggingViewport(data: import('./input').PointerInput): void {
    if (this.state.kind !== 'dragging-viewport') return;

    if (data.type === 'move') {
      // 直接用屏幕坐标计算位移，这是关键！
      const deltaX = data.x - this.state.startX;
      const deltaY = data.y - this.state.startY;
      
      this.viewport.x = this.state.viewportStartX + deltaX;
      this.viewport.y = this.state.viewportStartY + deltaY;
      // 实时通知视口变化，让 MiniMap 能够实时更新
      this.onViewportChange?.(this.viewport);
      this.requestRender();
    } else if (data.type === 'up') {
      // 拖拽结束
      this.state = { kind: 'idle' };
      // 通知拖拽结束
      this.onDragStateChange?.(false);
    }
  }

  /**
   * 创建连接状态下处理指针输入
   * data.x/y 是屏幕坐标，需要转换为世界坐标
   */
  private handlePointerInConnecting(data: import('./input').PointerInput): void {
    if (this.state.kind !== 'connecting') return;

    if (data.type === 'move') {
      // 转换为世界坐标
      const worldPos = this.screenToCanvas(data.x, data.y);
      
      // 更新连接预览线终点
      this.state = {
        ...this.state,
        currentX: worldPos.x,
        currentY: worldPos.y,
      };
      this.requestRender();
    } else if (data.type === 'up') {
      // 转换为世界坐标用于命中测试
      const worldPos = this.screenToCanvas(data.x, data.y);
      const hit = this.hitTest(worldPos.x, worldPos.y);
      
      if (hit.kind === 'handle' && hit.nodeId !== this.state.sourceNodeId) {
        // 尝试创建连接
        this.onConnectionAttempt?.(
          this.state.sourceNodeId,
          this.state.sourceHandleId,
          hit.nodeId,
          hit.handleId
        );
      }
      
      this.state = { kind: 'idle' };
      this.requestRender();
    }
  }

  /**
   * 框选状态下处理指针输入
   * data.x/y 是屏幕坐标，需要转换为世界坐标
   */
  private handlePointerInBoxSelecting(data: import('./input').PointerInput): void {
    if (this.state.kind !== 'box-selecting') return;

    if (data.type === 'move') {
      // 转换为世界坐标
      const worldPos = this.screenToCanvas(data.x, data.y);
      
      // 更新选择框
      this.state = {
        ...this.state,
        currentX: worldPos.x,
        currentY: worldPos.y,
      };
      this.requestRender();
    } else if (data.type === 'up') {
      // 计算选择框范围（世界坐标）
      const minX = Math.min(this.state.startX, this.state.currentX);
      const maxX = Math.max(this.state.startX, this.state.currentX);
      const minY = Math.min(this.state.startY, this.state.currentY);
      const maxY = Math.max(this.state.startY, this.state.currentY);
      
      // 选择框内的节点（使用 LayoutBox 尺寸）
      const graphData = this.getGraphData();
      if (graphData) {
        this.selectedNodeIds.clear();
        for (const node of graphData.nodes) {
          const layoutBox = this.getLayoutBox(node);
          // 检查节点是否与选择框相交
          if (layoutBox.x < maxX && layoutBox.x + layoutBox.width > minX &&
              layoutBox.y < maxY && layoutBox.y + layoutBox.height > minY) {
            this.selectedNodeIds.add(node.id);
          }
        }
        // 通知选择变化
        this.notifySelectionChange();
      }
      
      this.state = { kind: 'idle' };
      this.requestRender();
    }
  }

  /**
   * 处理滚轮输入
   * data.x/y 是屏幕坐标
   */
  private handleWheelInput(data: import('./input').WheelInput): void {
    // UI 事件预处理
    if (this.onPreWheel?.(data.x, data.y, data.deltaX, data.deltaY)) {
      return; // UI 组件已处理
    }

    // 通知缩放开始
    this.onZoomStateChange?.(true);
    
    // 滚轮缩放，以鼠标位置为中心（屏幕坐标）
    const factor = data.deltaY > 0 ? 0.9 : 1.1;
    this.zoomViewportAtScreen(factor, data.x, data.y);
    
    // 延迟通知缩放结束（避免频繁触发）
    this.scheduleZoomEnd();
  }

  // 缩放结束定时器
  private zoomEndTimer: number | null = null;

  private scheduleZoomEnd(): void {
    if (this.zoomEndTimer !== null) {
      clearTimeout(this.zoomEndTimer);
    }
    this.zoomEndTimer = window.setTimeout(() => {
      this.onZoomStateChange?.(false);
      this.zoomEndTimer = null;
    }, 150);
  }

  /**
   * 处理键盘输入
   */
  private handleKeyInput(data: import('./input').KeyInput): void {
    // UI 事件预处理
    if (data.type === 'down' && this.onPreKeyDown?.(data.key, data.code, data.modifiers.ctrl, data.modifiers.shift, data.modifiers.alt)) {
      return; // UI 组件已处理
    }

    if (data.type !== 'down') return;

    switch (data.key) {
      case 'Delete':
      case 'Backspace':
        // 删除选中元素
        if (this.selectedNodeIds.size > 0 || this.selectedEdgeIds.size > 0) {
          this.onDeleteSelected?.();
        }
        break;

      case 'Escape':
        // 取消当前操作
        if (this.state.kind !== 'idle') {
          this.state = { kind: 'idle' };
          this.requestRender();
        } else {
          // 清除选择
          this.clearSelection();
        }
        break;

      case 'a':
        // Ctrl+A 全选
        if (data.modifiers.ctrl || data.modifiers.meta) {
          const graphData = this.getGraphData();
          if (graphData) {
            this.selectedNodeIds.clear();
            for (const node of graphData.nodes) {
              this.selectedNodeIds.add(node.id);
            }
            this.requestRender();
          }
        }
        break;
    }
  }

  // ============================================================
  // 选择管理
  // ============================================================

  /**
   * 获取选中的节点 ID
   */
  getSelectedNodeIds(): string[] {
    return Array.from(this.selectedNodeIds);
  }

  /**
   * 获取选中的边 ID
   */
  getSelectedEdgeIds(): string[] {
    return Array.from(this.selectedEdgeIds);
  }

  /**
   * 选择节点
   */
  selectNode(nodeId: string, addToSelection: boolean = false): void {
    if (!addToSelection) {
      this.selectedNodeIds.clear();
      this.selectedEdgeIds.clear();
    }
    this.selectedNodeIds.add(nodeId);
    this.notifySelectionChange();
    this.requestRender();
  }

  /**
   * 取消选择节点
   */
  deselectNode(nodeId: string): void {
    this.selectedNodeIds.delete(nodeId);
    this.notifySelectionChange();
    this.requestRender();
  }

  /**
   * 清除所有选择
   */
  clearSelection(): void {
    this.selectedNodeIds.clear();
    this.selectedEdgeIds.clear();
    this.notifySelectionChange();
    this.requestRender();
  }

  /**
   * 通知选择变化
   */
  private notifySelectionChange(): void {
    this.onSelectionChange?.(
      Array.from(this.selectedNodeIds),
      Array.from(this.selectedEdgeIds)
    );
  }

  /**
   * 从外部同步选择状态（不触发回调，避免循环）
   * 用于 React Flow nodes.selected 变化时同步到 Canvas
   */
  syncSelectionFromExternal(nodeIds: string[]): void {
    const newSet = new Set(nodeIds);
    // 检查是否有变化
    if (this.selectedNodeIds.size === newSet.size &&
        [...this.selectedNodeIds].every(id => newSet.has(id))) {
      return; // 无变化
    }
    this.selectedNodeIds = newSet;
    this.requestRender();
  }

  // ============================================================
  // 命中测试
  // ============================================================

  /**
   * 命中测试
   * 统一使用 LayoutBox 系统，确保命中区域与渲染区域一致
   * @param x - 画布坐标 X
   * @param y - 画布坐标 Y
   * @returns 命中结果
   */
  hitTest(x: number, y: number): HitResult {
    const graphData = this.getGraphData();
    if (!graphData) {
      return { kind: 'none' };
    }

    // 按 zIndex 从高到低排序节点（选中的节点优先）
    const sortedNodes = [...graphData.nodes].sort((a, b) => {
      const aSelected = this.selectedNodeIds.has(a.id) ? 1 : 0;
      const bSelected = this.selectedNodeIds.has(b.id) ? 1 : 0;
      return bSelected - aSelected;
    });

    // 1. 首先检查端口（优先级最高）
    for (const node of sortedNodes) {
      const layoutBox = this.getLayoutBox(node);
      const handles = extractHandlePositions(layoutBox);
      
      for (const handle of handles) {
        const hitRadius = NODE_LAYOUT.HANDLE_RADIUS + 4;
        if (isPointInCircle(x, y, handle.x, handle.y, hitRadius)) {
          return {
            kind: 'handle',
            nodeId: node.id,
            handleId: handle.handleId,
            isOutput: handle.isOutput,
          };
        }
      }
    }

    // 2. 然后检查节点（使用 LayoutBox 尺寸）
    for (const node of sortedNodes) {
      const layoutBox = this.getLayoutBox(node);
      
      if (isPointInRect(x, y, layoutBox.x, layoutBox.y, layoutBox.width, layoutBox.height)) {
        return {
          kind: 'node',
          nodeId: node.id,
        };
      }
    }

    // 3. 最后检查边（使用 LayoutBox 的 Handle 位置）
    for (const edge of graphData.edges) {
      const sourceNode = graphData.nodes.find(n => n.id === edge.source);
      const targetNode = graphData.nodes.find(n => n.id === edge.target);
      
      if (!sourceNode || !targetNode) continue;

      const sourceBox = this.getLayoutBox(sourceNode);
      const targetBox = this.getLayoutBox(targetNode);
      const sourceHandles = extractHandlePositions(sourceBox);
      const targetHandles = extractHandlePositions(targetBox);

      const sourceHandle = sourceHandles.find(h => h.handleId === edge.sourceHandle);
      const targetHandle = targetHandles.find(h => h.handleId === edge.targetHandle);

      const sourceX = sourceHandle?.x ?? sourceBox.x + sourceBox.width;
      const sourceY = sourceHandle?.y ?? sourceBox.y + sourceBox.height / 2;
      const targetX = targetHandle?.x ?? targetBox.x;
      const targetY = targetHandle?.y ?? targetBox.y + targetBox.height / 2;

      const points = computeEdgePath(sourceX, sourceY, targetX, targetY);
      const distance = distanceToEdge(x, y, points);
      
      if (distance < 8) { // 8 像素容差
        const edgeId = `${edge.source}-${edge.sourceHandle}-${edge.target}-${edge.targetHandle}`;
        return {
          kind: 'edge',
          edgeId,
        };
      }
    }

    return { kind: 'none' };
  }

  // ============================================================
  // 图数据源
  // ============================================================

  /** 图数据获取函数（由外部设置） */
  private graphDataProvider: (() => GraphState | null) | null = null;

  /**
   * 设置图数据提供者
   * GraphController 不直接依赖 store，而是通过回调获取数据
   */
  setGraphDataProvider(provider: () => GraphState | null): void {
    this.graphDataProvider = provider;
  }

  /**
   * 获取当前图数据
   */
  private getGraphData(): GraphState | null {
    return this.graphDataProvider?.() ?? null;
  }

  // ============================================================
  // 节点布局 - 统一使用 LayoutBox 系统
  // ============================================================

  /**
   * 获取节点的 LayoutBox
   * 优先从缓存获取，否则实时计算
   */
  private getLayoutBox(node: GraphNode): LayoutBox {
    // 优先从缓存获取
    const cached = this.cachedRenderData?.layoutBoxes?.get(node.id);
    if (cached) {
      return cached;
    }
    // 实时计算
    return computeNodeLayoutBox(node, node.position.x, node.position.y);
  }

  /**
   * 清除布局缓存（保留接口兼容性）
   */
  clearLayoutCache(): void {
    // 不再使用缓存，此方法保留为空
  }

  // ============================================================
  // 渲染数据计算
  // ============================================================

  /**
   * 计算渲染数据
   * @returns 完整的渲染数据
   */
  computeRenderData(): RenderData {
    const graphData = this.getGraphData();
    if (!graphData) {
      return createEmptyRenderData();
    }

    // 从 Design Tokens 获取样式，确保与 ReactFlow/VueFlow 一致

    const rects: RenderRect[] = [];
    const texts: RenderText[] = [];
    const paths: RenderPath[] = [];
    const circles: RenderCircle[] = [];
    const triangles: RenderTriangle[] = [];
    const overlays: OverlayInfo[] = [];

    // 首先计算所有节点的 LayoutBox（统一布局系统）
    const layoutBoxes = new Map<string, LayoutBox>();
    const layoutBoxHandles = new Map<string, HandlePosition[]>();
    for (const node of graphData.nodes) {
      try {
        const layoutBox = computeNodeLayoutBox(node, node.position.x, node.position.y);
        layoutBoxes.set(node.id, layoutBox);
        // 提取 Handle 位置
        const handles = extractHandlePositions(layoutBox);
        layoutBoxHandles.set(node.id, handles);
      } catch (e) {
        console.warn(`Failed to compute LayoutBox for node ${node.id}:`, e);
      }
    }

    // 生成节点相关的渲染数据（基于 LayoutBox）
    for (const node of graphData.nodes) {
      const layoutBox = layoutBoxes.get(node.id);
      if (!layoutBox) continue;
      
      const selected = this.selectedNodeIds.has(node.id);
      const zIndex = selected ? 100 : 0;
      
      // 生成节点矩形（仅用于传递 selected/zIndex 信息给渲染器）
      rects.push({
        id: `rect-${node.id}`,
        x: layoutBox.x,
        y: layoutBox.y,
        width: layoutBox.width,
        height: layoutBox.height,
        fillColor: 'transparent', // 不渲染背景，由 LayoutBox 系统负责
        borderColor: 'transparent',
        borderWidth: 0,
        borderRadius: NODE_LAYOUT.BORDER_RADIUS,
        selected,
        zIndex,
      });

      // 生成覆盖层信息（选中节点）
      if (selected) {
        const screenPos = this.canvasToScreen(layoutBox.x, layoutBox.y);
        overlays.push({
          nodeId: node.id,
          screenX: screenPos.x,
          screenY: screenPos.y,
          width: layoutBox.width * this.viewport.zoom,
          height: layoutBox.height * this.viewport.zoom,
        });
      }
    }

    // 计算所有边（完全使用 LayoutBox 系统）
    for (const edge of graphData.edges) {
      const sourceHandles = layoutBoxHandles.get(edge.source);
      const targetHandles = layoutBoxHandles.get(edge.target);
      const sourceBox = layoutBoxes.get(edge.source);
      const targetBox = layoutBoxes.get(edge.target);
      
      if (!sourceBox || !targetBox) continue;
      
      // 从 LayoutBox 获取 Handle 位置
      const sourceHandle = sourceHandles?.find(h => h.handleId === edge.sourceHandle);
      const targetHandle = targetHandles?.find(h => h.handleId === edge.targetHandle);
      
      const sourceX = sourceHandle?.x ?? sourceBox.x + sourceBox.width;
      const sourceY = sourceHandle?.y ?? sourceBox.y + sourceBox.height / 2;
      const targetX = targetHandle?.x ?? targetBox.x;
      const targetY = targetHandle?.y ?? targetBox.y + targetBox.height / 2;
      
      // 判断是否为执行流边
      const isExec = edge.sourceHandle?.includes('exec') || edge.targetHandle?.includes('exec') || false;
      
      // 边颜色：执行流用白色，数据流优先从 edge.data.color 获取，其次从源 Handle 的 pinColor 获取
      const edgeDataColor = edge.data?.color;
      const edgeColor = isExec 
        ? EDGE_LAYOUT.EXEC_COLOR 
        : edgeDataColor ?? sourceHandle?.pinColor ?? EDGE_LAYOUT.DEFAULT_DATA_COLOR;

      // 计算贝塞尔曲线路径
      const points = computeEdgePath(sourceX, sourceY, targetX, targetY);

      const edgeId = `${edge.source}-${edge.sourceHandle}-${edge.target}-${edge.targetHandle}`;
      const selected = this.selectedEdgeIds.has(edgeId);

      paths.push({
        id: `edge-${edgeId}`,
        points,
        color: edgeColor,
        width: selected ? EDGE_LAYOUT.SELECTED_WIDTH : EDGE_LAYOUT.WIDTH,
        dashed: false,
        animated: false,
        arrowEnd: false,
      });
    }

    // 计算交互提示
    const hint = this.computeInteractionHint();

    const renderData: RenderData = {
      viewport: { ...this.viewport },
      rects,
      texts,
      paths,
      circles,
      triangles,
      hint,
      overlays,
      layoutBoxes,
    };

    // 调用扩展回调已废弃，渲染扩展已移至 LayoutBox 系统

    return renderData;
  }

  /**
   * 计算交互提示
   */
  private computeInteractionHint(): InteractionHint {
    const hint = createDefaultHint();
    
    switch (this.state.kind) {
      case 'idle':
        hint.cursor = 'default';
        break;
        
      case 'dragging-node':
        hint.cursor = 'grabbing';
        break;
        
      case 'dragging-viewport':
        hint.cursor = 'grabbing';
        break;
        
      case 'connecting': {
        hint.cursor = 'crosshair';
        // 添加连接预览线（使用 LayoutBox 系统）
        const connectingState = this.state;
        const graphData = this.getGraphData();
        const sourceNode = graphData?.nodes.find(n => n.id === connectingState.sourceNodeId);
        if (sourceNode) {
          const sourceBox = this.getLayoutBox(sourceNode);
          const sourceHandles = extractHandlePositions(sourceBox);
          const sourceHandle = sourceHandles.find(h => h.handleId === connectingState.sourceHandleId);
          if (sourceHandle) {
            const points = computeEdgePath(sourceHandle.x, sourceHandle.y, connectingState.currentX, connectingState.currentY);
            // 判断是否为执行流
            const isExec = connectingState.sourceHandleId.includes('exec');
            // 连接预览颜色：执行流用白色，数据流用源 Handle 的 pinColor
            const previewColor = isExec 
              ? EDGE_LAYOUT.EXEC_COLOR 
              : sourceHandle.pinColor ?? EDGE_LAYOUT.DEFAULT_DATA_COLOR;
            hint.connectionPreview = {
              id: 'connection-preview',
              points,
              color: previewColor,
              width: 2,
              dashed: true,
              dashPattern: [5, 5],
              animated: true,
              arrowEnd: false,
            };
          }
        }
        break;
      }
        
      case 'box-selecting': {
        hint.cursor = 'crosshair';
        // 添加选择框
        const minX = Math.min(this.state.startX, this.state.currentX);
        const minY = Math.min(this.state.startY, this.state.currentY);
        const width = Math.abs(this.state.currentX - this.state.startX);
        const height = Math.abs(this.state.currentY - this.state.startY);
        hint.selectionBox = {
          id: 'selection-box',
          x: minX,
          y: minY,
          width,
          height,
          fillColor: 'rgba(59, 130, 246, 0.1)',
          borderColor: 'rgba(59, 130, 246, 0.5)',
          borderWidth: 1,
          borderRadius: 0,
          selected: false,
          zIndex: 1000,
        };
        break;
      }
    }
    
    return hint;
  }

  // ============================================================
  // 渲染控制
  // ============================================================

  /**
   * 请求渲染
   * 使用 requestAnimationFrame 合并多次请求
   */
  requestRender(): void {
    if (this.renderRequested) return;
    
    this.renderRequested = true;
    requestAnimationFrame(() => {
      this.renderRequested = false;
      this.render();
    });
  }

  /**
   * 立即渲染
   */
  private render(): void {
    if (!this.renderer) return;
    
    const startTime = performance.now();
    
    this.cachedRenderData = this.computeRenderData();
    this.renderer.render(this.cachedRenderData);
    
    const endTime = performance.now();
    performanceMonitor.recordFrame(endTime - startTime);
  }

  /**
   * 获取缓存的渲染数据
   */
  getCachedRenderData(): RenderData {
    return this.cachedRenderData;
  }
}
