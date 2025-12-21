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
  computeNodeLayout,
  computeEdgePath,
  distanceToEdge,
  isPointInRect,
  isPointInCircle,
  NODE_LAYOUT,
  EDGE_LAYOUT,
  type NodeLayout,
} from './layout';

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

  /** 渲染数据扩展回调（用于添加额外的渲染元素，如类型标签、按钮等） */
  onExtendRenderData: ((data: RenderData, nodeLayouts: Map<string, NodeLayout>) => void) | null = null;

  /** 连接尝试回调 */
  onConnectionAttempt: ((
    sourceNodeId: string,
    sourceHandleId: string,
    targetNodeId: string,
    targetHandleId: string
  ) => void) | null = null;

  /** 删除选中元素回调 */
  onDeleteSelected: (() => void) | null = null;

  /** 选择变化回调 */
  onSelectionChange: ((nodeIds: string[], edgeIds: string[]) => void) | null = null;

  /** 视口变化回调 */
  onViewportChange: ((viewport: Viewport) => void) | null = null;

  /** 双击边删除回调 */
  onEdgeDoubleClick: ((edgeId: string) => void) | null = null;

  /** 拖放回调 */
  onDrop: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null = null;

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
    // 如果有旧后端，先卸载
    if (this.renderer) {
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
    if (data.type !== 'down') return;

    // 转换为世界坐标用于命中测试
    const worldPos = this.screenToCanvas(data.x, data.y);
    const hit = this.hitTest(worldPos.x, worldPos.y);

    // 双击检测
    const now = Date.now();
    const isDoubleClick = now - this.lastClickTime < 300 && 
      this.lastClickTarget.kind === hit.kind &&
      (hit.kind === 'edge' && this.lastClickTarget.kind === 'edge' && hit.edgeId === this.lastClickTarget.edgeId);
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
          
          // 收集所有选中节点的初始位置
          const nodeStartPositions = new Map<string, { x: number; y: number }>();
          const graphData = this.getGraphData();
          if (graphData) {
            for (const nodeId of this.selectedNodeIds) {
              const node = graphData.nodes.find(n => n.id === nodeId);
              if (node) {
                const layout = this.getNodeLayout(node);
                nodeStartPositions.set(nodeId, { x: layout.x, y: layout.y });
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
        // 清除布局缓存以便重新计算
        this.nodeLayoutCache.delete(nodeId);
      }
      
      this.requestRender();
    } else if (data.type === 'up') {
      // 拖拽结束
      this.state = { kind: 'idle' };
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
      this.requestRender();
    } else if (data.type === 'up') {
      // 拖拽结束，通知视口变化
      this.onViewportChange?.(this.viewport);
      this.state = { kind: 'idle' };
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
      
      // 选择框内的节点
      const graphData = this.getGraphData();
      if (graphData) {
        this.selectedNodeIds.clear();
        for (const node of graphData.nodes) {
          const layout = this.getNodeLayout(node);
          // 检查节点是否与选择框相交
          if (layout.x < maxX && layout.x + layout.width > minX &&
              layout.y < maxY && layout.y + layout.height > minY) {
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
    // 滚轮缩放，以鼠标位置为中心（屏幕坐标）
    const factor = data.deltaY > 0 ? 0.9 : 1.1;
    this.zoomViewportAtScreen(factor, data.x, data.y);
  }

  /**
   * 处理键盘输入
   */
  private handleKeyInput(data: import('./input').KeyInput): void {
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
      const layout = this.getNodeLayout(node);
      
      for (const handle of layout.handles) {
        const handleX = layout.x + handle.x;
        const handleY = layout.y + handle.y;
        const hitRadius = NODE_LAYOUT.HANDLE_RADIUS + 4; // 增加点击容差
        
        if (isPointInCircle(x, y, handleX, handleY, hitRadius)) {
          return {
            kind: 'handle',
            nodeId: node.id,
            handleId: handle.handleId,
            isOutput: handle.isOutput,
          };
        }
      }
    }

    // 2. 然后检查节点
    for (const node of sortedNodes) {
      const layout = this.getNodeLayout(node);
      
      if (isPointInRect(x, y, layout.x, layout.y, layout.width, layout.height)) {
        return {
          kind: 'node',
          nodeId: node.id,
        };
      }
    }

    // 3. 最后检查边
    for (const edge of graphData.edges) {
      const sourceLayout = this.nodeLayoutCache.get(edge.source);
      const targetLayout = this.nodeLayoutCache.get(edge.target);
      
      if (!sourceLayout || !targetLayout) continue;

      const sourceHandle = sourceLayout.handles.find(h => h.handleId === edge.sourceHandle);
      const targetHandle = targetLayout.handles.find(h => h.handleId === edge.targetHandle);

      const sourceX = sourceLayout.x + (sourceHandle?.x ?? sourceLayout.width);
      const sourceY = sourceLayout.y + (sourceHandle?.y ?? sourceLayout.height / 2);
      const targetX = targetLayout.x + (targetHandle?.x ?? 0);
      const targetY = targetLayout.y + (targetHandle?.y ?? targetLayout.height / 2);

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
  // 节点布局缓存
  // ============================================================

  /** 节点布局缓存 */
  private nodeLayoutCache: Map<string, NodeLayout> = new Map();

  /**
   * 获取或计算节点布局
   */
  private getNodeLayout(node: GraphNode): NodeLayout {
    const cached = this.nodeLayoutCache.get(node.id);
    if (cached && cached.x === node.position.x && cached.y === node.position.y) {
      // 位置未变，使用缓存
      return { ...cached, selected: this.selectedNodeIds.has(node.id) };
    }
    
    // 重新计算
    const layout = computeNodeLayout(node, this.selectedNodeIds.has(node.id));
    this.nodeLayoutCache.set(node.id, layout);
    return layout;
  }

  /**
   * 清除布局缓存
   */
  clearLayoutCache(): void {
    this.nodeLayoutCache.clear();
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

    const rects: RenderRect[] = [];
    const texts: RenderText[] = [];
    const paths: RenderPath[] = [];
    const circles: RenderCircle[] = [];
    const triangles: RenderTriangle[] = [];
    const overlays: OverlayInfo[] = [];

    // 计算所有节点布局
    const nodeLayouts = new Map<string, NodeLayout>();
    for (const node of graphData.nodes) {
      const layout = this.getNodeLayout(node);
      nodeLayouts.set(node.id, layout);
      
      // 生成节点矩形
      rects.push({
        id: `rect-${node.id}`,
        x: layout.x,
        y: layout.y,
        width: layout.width,
        height: layout.height,
        fillColor: layout.backgroundColor,
        borderColor: layout.selected ? NODE_LAYOUT.SELECTED_BORDER_COLOR : NODE_LAYOUT.DEFAULT_BORDER_COLOR,
        borderWidth: layout.selected ? 2 : 1,
        borderRadius: NODE_LAYOUT.BORDER_RADIUS,
        selected: layout.selected,
        zIndex: layout.zIndex,
      });

      // 生成节点头部矩形（只有上方圆角）
      rects.push({
        id: `header-${node.id}`,
        x: layout.x,
        y: layout.y,
        width: layout.width,
        height: layout.headerHeight,
        fillColor: layout.headerColor,
        borderColor: 'transparent',
        borderWidth: 0,
        borderRadius: {
          topLeft: NODE_LAYOUT.BORDER_RADIUS,
          topRight: NODE_LAYOUT.BORDER_RADIUS,
          bottomLeft: 0,
          bottomRight: 0,
        },
        selected: false,
        zIndex: layout.zIndex + 1,
      });

      // 生成节点标题文字 - 与 React Flow 一致
      // Operation/FunctionCall: subtitle(uppercase) + title
      // Entry/Return: title + subtitle (不 uppercase)
      if (layout.subtitle) {
        const isEntryOrReturn = node.type === 'function-entry' || node.type === 'function-return';
        
        if (isEntryOrReturn) {
          // Entry/Return: title 在前，subtitle (main) 在后，不 uppercase
          texts.push({
            id: `title-${node.id}`,
            text: layout.title,
            x: layout.x + NODE_LAYOUT.PADDING,
            y: layout.y + layout.headerHeight / 2,
            fontSize: 14,
            fontFamily: 'system-ui, sans-serif',
            color: '#ffffff',
            align: 'left',
            baseline: 'middle',
          });
          // 计算 title 宽度（近似），subtitle 紧跟其后
          const titleWidth = layout.title.length * 8; // 近似每字符 8px (14px font)
          texts.push({
            id: `subtitle-${node.id}`,
            text: layout.subtitle, // 不 uppercase
            x: layout.x + NODE_LAYOUT.PADDING + titleWidth + 4,
            y: layout.y + layout.headerHeight / 2,
            fontSize: 12,
            fontFamily: 'system-ui, sans-serif',
            color: 'rgba(255,255,255,0.7)',
            align: 'left',
            baseline: 'middle',
          });
        } else {
          // Operation/FunctionCall: subtitle(uppercase) 在前，title 在后
          texts.push({
            id: `subtitle-${node.id}`,
            text: layout.subtitle.toUpperCase(),
            x: layout.x + NODE_LAYOUT.PADDING,
            y: layout.y + layout.headerHeight / 2,
            fontSize: 12,
            fontFamily: 'system-ui, sans-serif',
            color: 'rgba(255,255,255,0.7)',
            align: 'left',
            baseline: 'middle',
          });
          // 计算 subtitle 宽度（近似），title 紧跟其后
          const subtitleWidth = layout.subtitle.length * 8; // 近似每字符 8px (12px font)
          texts.push({
            id: `title-${node.id}`,
            text: layout.title,
            x: layout.x + NODE_LAYOUT.PADDING + subtitleWidth + 4,
            y: layout.y + layout.headerHeight / 2,
            fontSize: 14,
            fontFamily: 'system-ui, sans-serif',
            color: '#ffffff',
            align: 'left',
            baseline: 'middle',
          });
        }
      } else {
        // 无副标题时：只有标题
        texts.push({
          id: `title-${node.id}`,
          text: layout.title,
          x: layout.x + NODE_LAYOUT.PADDING,
          y: layout.y + layout.headerHeight / 2,
          fontSize: 14,
          fontFamily: 'system-ui, sans-serif',
          color: '#ffffff',
          align: 'left',
          baseline: 'middle',
        });
      }

      // 生成端口圆形/三角形和标签
      for (const handle of layout.handles) {
        if (handle.kind === 'exec') {
          // 执行引脚使用三角形，统一向右
          triangles.push({
            id: `handle-${node.id}-${handle.handleId}`,
            x: layout.x + handle.x,
            y: layout.y + handle.y,
            size: NODE_LAYOUT.HANDLE_RADIUS * 1.5,
            fillColor: '#ffffff',
            borderColor: '#ffffff',
            borderWidth: 0,
            direction: 'right',
          });
        } else {
          // 数据引脚使用圆形
          circles.push({
            id: `handle-${node.id}-${handle.handleId}`,
            x: layout.x + handle.x,
            y: layout.y + handle.y,
            radius: NODE_LAYOUT.HANDLE_RADIUS,
            fillColor: handle.color,
            borderColor: handle.color,
            borderWidth: 1,
          });
        }
        
        // 端口标签
        if (handle.label) {
          const labelX = handle.isOutput 
            ? layout.x + handle.x - NODE_LAYOUT.HANDLE_RADIUS - 4
            : layout.x + handle.x + NODE_LAYOUT.HANDLE_RADIUS + 4;
          texts.push({
            id: `handle-label-${node.id}-${handle.handleId}`,
            text: handle.label,
            x: labelX,
            y: layout.y + handle.y,
            fontSize: 10,
            fontFamily: 'system-ui, sans-serif',
            color: '#cccccc',
            align: handle.isOutput ? 'right' : 'left',
            baseline: 'middle',
          });
        }
      }

      // 生成覆盖层信息（选中节点）
      if (layout.selected) {
        const screenPos = this.canvasToScreen(layout.x, layout.y);
        overlays.push({
          nodeId: node.id,
          screenX: screenPos.x,
          screenY: screenPos.y,
          width: layout.width * this.viewport.zoom,
          height: layout.height * this.viewport.zoom,
        });
      }
    }

    // 计算所有边
    for (const edge of graphData.edges) {
      const sourceLayout = nodeLayouts.get(edge.source);
      const targetLayout = nodeLayouts.get(edge.target);
      
      if (!sourceLayout || !targetLayout) continue;

      // 查找源端口和目标端口
      const sourceHandle = sourceLayout.handles.find(h => h.handleId === edge.sourceHandle);
      const targetHandle = targetLayout.handles.find(h => h.handleId === edge.targetHandle);

      // 计算端口绝对坐标
      const sourceX = sourceLayout.x + (sourceHandle?.x ?? sourceLayout.width);
      const sourceY = sourceLayout.y + (sourceHandle?.y ?? sourceLayout.height / 2);
      const targetX = targetLayout.x + (targetHandle?.x ?? 0);
      const targetY = targetLayout.y + (targetHandle?.y ?? targetLayout.height / 2);

      // 计算贝塞尔曲线路径
      const points = computeEdgePath(sourceX, sourceY, targetX, targetY);

      // 判断是否为执行流边
      const isExec = sourceHandle?.kind === 'exec' || targetHandle?.kind === 'exec';
      const edgeId = `${edge.source}-${edge.sourceHandle}-${edge.target}-${edge.targetHandle}`;
      const selected = this.selectedEdgeIds.has(edgeId);

      paths.push({
        id: `edge-${edgeId}`,
        points,
        color: isExec ? EDGE_LAYOUT.EXEC_COLOR : (sourceHandle?.color ?? EDGE_LAYOUT.DEFAULT_DATA_COLOR),
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
    };

    // 调用扩展回调，允许外部添加额外的渲染元素
    this.onExtendRenderData?.(renderData, nodeLayouts);

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
        // 添加连接预览线
        const connectingState = this.state;
        const sourceLayout = this.nodeLayoutCache.get(connectingState.sourceNodeId);
        if (sourceLayout) {
          const sourceHandle = sourceLayout.handles.find(h => h.handleId === connectingState.sourceHandleId);
          if (sourceHandle) {
            const sourceX = sourceLayout.x + sourceHandle.x;
            const sourceY = sourceLayout.y + sourceHandle.y;
            const points = computeEdgePath(sourceX, sourceY, connectingState.currentX, connectingState.currentY);
            hint.connectionPreview = {
              id: 'connection-preview',
              points,
              color: sourceHandle.color,
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
    
    this.cachedRenderData = this.computeRenderData();
    this.renderer.render(this.cachedRenderData);
  }

  /**
   * 获取缓存的渲染数据
   */
  getCachedRenderData(): RenderData {
    return this.cachedRenderData;
  }
}
