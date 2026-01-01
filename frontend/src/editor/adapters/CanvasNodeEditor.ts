/**
 * Canvas 节点编辑器
 * 
 * 实现 INodeEditor 接口，封装 GraphController 和渲染器。
 * 将旧架构的图元级别 API 适配为 Node Editor 语义级别。
 * 
 * 支持多种渲染后端：
 * - CanvasRenderer: Canvas 2D 渲染
 * - GPURenderer: WebGL/WebGPU 渲染
 * 
 * 覆盖层支持：
 * - 使用 OverlayManager 管理 HTML 覆盖层
 * - 支持类型选择器、属性编辑器等 UI 组件
 */

import type { INodeEditor } from '../INodeEditor';
import type {
  EditorNode,
  EditorEdge,
  EditorViewport,
  EditorSelection,
  ConnectionRequest,
  NodeChange,
  EdgeChange,
} from '../types';
import type { FunctionTrait } from '../../types';
import type { IRenderer } from '../core/IRenderer';
import type { Viewport } from './canvas/types';
import { CanvasRenderer } from './canvas/CanvasRenderer';
import { GraphController } from './canvas/GraphController';
import { UIManager, type TypeSelectorState, type EditableNameState, type AttributeEditorState } from './canvas/UIManager';
import type { TypeOption, ConstraintData } from './canvas/ui/TypeSelector';
import type { AttributeDefinition } from './canvas/ui/AttributeEditor';
import type { GraphState, GraphNode, FunctionEntryData, FunctionReturnData } from '../../types';
// 新布局系统
import { 
  hitTestLayoutBox, 
  parseInteractiveId, 
} from '../core/layout';

/**
 * 扩展渲染器接口
 * 
 * 在 IRenderer 基础上添加 Canvas 系编辑器需要的额外方法
 */
export interface IExtendedRenderer extends IRenderer {
  setDropCallback(callback: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null): void;
  setUIRenderCallback(callback: ((ctx: CanvasRenderingContext2D) => void) | null): void;
  getViewport(): Viewport;
  waitForReady?(): Promise<void>;
  isReady?(): boolean;
}

/**
 * 将 EditorNode/EditorEdge 转换为 GraphState
 */
function toGraphState(nodes: EditorNode[], edges: EditorEdge[]): GraphState {
  return {
    nodes: nodes.map(n => ({
      id: n.id,
      type: n.type as GraphNode['type'],
      position: { x: n.position.x, y: n.position.y },
      data: n.data,
    })) as GraphNode[],
    edges: edges.map(e => ({
      source: e.source,
      sourceHandle: e.sourceHandle,
      target: e.target,
      targetHandle: e.targetHandle,
      data: e.data as { color?: string; [key: string]: unknown } | undefined,
    })),
  };
}

/**
 * Canvas 节点编辑器
 * 
 * 实现 INodeEditor 接口，支持多种渲染后端。
 */
export class CanvasNodeEditor implements INodeEditor {
  private renderer: IExtendedRenderer | null = null;
  private rendererFactory: (() => IExtendedRenderer) | null = null;
  private controller: GraphController | null = null;
  private uiManager: UIManager | null = null;
  private container: HTMLElement | null = null;
  
  // 当前数据
  private nodes: EditorNode[] = [];
  private edges: EditorEdge[] = [];
  private selection: EditorSelection = { nodeIds: [], edgeIds: [] };
  
  // 回调
  onNodesChange: ((changes: NodeChange[]) => void) | null = null;
  onEdgesChange: ((changes: EdgeChange[]) => void) | null = null;
  onSelectionChange: ((selection: EditorSelection) => void) | null = null;
  onViewportChange: ((viewport: EditorViewport) => void) | null = null;
  onConnect: ((request: ConnectionRequest) => void) | null = null;
  onNodeDoubleClick: ((nodeId: string) => void) | null = null;
  onEdgeDoubleClick: ((edgeId: string) => void) | null = null;
  onDrop: ((x: number, y: number, dataTransfer: DataTransfer) => void) | null = null;
  onDeleteRequest: ((nodeIds: string[], edgeIds: string[]) => void) | null = null;
  
  // 业务事件回调
  onAttributeChange: ((nodeId: string, attributeName: string, value: string) => void) | null = null;
  onVariadicAdd: ((nodeId: string, groupName: string) => void) | null = null;
  onVariadicRemove: ((nodeId: string, groupName: string) => void) | null = null;
  onParameterAdd: ((functionId: string) => void) | null = null;
  onParameterRemove: ((functionId: string, parameterName: string) => void) | null = null;
  onParameterRename: ((functionId: string, oldName: string, newName: string) => void) | null = null;
  onReturnTypeAdd: ((functionId: string) => void) | null = null;
  onReturnTypeRemove: ((functionId: string, returnName: string) => void) | null = null;
  onReturnTypeRename: ((functionId: string, oldName: string, newName: string) => void) | null = null;
  onTraitsChange: ((functionId: string, traits: FunctionTrait[]) => void) | null = null;
  
  // 覆盖层回调
  onTypeSelectorChange: ((state: TypeSelectorState | null) => void) | null = null;
  onEditableNameChange: ((state: EditableNameState | null) => void) | null = null;
  onAttributeEditorChange: ((state: AttributeEditorState | null) => void) | null = null;
  onTypeLabelClick: ((nodeId: string, handleId: string, canvasX: number, canvasY: number) => void) | null = null;
  onParamNameClick: ((nodeId: string, paramIndex: number, currentName: string, canvasX: number, canvasY: number) => void) | null = null;
  onReturnNameClick: ((nodeId: string, returnIndex: number, currentName: string, canvasX: number, canvasY: number) => void) | null = null;
  onTraitsToggleClick: ((nodeId: string, canvasX: number, canvasY: number) => void) | null = null;
  
  // onNodeDataChange 已废弃：节点组件现在直接更新 editorStore
  onNodeDataChange: ((nodeId: string, data: Record<string, unknown>) => void) | null = null;
  
  // Hover 状态（用于显示删除按钮）
  private hoveredParamIndex: number | null = null;
  private hoveredReturnIndex: number | null = null;
  private hoveredNodeId: string | null = null;

  // Summary 展开状态
  private summaryExpandedMap: Map<string, boolean> = new Map();

  /**
   * 构造函数
   * @param rendererFactory 渲染器工厂函数，延迟到 mount 时创建渲染器
   */
  constructor(rendererFactory?: () => IExtendedRenderer) {
    this.rendererFactory = rendererFactory ?? (() => new CanvasRenderer() as IExtendedRenderer);
  }

  // ============================================================
  // 生命周期
  // ============================================================

  mount(container: HTMLElement): void {
    this.container = container;
    this.renderer = this.rendererFactory!();
    this.controller = new GraphController();
    this.uiManager = new UIManager();
    
    this.renderer.mount(container);
    this.uiManager.mount(container);
    this.controller.setRenderer(this.renderer);
    
    // 如果是 GPU 渲染器，等待就绪后再初始化
    const initAfterReady = () => {
      this.setupCallbacks();
      this.controller?.requestRender();
    };
    
    if (this.renderer.waitForReady) {
      this.renderer.waitForReady().then(initAfterReady);
    } else {
      initAfterReady();
    }
    
    // 监听窗口大小变化
    window.addEventListener('resize', this.handleResize);
  }
  
  /**
   * 设置所有回调
   */
  private setupCallbacks(): void {
    if (!this.renderer || !this.controller || !this.uiManager) return;
    
    // 设置 UIManager 回调
    this.uiManager.setCallbacks({
      onTypeSelect: () => {
        // 类型选择由外部通过 setTypeSelectCallback 处理
        // 这里只是占位，实际回调在 setTypeSelectCallback 中设置
      },
      onTypeSelectorClose: () => {
        this.onTypeSelectorChange?.(null);
        this.controller?.requestRender();
      },
    });
    
    // 设置图数据提供者
    this.controller.setGraphDataProvider(() => toGraphState(this.nodes, this.edges));
    
    // 渲染扩展已移至 LayoutBox 系统，不再需要 onExtendRenderData 回调
    
    // 设置 UI 渲染回调
    this.renderer.setUIRenderCallback((ctx) => {
      this.uiManager?.render(ctx);
    });
    
    // 设置拖放回调
    this.renderer.setDropCallback((x, y, dataTransfer) => {
      this.onDrop?.(x, y, dataTransfer);
    });
    
    // 适配 GraphController 回调到 INodeEditor 接口
    this.controller.onNodePositionChange = (nodeId, x, y) => {
      const change: NodeChange = {
        type: 'position',
        id: nodeId,
        position: { x, y },
        dragging: true,
      };
      this.onNodesChange?.([change]);
    };
    
    this.controller.onSelectionChange = (nodeIds, edgeIds) => {
      const newSelection: EditorSelection = { nodeIds, edgeIds };
      this.selection = newSelection;
      this.onSelectionChange?.(newSelection);
    };
    
    this.controller.onConnectionAttempt = (sourceNodeId, sourceHandleId, targetNodeId, targetHandleId) => {
      const request: ConnectionRequest = {
        source: sourceNodeId,
        sourceHandle: sourceHandleId,
        target: targetNodeId,
        targetHandle: targetHandleId,
      };
      this.onConnect?.(request);
    };
    
    this.controller.onDeleteSelected = () => {
      this.onDeleteRequest?.(this.selection.nodeIds, this.selection.edgeIds);
    };
    
    this.controller.onEdgeDoubleClick = (edgeId) => {
      this.onEdgeDoubleClick?.(edgeId);
    };
    
    this.controller.onNodeDoubleClick = (nodeId) => {
      this.onNodeDoubleClick?.(nodeId);
    };
    
    this.controller.onViewportChange = (viewport) => {
      this.onViewportChange?.(viewport);
    };
    
    // Hover 状态更新回调
    this.controller.onHoverChange = (screenX, screenY) => {
      this.updateHoverState(screenX, screenY);
    };
    
    // 扩展命中测试回调（类型标签、按钮等）
    this.controller.onExtendedHitTest = (screenX, screenY) => {
      return this.handleExtendedHitTest(screenX, screenY);
    };
    
    // UI 事件路由（UI 组件优先）
    this.controller.onPreMouseDown = (screenX, screenY) => {
      return this.uiManager?.handleMouseDown({ x: screenX, y: screenY, button: 0 }) ?? false;
    };
    
    this.controller.onPreMouseMove = (screenX, screenY) => {
      return this.uiManager?.handleMouseMove({ x: screenX, y: screenY, button: 0 }) ?? false;
    };
    
    this.controller.onPreMouseUp = (screenX, screenY) => {
      return this.uiManager?.handleMouseUp({ x: screenX, y: screenY, button: 0 }) ?? false;
    };
    
    this.controller.onPreWheel = (screenX, screenY, deltaX, deltaY) => {
      const handled = this.uiManager?.handleWheel({ x: screenX, y: screenY, deltaX, deltaY }) ?? false;
      if (handled) {
        // UI 组件处理了滚轮事件，需要触发重新渲染
        this.controller?.requestRender();
      }
      return handled;
    };
    
    this.controller.onPreKeyDown = (key, code, ctrlKey, shiftKey, altKey) => {
      return this.uiManager?.handleKeyDown({ key, code, ctrlKey, shiftKey, altKey }) ?? false;
    };
  }

  unmount(): void {
    window.removeEventListener('resize', this.handleResize);
    
    if (this.uiManager) {
      this.uiManager.unmount();
      this.uiManager.dispose();
      this.uiManager = null;
    }
    if (this.renderer) {
      this.renderer.unmount();
      this.renderer = null;
    }
    this.controller = null;
    this.container = null;
  }

  private handleResize = (): void => {
    this.renderer?.resize();
    this.controller?.requestRender();
  };

  // ============================================================
  // 数据设置
  // ============================================================

  setNodes(nodes: EditorNode[]): void {
    this.nodes = nodes;
    
    // 同步选择状态
    const selectedIds = nodes.filter(n => n.selected).map(n => n.id);
    this.controller?.syncSelectionFromExternal(selectedIds);
    
    // 只有在非拖拽状态时才清除缓存
    const state = this.controller?.getState();
    if (state?.kind === 'idle') {
      this.controller?.clearLayoutCache();
    }
    this.controller?.requestRender();
  }

  setEdges(edges: EditorEdge[]): void {
    this.edges = edges;
    this.controller?.requestRender();
  }

  setSelection(selection: EditorSelection): void {
    this.selection = selection;
    this.controller?.syncSelectionFromExternal(selection.nodeIds);
  }

  setViewport(viewport: EditorViewport): void {
    // 静默设置，不触发回调（避免循环）
    this.controller?.setViewportSilent(viewport);
  }

  // ============================================================
  // 命令
  // ============================================================

  fitView(options?: { padding?: number; maxZoom?: number }): void {
    if (!this.controller || !this.container) return;
    if (this.nodes.length === 0) return;
    
    const padding = options?.padding ?? 50;
    const maxZoom = options?.maxZoom ?? 1;
    
    // 计算边界
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const node of this.nodes) {
      minX = Math.min(minX, node.position.x);
      minY = Math.min(minY, node.position.y);
      maxX = Math.max(maxX, node.position.x + 200); // 估算节点宽度
      maxY = Math.max(maxY, node.position.y + 100); // 估算节点高度
    }
    
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    const graphWidth = maxX - minX + padding * 2;
    const graphHeight = maxY - minY + padding * 2;
    
    const zoom = Math.min(width / graphWidth, height / graphHeight, maxZoom);
    const x = (width - graphWidth * zoom) / 2 - minX * zoom + padding * zoom;
    const y = (height - graphHeight * zoom) / 2 - minY * zoom + padding * zoom;
    
    this.controller.setViewport({ x, y, zoom });
  }

  getViewport(): EditorViewport {
    const vp = this.controller?.getViewport();
    return vp ?? { x: 0, y: 0, zoom: 1 };
  }

  screenToCanvas(screenX: number, screenY: number): { x: number; y: number } {
    return this.controller?.screenToCanvas(screenX, screenY) ?? { x: 0, y: 0 };
  }

  // ============================================================
  // 元信息
  // ============================================================

  getName(): string {
    return 'Canvas';
  }

  isAvailable(): boolean {
    return typeof document !== 'undefined' && 
           typeof document.createElement('canvas').getContext === 'function';
  }

  // ============================================================
  // 渲染模式控制（仅 GPU 渲染器有效）
  // ============================================================

  /**
   * 设置文字渲染模式（仅 GPU 渲染器有效）
   */
  setTextRenderMode(mode: 'gpu' | 'canvas'): void {
    const renderer = this.renderer as unknown as { setTextRenderMode?: (m: 'gpu' | 'canvas') => void };
    if (typeof renderer?.setTextRenderMode === 'function') {
      renderer.setTextRenderMode(mode);
    }
  }

  /**
   * 设置边渲染模式（仅 GPU 渲染器有效）
   */
  setEdgeRenderMode(mode: 'gpu' | 'canvas'): void {
    const renderer = this.renderer as unknown as { setEdgeRenderMode?: (m: 'gpu' | 'canvas') => void };
    if (typeof renderer?.setEdgeRenderMode === 'function') {
      renderer.setEdgeRenderMode(mode);
    }
  }

  // ============================================================
  // 类型选择器管理（原生 Canvas UI）
  // ============================================================

  /**
   * 显示类型选择器
   */
  showTypeSelector(
    nodeId: string,
    handleId: string,
    screenX: number,
    screenY: number,
    options: TypeOption[],
    currentType?: string,
    constraintData?: ConstraintData
  ): void {
    if (!this.uiManager) return;
    
    this.uiManager.showTypeSelector(nodeId, handleId, screenX, screenY, options, currentType, constraintData);
    this.onTypeSelectorChange?.(this.uiManager.getTypeSelectorState());
    this.controller?.requestRender();
  }

  /**
   * 隐藏类型选择器
   */
  hideTypeSelector(): void {
    this.uiManager?.hideTypeSelector();
    this.onTypeSelectorChange?.(null);
    this.controller?.requestRender();
  }

  /**
   * 类型选择器是否可见
   */
  isTypeSelectorVisible(): boolean {
    return this.uiManager?.isTypeSelectorVisible() ?? false;
  }

  /**
   * 设置类型选择回调
   */
  setTypeSelectCallback(callback: (nodeId: string, handleId: string, type: string) => void): void {
    this.uiManager?.setCallbacks({
      ...this.uiManager?.['callbacks'],
      onTypeSelect: callback,
    });
  }

  /**
   * 处理类型标签点击（供外部调用）
   */
  handleTypeLabelClick(screenX: number, screenY: number): boolean {
    return this.handleExtendedHitTest(screenX, screenY);
  }

  /**
   * 处理 Variadic 按钮点击（供外部调用）
   */
  handleVariadicButtonClick(screenX: number, screenY: number): boolean {
    return this.handleExtendedHitTest(screenX, screenY);
  }

  // ============================================================
  // 可编辑名称管理（原生 Canvas UI）
  // ============================================================

  /**
   * 显示可编辑名称
   */
  showEditableName(
    nodeId: string,
    fieldId: string,
    screenX: number,
    screenY: number,
    width: number,
    value: string,
    placeholder?: string
  ): void {
    if (!this.uiManager) return;
    
    this.uiManager.showEditableName(nodeId, fieldId, screenX, screenY, width, value, placeholder);
    this.onEditableNameChange?.(this.uiManager.getEditableNameState());
    this.controller?.requestRender();
  }

  /**
   * 隐藏可编辑名称
   */
  hideEditableName(): void {
    this.uiManager?.hideEditableName();
    this.onEditableNameChange?.(null);
    this.controller?.requestRender();
  }

  /**
   * 可编辑名称是否可见
   */
  isEditableNameVisible(): boolean {
    return this.uiManager?.isEditableNameVisible() ?? false;
  }

  /**
   * 设置名称提交回调
   */
  setNameSubmitCallback(callback: (nodeId: string, fieldId: string, value: string) => void): void {
    this.uiManager?.setCallbacks({
      ...this.uiManager?.['callbacks'],
      onNameSubmit: callback,
    });
  }

  // ============================================================
  // 属性编辑器管理（原生 Canvas UI）
  // ============================================================

  /**
   * 显示属性编辑器
   */
  showAttributeEditor(
    nodeId: string,
    screenX: number,
    screenY: number,
    attributes: AttributeDefinition[],
    title?: string
  ): void {
    if (!this.uiManager) return;
    
    this.uiManager.showAttributeEditor(nodeId, screenX, screenY, attributes, title);
    this.onAttributeEditorChange?.(this.uiManager.getAttributeEditorState());
    this.controller?.requestRender();
  }

  /**
   * 隐藏属性编辑器
   */
  hideAttributeEditor(): void {
    this.uiManager?.hideAttributeEditor();
    this.onAttributeEditorChange?.(null);
    this.controller?.requestRender();
  }

  /**
   * 属性编辑器是否可见
   */
  isAttributeEditorVisible(): boolean {
    return this.uiManager?.isAttributeEditorVisible() ?? false;
  }

  /**
   * 设置属性变更回调
   */
  setAttributeChangeCallback(callback: (nodeId: string, attrName: string, value: unknown) => void): void {
    this.uiManager?.setCallbacks({
      ...this.uiManager?.['callbacks'],
      onAttributeChange: callback,
    });
  }

  // ============================================================
  // 扩展命中测试处理
  // ============================================================

  /**
   * 处理扩展命中测试（基于 LayoutBox）
   * 返回是否命中了交互区域
   */
  handleExtendedHitTest(screenX: number, screenY: number): boolean {
    if (!this.controller) return false;
    
    // 获取缓存的渲染数据中的 layoutBoxes
    const renderData = this.controller.getCachedRenderData();
    if (!renderData.layoutBoxes || renderData.layoutBoxes.size === 0) {
      return false;
    }
    
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    
    // 遍历所有节点的 LayoutBox 进行命中测试
    for (const [nodeId, layoutBox] of renderData.layoutBoxes) {
      // 计算相对于节点的坐标
      const localX = canvasPos.x - layoutBox.x;
      const localY = canvasPos.y - layoutBox.y;
      
      // 检查是否在节点范围内
      if (localX < 0 || localX > layoutBox.width || localY < 0 || localY > layoutBox.height) {
        continue;
      }
      
      // 使用 LayoutBox 命中测试
      const hit = hitTestLayoutBox(layoutBox, localX, localY);
      if (!hit || !hit.box.interactive?.id) {
        continue;
      }
      
      // 解析 interactive.id
      const parsed = parseInteractiveId(hit.box.interactive.id);
      
      // 找到对应的节点
      const node = this.nodes.find(n => n.id === nodeId);
      if (!node) continue;
      
      // 处理命中结果
      if (this.handleParsedHitResult(parsed, nodeId, node, canvasPos.x, canvasPos.y)) {
        return true;
      }
    }
    
    return false;
  }

  /**
   * 处理解析后的命中结果
   */
  private handleParsedHitResult(
    parsed: ReturnType<typeof parseInteractiveId>,
    nodeId: string,
    node: EditorNode,
    canvasX: number,
    canvasY: number
  ): boolean {
    switch (parsed.type) {
      case 'type-label':
        if (parsed.handleId) {
          this.onTypeLabelClick?.(nodeId, parsed.handleId, canvasX, canvasY);
          return true;
        }
        break;
        
      case 'variadic':
        if (parsed.group && parsed.action) {
          if (parsed.action === 'add') {
            this.onVariadicAdd?.(nodeId, parsed.group);
          } else {
            this.onVariadicRemove?.(nodeId, parsed.group);
          }
          return true;
        }
        break;
        
      case 'param-add':
        this.onParameterAdd?.(nodeId);
        return true;
        
      case 'param-remove':
        if (parsed.index !== undefined) {
          const entryData = node.data as FunctionEntryData;
          const paramName = entryData.outputs?.[parsed.index]?.name;
          if (paramName) {
            this.onParameterRemove?.(nodeId, paramName);
          }
          return true;
        }
        break;
        
      case 'param-name':
        if (parsed.index !== undefined) {
          const entryData = node.data as FunctionEntryData;
          const paramName = entryData.outputs?.[parsed.index]?.name ?? '';
          this.onParamNameClick?.(nodeId, parsed.index, paramName, canvasX, canvasY);
          return true;
        }
        break;
        
      case 'return-add':
        this.onReturnTypeAdd?.(nodeId);
        return true;
        
      case 'return-remove':
        if (parsed.index !== undefined) {
          const returnData = node.data as FunctionReturnData;
          const returnName = returnData.inputs?.[parsed.index]?.name;
          if (returnName) {
            this.onReturnTypeRemove?.(nodeId, returnName);
          }
          return true;
        }
        break;
        
      case 'return-name':
        if (parsed.index !== undefined) {
          const returnData = node.data as FunctionReturnData;
          const returnName = returnData.inputs?.[parsed.index]?.name ?? '';
          this.onReturnNameClick?.(nodeId, parsed.index, returnName, canvasX, canvasY);
          return true;
        }
        break;
        
      case 'traits-toggle':
        this.onTraitsToggleClick?.(nodeId, canvasX, canvasY);
        return true;
        
      case 'summary-toggle':
        this.toggleSummaryExpanded(nodeId);
        return true;
    }
    
    return false;
  }

  /**
   * 切换 Summary 展开状态
   */
  private toggleSummaryExpanded(nodeId: string): void {
    const current = this.summaryExpandedMap.get(nodeId) ?? false;
    this.summaryExpandedMap.set(nodeId, !current);
    this.controller?.requestRender();
  }

  /**
   * 更新 hover 状态（用于显示删除按钮）
   * 基于 LayoutBox 命中测试
   */
  updateHoverState(screenX: number, screenY: number): void {
    if (!this.controller) return;
    
    const renderData = this.controller.getCachedRenderData();
    if (!renderData.layoutBoxes || renderData.layoutBoxes.size === 0) {
      return;
    }
    
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    let newHoveredNodeId: string | null = null;
    let newHoveredParamIndex: number | null = null;
    let newHoveredReturnIndex: number | null = null;
    
    for (const [nodeId, layoutBox] of renderData.layoutBoxes) {
      const localX = canvasPos.x - layoutBox.x;
      const localY = canvasPos.y - layoutBox.y;
      
      if (localX < 0 || localX > layoutBox.width || localY < 0 || localY > layoutBox.height) {
        continue;
      }
      
      const hit = hitTestLayoutBox(layoutBox, localX, localY);
      if (!hit || !hit.box.interactive?.id) {
        continue;
      }
      
      const parsed = parseInteractiveId(hit.box.interactive.id);
      
      // 检查是否命中参数或返回值区域
      if (parsed.type === 'param-name' || parsed.type === 'param-remove') {
        newHoveredNodeId = nodeId;
        newHoveredParamIndex = parsed.index ?? null;
        break;
      } else if (parsed.type === 'return-name' || parsed.type === 'return-remove') {
        newHoveredNodeId = nodeId;
        newHoveredReturnIndex = parsed.index ?? null;
        break;
      }
    }
    
    // 检查是否有变化
    if (newHoveredNodeId !== this.hoveredNodeId ||
        newHoveredParamIndex !== this.hoveredParamIndex ||
        newHoveredReturnIndex !== this.hoveredReturnIndex) {
      this.hoveredNodeId = newHoveredNodeId;
      this.hoveredParamIndex = newHoveredParamIndex;
      this.hoveredReturnIndex = newHoveredReturnIndex;
      this.controller?.requestRender();
    }
  }
}

/**
 * 创建 Canvas 节点编辑器实例
 * @param rendererFactory 可选的渲染器工厂函数，用于创建自定义渲染器
 */
export function createCanvasNodeEditor(rendererFactory?: () => IExtendedRenderer): INodeEditor {
  return new CanvasNodeEditor(rendererFactory);
}
