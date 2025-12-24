/**
 * Canvas 节点编辑器
 * 
 * 实现 INodeEditor 接口，封装 GraphController 和 CanvasRenderer。
 * 将旧架构的图元级别 API 适配为 Node Editor 语义级别。
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
import { CanvasRenderer } from './canvas/CanvasRenderer';
import { GraphController } from './canvas/GraphController';
import { UIManager, type TypeSelectorState } from './canvas/UIManager';
import type { TypeOption } from './canvas/ui/TypeSelector';
import { 
  hitTestTypeLabel, 
  hitTestVariadicButton, 
  hitTestNodeExtended,
  computeHoveredIndex,
  type HitResult,
  type EntryNodeHitData,
  type ReturnNodeHitData,
  type NodeType,
} from './canvas/HitTest';
import { computeNodeLayout } from './canvas/layout';
import { extendRenderData, type RenderExtensionConfig } from './canvas/RenderExtensions';
import type { GraphState, GraphNode, BlueprintNodeData, FunctionEntryData, FunctionReturnData } from '../../types';

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
    })),
  };
}

/**
 * Canvas 节点编辑器
 * 
 * 实现 INodeEditor 接口，使用 Canvas 2D 渲染。
 */
export class CanvasNodeEditor implements INodeEditor {
  private renderer: CanvasRenderer | null = null;
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

  // Traits 展开状态
  private traitsExpandedMap: Map<string, boolean> = new Map();

  // ============================================================
  // 生命周期
  // ============================================================

  mount(container: HTMLElement): void {
    this.container = container;
    this.renderer = new CanvasRenderer();
    this.controller = new GraphController();
    this.uiManager = new UIManager();
    
    this.renderer.mount(container);
    this.uiManager.mount(container);
    this.controller.setRenderer(this.renderer);
    
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
    
    // 设置渲染扩展回调
    this.controller.onExtendRenderData = (data, nodeLayouts) => {
      const nodesMap = new Map<string, GraphNode>();
      for (const node of this.nodes) {
        nodesMap.set(node.id, {
          id: node.id,
          type: node.type as GraphNode['type'],
          position: node.position,
          data: node.data as GraphNode['data'],
        });
      }
      
      const config: RenderExtensionConfig = {
        nodes: nodesMap,
        nodeList: this.nodes,
        hoveredNodeId: this.hoveredNodeId,
        hoveredParamIndex: this.hoveredParamIndex,
        hoveredReturnIndex: this.hoveredReturnIndex,
        traitsExpandedMap: this.traitsExpandedMap,
      };
      
      extendRenderData(data, nodeLayouts, config);
    };
    
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
    
    this.controller.onViewportChange = (viewport) => {
      this.onViewportChange?.(viewport);
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
      return this.uiManager?.handleWheel({ x: screenX, y: screenY, deltaX, deltaY }) ?? false;
    };
    
    this.controller.onPreKeyDown = (key, code, ctrlKey, shiftKey, altKey) => {
      return this.uiManager?.handleKeyDown({ key, code, ctrlKey, shiftKey, altKey }) ?? false;
    };
    
    // 初始渲染
    this.controller.requestRender();
    
    // 监听窗口大小变化
    window.addEventListener('resize', this.handleResize);
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
    currentType?: string
  ): void {
    if (!this.uiManager) return;
    
    this.uiManager.showTypeSelector(nodeId, handleId, screenX, screenY, options, currentType);
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
    if (!this.controller) return false;
    
    // 转换为画布坐标
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    
    // 遍历节点进行命中测试
    for (const node of this.nodes) {
      const graphNode: GraphNode = {
        id: node.id,
        type: node.type as GraphNode['type'],
        position: node.position,
        data: node.data as GraphNode['data'],
      };
      const layout = computeNodeLayout(graphNode, false);
      const hit = hitTestTypeLabel(canvasPos.x, canvasPos.y, layout);
      
      if (hit) {
        this.onTypeLabelClick?.(hit.nodeId, hit.handleId, hit.canvasX, hit.canvasY);
        return true;
      }
    }
    
    return false;
  }

  /**
   * 处理 Variadic 按钮点击（供外部调用）
   */
  handleVariadicButtonClick(screenX: number, screenY: number): boolean {
    if (!this.controller) return false;
    
    // 转换为画布坐标
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    
    // 遍历节点进行命中测试
    for (const node of this.nodes) {
      // 只有 operation 节点有 Variadic 端口
      if (node.type !== 'operation') continue;
      
      const data = node.data as BlueprintNodeData;
      const variadicGroups = this.getVariadicGroups(data);
      if (variadicGroups.length === 0) continue;
      
      const graphNode: GraphNode = {
        id: node.id,
        type: node.type as GraphNode['type'],
        position: node.position,
        data: node.data as GraphNode['data'],
      };
      const layout = computeNodeLayout(graphNode, false);
      const hit = hitTestVariadicButton(canvasPos.x, canvasPos.y, layout, variadicGroups);
      
      if (hit) {
        if (hit.action === 'add') {
          this.onVariadicAdd?.(hit.nodeId, hit.groupName);
        } else {
          this.onVariadicRemove?.(hit.nodeId, hit.groupName);
        }
        return true;
      }
    }
    
    return false;
  }

  /**
   * 获取节点的 Variadic 组名列表
   */
  private getVariadicGroups(data: BlueprintNodeData): string[] {
    const groups: string[] = [];
    const op = data.operation;
    
    // 检查 operands
    for (const arg of op.arguments) {
      if (arg.kind === 'operand' && arg.isVariadic && !groups.includes(arg.name)) {
        groups.push(arg.name);
      }
    }
    
    // 检查 results
    for (const result of op.results) {
      if (result.isVariadic && !groups.includes(result.name)) {
        groups.push(result.name);
      }
    }
    
    return groups;
  }

  // ============================================================
  // 扩展命中测试处理
  // ============================================================

  /**
   * 处理扩展命中测试（支持所有交互区域）
   * 返回是否命中了交互区域
   */
  handleExtendedHitTest(screenX: number, screenY: number): boolean {
    if (!this.controller) return false;
    
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    
    for (const node of this.nodes) {
      const graphNode: GraphNode = {
        id: node.id,
        type: node.type as GraphNode['type'],
        position: node.position,
        data: node.data as GraphNode['data'],
      };
      const layout = computeNodeLayout(graphNode, false);
      
      // 构建命中测试选项
      const options = this.buildHitTestOptions(node);
      const hit = hitTestNodeExtended(canvasPos.x, canvasPos.y, layout, options);
      
      if (this.handleHitResult(hit, node)) {
        return true;
      }
    }
    
    return false;
  }

  /**
   * 构建命中测试选项
   */
  private buildHitTestOptions(node: EditorNode): {
    nodeType: NodeType;
    variadicGroups?: string[];
    entryData?: EntryNodeHitData;
    returnData?: ReturnNodeHitData;
    hoveredParamIndex?: number | null;
    hoveredReturnIndex?: number | null;
  } {
    const nodeType = node.type as NodeType;
    
    switch (nodeType) {
      case 'operation': {
        const data = node.data as BlueprintNodeData;
        return {
          nodeType,
          variadicGroups: this.getVariadicGroups(data),
        };
      }
      case 'function-entry': {
        const data = node.data as FunctionEntryData;
        return {
          nodeType,
          entryData: {
            isMain: data.isMain,
            parameters: data.outputs?.map(o => ({ name: o.name })) ?? [],
            traitsExpanded: false, // TODO: 从状态获取
          },
          hoveredParamIndex: this.hoveredNodeId === node.id ? this.hoveredParamIndex : null,
        };
      }
      case 'function-return': {
        const data = node.data as FunctionReturnData;
        return {
          nodeType,
          returnData: {
            isMain: data.isMain,
            returnTypes: data.inputs?.map(i => ({ name: i.name })) ?? [],
          },
          hoveredReturnIndex: this.hoveredNodeId === node.id ? this.hoveredReturnIndex : null,
        };
      }
      case 'function-call':
        return { nodeType };
      default:
        return { nodeType: 'operation' };
    }
  }

  /**
   * 处理命中结果
   */
  private handleHitResult(hit: HitResult, node: EditorNode): boolean {
    switch (hit.kind) {
      case 'type-label':
        this.onTypeLabelClick?.(hit.nodeId, hit.handleId, hit.canvasX, hit.canvasY);
        return true;
        
      case 'variadic-button':
        if (hit.action === 'add') {
          this.onVariadicAdd?.(hit.nodeId, hit.groupName);
        } else {
          this.onVariadicRemove?.(hit.nodeId, hit.groupName);
        }
        return true;
        
      case 'param-add':
        this.onParameterAdd?.(hit.nodeId);
        return true;
        
      case 'param-remove': {
        const entryData = node.data as FunctionEntryData;
        const paramName = entryData.outputs?.[hit.paramIndex]?.name;
        if (paramName) {
          this.onParameterRemove?.(hit.nodeId, paramName);
        }
        return true;
      }
        
      case 'param-name':
        this.onParamNameClick?.(hit.nodeId, hit.paramIndex, hit.currentName, hit.canvasX, hit.canvasY);
        return true;
        
      case 'return-add':
        this.onReturnTypeAdd?.(hit.nodeId);
        return true;
        
      case 'return-remove': {
        const returnData = node.data as FunctionReturnData;
        const returnName = returnData.inputs?.[hit.returnIndex]?.name;
        if (returnName) {
          this.onReturnTypeRemove?.(hit.nodeId, returnName);
        }
        return true;
      }
        
      case 'return-name':
        this.onReturnNameClick?.(hit.nodeId, hit.returnIndex, hit.currentName, hit.canvasX, hit.canvasY);
        return true;
        
      case 'traits-toggle':
        this.onTraitsToggleClick?.(hit.nodeId, hit.canvasX, hit.canvasY);
        return true;
        
      default:
        return false;
    }
  }

  /**
   * 更新 hover 状态（用于显示删除按钮）
   */
  updateHoverState(screenX: number, screenY: number): void {
    if (!this.controller) return;
    
    const canvasPos = this.controller.screenToCanvas(screenX, screenY);
    let newHoveredNodeId: string | null = null;
    let newHoveredParamIndex: number | null = null;
    let newHoveredReturnIndex: number | null = null;
    
    for (const node of this.nodes) {
      const graphNode: GraphNode = {
        id: node.id,
        type: node.type as GraphNode['type'],
        position: node.position,
        data: node.data as GraphNode['data'],
      };
      const layout = computeNodeLayout(graphNode, false);
      
      if (node.type === 'function-entry') {
        const data = node.data as FunctionEntryData;
        if (!data.isMain) {
          const paramCount = data.outputs?.length ?? 0;
          const index = computeHoveredIndex(canvasPos.x, canvasPos.y, layout, 'function-entry', paramCount);
          if (index !== null) {
            newHoveredNodeId = node.id;
            newHoveredParamIndex = index;
            break;
          }
        }
      } else if (node.type === 'function-return') {
        const data = node.data as FunctionReturnData;
        if (!data.isMain) {
          const returnCount = data.inputs?.length ?? 0;
          const index = computeHoveredIndex(canvasPos.x, canvasPos.y, layout, 'function-return', returnCount);
          if (index !== null) {
            newHoveredNodeId = node.id;
            newHoveredReturnIndex = index;
            break;
          }
        }
      }
    }
    
    // 检查是否有变化
    if (newHoveredNodeId !== this.hoveredNodeId ||
        newHoveredParamIndex !== this.hoveredParamIndex ||
        newHoveredReturnIndex !== this.hoveredReturnIndex) {
      this.hoveredNodeId = newHoveredNodeId;
      this.hoveredParamIndex = newHoveredParamIndex;
      this.hoveredReturnIndex = newHoveredReturnIndex;
      // 触发重渲染以显示/隐藏删除按钮
      this.controller?.requestRender();
    }
  }
}

/**
 * 创建 Canvas 节点编辑器实例
 */
export function createCanvasNodeEditor(): INodeEditor {
  return new CanvasNodeEditor();
}
