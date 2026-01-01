/**
 * MiniMap 组件
 * 
 * 使用 Canvas 2D 渲染图的缩略图。
 * 显示视口指示器，支持点击和拖拽导航。
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import type { EditorNode, EditorViewport } from '../../types';

/** 节点尺寸 */
export interface NodeSize {
  width: number;
  height: number;
}

/** 获取节点尺寸的函数类型 */
export type GetNodeSizeFn = (node: EditorNode) => NodeSize;

export interface MiniMapProps {
  /** 节点列表 */
  nodes: EditorNode[];
  /** 当前视口 */
  viewport: EditorViewport;
  /** 容器宽度 */
  containerWidth: number;
  /** 容器高度 */
  containerHeight: number;
  /** MiniMap 宽度 */
  width?: number;
  /** MiniMap 高度 */
  height?: number;
  /** 视口变更回调 */
  onViewportChange?: (viewport: EditorViewport) => void;
  /** 自定义获取节点尺寸函数 */
  getNodeSize?: GetNodeSizeFn;
  /** 自定义类名 */
  className?: string;
  /** 自定义样式 */
  style?: React.CSSProperties;
}

/** 视口指示器颜色 */
const VIEWPORT_COLOR = 'rgba(66, 153, 225, 0.3)';
/** 视口边框颜色 */
const VIEWPORT_BORDER_COLOR = 'rgba(66, 153, 225, 0.8)';
/** 背景颜色 - 与 ReactFlow/VueFlow MiniMap 默认一致 */
const BACKGROUND_COLOR = '#1f2937';

/**
 * 根据节点类型获取颜色
 * 与 ReactFlow/VueFlow MiniMap 保持一致
 */
function getNodeColor(nodeType: string | undefined): string {
  switch (nodeType) {
    case 'function-entry': return '#22c55e';  // 绿色
    case 'function-return': return '#ef4444'; // 红色
    case 'function-call': return '#a855f7';   // 紫色
    default: return '#3b82f6';                // 蓝色（operation）
  }
}


/** 默认获取节点尺寸函数 */
function defaultGetNodeSize(node: EditorNode): NodeSize {
  const nodeData = node.data as Record<string, unknown> | undefined;
  return {
    width: typeof nodeData?.width === 'number' ? nodeData.width : 200,
    height: typeof nodeData?.height === 'number' ? nodeData.height : 100,
  };
}


/**
 * 计算 MiniMap 的显示边界
 * 
 * 策略：以节点边界为基础，确保视口框始终完全可见
 * 当视口超出节点边界时，扩展边界以包含视口
 */
function computeBounds(
  nodes: EditorNode[],
  viewport: { x: number; y: number; zoom: number },
  containerWidth: number,
  containerHeight: number,
  getNodeSize: GetNodeSizeFn
): {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
} {
  // 计算视口在画布坐标中的范围
  const viewportCanvasX = -viewport.x / viewport.zoom;
  const viewportCanvasY = -viewport.y / viewport.zoom;
  const viewportCanvasW = containerWidth / viewport.zoom;
  const viewportCanvasH = containerHeight / viewport.zoom;
  
  // 先计算节点边界
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  
  for (const node of nodes) {
    const { width, height } = getNodeSize(node);
    
    minX = Math.min(minX, node.position.x);
    minY = Math.min(minY, node.position.y);
    maxX = Math.max(maxX, node.position.x + width);
    maxY = Math.max(maxY, node.position.y + height);
  }
  
  // 如果没有节点，使用视口作为边界
  if (!isFinite(minX)) {
    minX = viewportCanvasX;
    minY = viewportCanvasY;
    maxX = viewportCanvasX + viewportCanvasW;
    maxY = viewportCanvasY + viewportCanvasH;
  } else {
    // 扩展边界以包含视口（确保视口框完全可见）
    minX = Math.min(minX, viewportCanvasX);
    minY = Math.min(minY, viewportCanvasY);
    maxX = Math.max(maxX, viewportCanvasX + viewportCanvasW);
    maxY = Math.max(maxY, viewportCanvasY + viewportCanvasH);
  }
  
  // 添加边距
  const padding = 50;
  minX -= padding;
  minY -= padding;
  maxX += padding;
  maxY += padding;
  
  return {
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

/**
 * MiniMap 组件
 */
export function MiniMap({
  nodes,
  viewport,
  containerWidth,
  containerHeight,
  width = 200,
  height = 150,
  onViewportChange,
  getNodeSize = defaultGetNodeSize,
  className,
  style,
}: MiniMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  // 计算缩放比例（包含视口和节点的联合边界）
  const bounds = computeBounds(nodes, viewport, containerWidth, containerHeight, getNodeSize);
  const scaleX = width / bounds.width;
  const scaleY = height / bounds.height;
  const scale = Math.min(scaleX, scaleY);
  
  // 计算居中偏移
  const contentWidth = bounds.width * scale;
  const contentHeight = bounds.height * scale;
  const offsetX = (width - contentWidth) / 2;
  const offsetY = (height - contentHeight) / 2;
  
  // 画布坐标转 MiniMap 坐标（带居中偏移）
  const canvasToMiniMap = useCallback((x: number, y: number) => {
    return {
      x: (x - bounds.minX) * scale + offsetX,
      y: (y - bounds.minY) * scale + offsetY,
    };
  }, [bounds.minX, bounds.minY, scale, offsetX, offsetY]);
  
  // MiniMap 坐标转画布坐标（带居中偏移）
  const miniMapToCanvas = useCallback((x: number, y: number) => {
    return {
      x: (x - offsetX) / scale + bounds.minX,
      y: (y - offsetY) / scale + bounds.minY,
    };
  }, [bounds.minX, bounds.minY, scale, offsetX, offsetY]);

  // 渲染 MiniMap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // 设置 canvas 尺寸
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    // 清空背景
    ctx.fillStyle = BACKGROUND_COLOR;
    ctx.fillRect(0, 0, width, height);
    
    // 绘制节点
    for (const node of nodes) {
      const { width: nodeWidth, height: nodeHeight } = getNodeSize(node);
      
      const pos = canvasToMiniMap(node.position.x, node.position.y);
      const w = nodeWidth * scale;
      const h = nodeHeight * scale;
      
      ctx.fillStyle = getNodeColor(node.type);
      ctx.fillRect(pos.x, pos.y, Math.max(w, 2), Math.max(h, 2));
    }
    
    // 绘制视口指示器
    // 视口在画布坐标中的位置：左上角 = (-viewport.x / viewport.zoom, -viewport.y / viewport.zoom)
    const viewportCanvasX = -viewport.x / viewport.zoom;
    const viewportCanvasY = -viewport.y / viewport.zoom;
    const viewportCanvasW = containerWidth / viewport.zoom;
    const viewportCanvasH = containerHeight / viewport.zoom;
    
    const vpPos = canvasToMiniMap(viewportCanvasX, viewportCanvasY);
    const vpW = viewportCanvasW * scale;
    const vpH = viewportCanvasH * scale;
    
    // 填充
    ctx.fillStyle = VIEWPORT_COLOR;
    ctx.fillRect(vpPos.x, vpPos.y, vpW, vpH);
    
    // 边框
    ctx.strokeStyle = VIEWPORT_BORDER_COLOR;
    ctx.lineWidth = 2;
    ctx.strokeRect(vpPos.x, vpPos.y, vpW, vpH);
  }, [nodes, viewport, containerWidth, containerHeight, width, height, scale, canvasToMiniMap, getNodeSize]);

  // 处理点击导航
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onViewportChange || isDragging) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // 转换为画布坐标
    const canvasPos = miniMapToCanvas(x, y);
    
    // 计算新视口中心
    const newX = -(canvasPos.x - containerWidth / (2 * viewport.zoom)) * viewport.zoom;
    const newY = -(canvasPos.y - containerHeight / (2 * viewport.zoom)) * viewport.zoom;
    
    onViewportChange({ x: newX, y: newY, zoom: viewport.zoom });
  }, [onViewportChange, isDragging, miniMapToCanvas, containerWidth, containerHeight, viewport.zoom]);

  // 处理拖拽导航
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onViewportChange) return;
    e.preventDefault();
    setIsDragging(true);
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    
    const handleMouseMove = (moveEvent: MouseEvent) => {
      const x = moveEvent.clientX - rect.left;
      const y = moveEvent.clientY - rect.top;
      
      const canvasPos = miniMapToCanvas(x, y);
      const newX = -(canvasPos.x - containerWidth / (2 * viewport.zoom)) * viewport.zoom;
      const newY = -(canvasPos.y - containerHeight / (2 * viewport.zoom)) * viewport.zoom;
      
      onViewportChange({ x: newX, y: newY, zoom: viewport.zoom });
    };
    
    const handleMouseUp = () => {
      setIsDragging(false);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [onViewportChange, miniMapToCanvas, containerWidth, containerHeight, viewport.zoom]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={className}
      style={{
        width,
        height,
        cursor: isDragging ? 'grabbing' : 'pointer',
        ...style,
      }}
      onClick={handleClick}
      onMouseDown={handleMouseDown}
    />
  );
}