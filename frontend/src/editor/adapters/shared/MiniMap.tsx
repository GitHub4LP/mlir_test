/**
 * MiniMap 组件
 * 
 * 使用 Canvas 2D 渲染图的缩略图。
 * 显示视口指示器，支持点击和拖拽导航。
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import type { EditorNode, EditorViewport } from '../../types';

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
  /** 自定义类名 */
  className?: string;
  /** 自定义样式 */
  style?: React.CSSProperties;
}

/** 默认节点颜色 */
const NODE_COLOR = '#4a5568';
/** 选中节点颜色 */
const SELECTED_NODE_COLOR = '#4299e1';
/** 视口指示器颜色 */
const VIEWPORT_COLOR = 'rgba(66, 153, 225, 0.3)';
/** 视口边框颜色 */
const VIEWPORT_BORDER_COLOR = 'rgba(66, 153, 225, 0.8)';
/** 背景颜色 */
const BACKGROUND_COLOR = '#1a1a1a';


/**
 * 计算节点边界
 */
function computeBounds(nodes: EditorNode[]): {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
} {
  if (nodes.length === 0) {
    return { minX: 0, minY: 0, maxX: 100, maxY: 100, width: 100, height: 100 };
  }
  
  let minX = Infinity, minY = Infinity;
  let maxX = -Infinity, maxY = -Infinity;
  
  for (const node of nodes) {
    const nodeData = node.data as Record<string, unknown> | undefined;
    const width = (typeof nodeData?.width === 'number' ? nodeData.width : 200);
    const height = (typeof nodeData?.height === 'number' ? nodeData.height : 100);
    
    minX = Math.min(minX, node.position.x);
    minY = Math.min(minY, node.position.y);
    maxX = Math.max(maxX, node.position.x + width);
    maxY = Math.max(maxY, node.position.y + height);
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
  className,
  style,
}: MiniMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  // 计算缩放比例
  const bounds = computeBounds(nodes);
  const scaleX = width / bounds.width;
  const scaleY = height / bounds.height;
  const scale = Math.min(scaleX, scaleY);
  
  // 画布坐标转 MiniMap 坐标
  const canvasToMiniMap = useCallback((x: number, y: number) => {
    return {
      x: (x - bounds.minX) * scale,
      y: (y - bounds.minY) * scale,
    };
  }, [bounds.minX, bounds.minY, scale]);
  
  // MiniMap 坐标转画布坐标
  const miniMapToCanvas = useCallback((x: number, y: number) => {
    return {
      x: x / scale + bounds.minX,
      y: y / scale + bounds.minY,
    };
  }, [bounds.minX, bounds.minY, scale]);

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
      const nodeData = node.data as Record<string, unknown> | undefined;
      const nodeWidth = (typeof nodeData?.width === 'number' ? nodeData.width : 200);
      const nodeHeight = (typeof nodeData?.height === 'number' ? nodeData.height : 100);
      
      const pos = canvasToMiniMap(node.position.x, node.position.y);
      const w = nodeWidth * scale;
      const h = nodeHeight * scale;
      
      ctx.fillStyle = node.selected ? SELECTED_NODE_COLOR : NODE_COLOR;
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
  }, [nodes, viewport, containerWidth, containerHeight, width, height, scale, canvasToMiniMap]);

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