/**
 * MSDF 字体图集加载器
 * 
 * 加载和管理 MSDF (Multi-channel Signed Distance Field) 字体图集。
 * MSDF 字体可以在任意缩放下保持清晰的边缘。
 */

/** 字符度量信息 */
export interface GlyphMetrics {
  /** 字符 Unicode 码点 */
  id: number;
  /** 在图集中的 X 位置 */
  x: number;
  /** 在图集中的 Y 位置 */
  y: number;
  /** 字符宽度 */
  width: number;
  /** 字符高度 */
  height: number;
  /** X 偏移（渲染时） */
  xoffset: number;
  /** Y 偏移（渲染时） */
  yoffset: number;
  /** 水平前进量 */
  xadvance: number;
}

/** 字体图集元数据 */
export interface FontAtlasData {
  /** 字体名称 */
  name: string;
  /** 字体大小（生成时） */
  size: number;
  /** 行高 */
  lineHeight: number;
  /** 基线位置 */
  base: number;
  /** 图集宽度 */
  scaleW: number;
  /** 图集高度 */
  scaleH: number;
  /** 字符度量映射 */
  chars: Map<number, GlyphMetrics>;
  /** 字距调整对 */
  kernings: Map<string, number>;
}

/** 字体图集加载状态 */
export type FontAtlasState = 'loading' | 'ready' | 'error';

/**
 * 字体图集加载器
 */
export class FontAtlas {
  private data: FontAtlasData | null = null;
  private texture: HTMLImageElement | null = null;
  private state: FontAtlasState = 'loading';
  private error: Error | null = null;

  
  /**
   * 获取加载状态
   */
  getState(): FontAtlasState {
    return this.state;
  }
  
  /**
   * 获取错误信息
   */
  getError(): Error | null {
    return this.error;
  }
  
  /**
   * 获取字体数据
   */
  getData(): FontAtlasData | null {
    return this.data;
  }
  
  /**
   * 获取纹理图像
   */
  getTexture(): HTMLImageElement | null {
    return this.texture;
  }
  
  /**
   * 加载字体图集
   * @param jsonUrl 字体元数据 JSON 文件 URL
   * @param imageUrl 字体图集图像 URL
   */
  async load(jsonUrl: string, imageUrl: string): Promise<void> {
    this.state = 'loading';
    
    try {
      // 并行加载 JSON 和图像
      const [jsonResponse, image] = await Promise.all([
        fetch(jsonUrl).then(r => r.json()),
        this.loadImage(imageUrl),
      ]);
      
      // 解析字体数据
      this.data = this.parseJsonData(jsonResponse);
      this.texture = image;
      this.state = 'ready';
    } catch (e) {
      this.error = e instanceof Error ? e : new Error(String(e));
      this.state = 'error';
      throw this.error;
    }
  }
  
  /**
   * 加载图像
   */
  private loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
      img.src = url;
    });
  }
  
  /**
   * 解析 BMFont JSON 格式数据
   */
  private parseJsonData(json: Record<string, unknown>): FontAtlasData {
    const info = json.info as Record<string, unknown> || {};
    const common = json.common as Record<string, unknown> || {};
    const chars = json.chars as Array<Record<string, number>> || [];
    const kernings = json.kernings as Array<Record<string, number>> || [];
    
    const charMap = new Map<number, GlyphMetrics>();
    for (const char of chars) {
      charMap.set(char.id, {
        id: char.id,
        x: char.x,
        y: char.y,
        width: char.width,
        height: char.height,
        xoffset: char.xoffset,
        yoffset: char.yoffset,
        xadvance: char.xadvance,
      });
    }
    
    const kerningMap = new Map<string, number>();
    for (const kern of kernings) {
      const key = `${kern.first}-${kern.second}`;
      kerningMap.set(key, kern.amount);
    }
    
    return {
      name: String(info.face || 'Unknown'),
      size: Number(info.size || 32),
      lineHeight: Number(common.lineHeight || 32),
      base: Number(common.base || 26),
      scaleW: Number(common.scaleW || 512),
      scaleH: Number(common.scaleH || 512),
      chars: charMap,
      kernings: kerningMap,
    };
  }

  
  /**
   * 获取字符度量
   */
  getGlyph(charCode: number): GlyphMetrics | undefined {
    return this.data?.chars.get(charCode);
  }
  
  /**
   * 获取字距调整
   */
  getKerning(first: number, second: number): number {
    const key = `${first}-${second}`;
    return this.data?.kernings.get(key) || 0;
  }
  
  /**
   * 计算文本宽度
   */
  measureText(text: string, fontSize: number): number {
    if (!this.data) return 0;
    
    const scale = fontSize / this.data.size;
    let width = 0;
    let prevCharCode = 0;
    
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const glyph = this.data.chars.get(charCode);
      
      if (glyph) {
        // 添加字距调整
        if (prevCharCode) {
          width += this.getKerning(prevCharCode, charCode) * scale;
        }
        width += glyph.xadvance * scale;
      }
      
      prevCharCode = charCode;
    }
    
    return width;
  }
  
  /**
   * 布局文本，返回每个字符的位置和 UV 坐标
   */
  layoutText(
    text: string,
    x: number,
    y: number,
    fontSize: number,
    align: 'left' | 'center' | 'right' = 'left'
  ): Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    u0: number;
    v0: number;
    u1: number;
    v1: number;
  }> {
    if (!this.data) return [];
    
    const scale = fontSize / this.data.size;
    const result: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      u0: number;
      v0: number;
      u1: number;
      v1: number;
    }> = [];
    
    // 计算对齐偏移
    let offsetX = 0;
    if (align !== 'left') {
      const textWidth = this.measureText(text, fontSize);
      offsetX = align === 'center' ? -textWidth / 2 : -textWidth;
    }
    
    let cursorX = x + offsetX;
    let prevCharCode = 0;
    
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const glyph = this.data.chars.get(charCode);
      
      if (glyph) {
        // 添加字距调整
        if (prevCharCode) {
          cursorX += this.getKerning(prevCharCode, charCode) * scale;
        }
        
        const charX = cursorX + glyph.xoffset * scale;
        const charY = y + glyph.yoffset * scale;
        const charW = glyph.width * scale;
        const charH = glyph.height * scale;
        
        // UV 坐标（归一化）
        const u0 = glyph.x / this.data.scaleW;
        const v0 = glyph.y / this.data.scaleH;
        const u1 = (glyph.x + glyph.width) / this.data.scaleW;
        const v1 = (glyph.y + glyph.height) / this.data.scaleH;
        
        result.push({
          x: charX,
          y: charY,
          width: charW,
          height: charH,
          u0,
          v0,
          u1,
          v1,
        });
        
        cursorX += glyph.xadvance * scale;
      }
      
      prevCharCode = charCode;
    }
    
    return result;
  }
}

/** 默认字体图集实例 */
let defaultFontAtlas: FontAtlas | null = null;

/**
 * 获取默认字体图集
 */
export function getDefaultFontAtlas(): FontAtlas {
  if (!defaultFontAtlas) {
    defaultFontAtlas = new FontAtlas();
  }
  return defaultFontAtlas;
}

/**
 * 加载默认字体图集
 */
export async function loadDefaultFontAtlas(
  jsonUrl: string,
  imageUrl: string
): Promise<FontAtlas> {
  const atlas = getDefaultFontAtlas();
  await atlas.load(jsonUrl, imageUrl);
  return atlas;
}
