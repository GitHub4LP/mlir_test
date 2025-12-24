/**
 * Design Tokens - 自动生成，请勿手动修改
 * 
 * 生成命令: npm run build:tokens
 * 源文件: frontend/tokens/
 */

export const tokens = {
  "color": {
    "gray": {
      "50": "#f9fafb",
      "100": "#f3f4f6",
      "200": "#e5e7eb",
      "300": "#d1d5db",
      "400": "#9ca3af",
      "500": "#6b7280",
      "600": "#4b5563",
      "700": "#374151",
      "800": "#1f2937",
      "900": "#111827",
      "950": "#030712"
    },
    "blue": {
      "400": "#60a5fa",
      "500": "#3b82f6",
      "600": "#2563eb"
    },
    "green": {
      "400": "#4ade80",
      "500": "#22c55e",
      "600": "#16a34a"
    },
    "red": {
      "400": "#f87171",
      "500": "#ef4444",
      "600": "#dc2626"
    },
    "amber": {
      "500": "#f59e0b"
    },
    "purple": {
      "500": "#a855f7"
    },
    "white": "#ffffff",
    "black": "#000000",
    "transparent": "transparent"
  },
  "size": {
    "0": "0",
    "1": 4,
    "2": 8,
    "3": 12,
    "4": 16,
    "5": 20,
    "6": 24,
    "7": 28,
    "8": 32,
    "10": 40,
    "12": 48,
    "16": 64,
    "20": 80,
    "24": 96,
    "xs": 12,
    "sm": 14,
    "base": 16,
    "lg": 18,
    "xl": 20
  },
  "radius": {
    "none": "0",
    "sm": 2,
    "default": 4,
    "md": 6,
    "lg": 8,
    "xl": 12,
    "full": 9999
  },
  "border": {
    "thin": 1,
    "medium": 2,
    "thick": 3
  },
  "font": {
    "family": {
      "sans": "system-ui, -apple-system, sans-serif",
      "mono": "ui-monospace, monospace"
    },
    "size": {
      "xs": 10,
      "sm": 12,
      "base": 14,
      "lg": 16,
      "xl": 18
    },
    "weight": {
      "normal": 400,
      "medium": 500,
      "semibold": 600,
      "bold": 700
    },
    "lineHeight": {
      "tight": "1.25",
      "normal": "1.5",
      "relaxed": "1.75"
    }
  },
  "button": {
    "size": 16,
    "borderRadius": 4,
    "bg": "rgba(255, 255, 255, 0.1)",
    "hoverBg": "rgba(255, 255, 255, 0.2)",
    "borderColor": "rgba(255, 255, 255, 0.3)",
    "borderWidth": 1,
    "textColor": "#ffffff",
    "fontSize": 12,
    "danger": {
      "color": "#ef4444",
      "hoverColor": "#f87171"
    }
  },
  "dialect": {
    "arith": "#4A90D9",
    "func": "#50C878",
    "scf": "#9B59B6",
    "memref": "#E74C3C",
    "tensor": "#1ABC9C",
    "linalg": "#F39C12",
    "vector": "#F1C40F",
    "affine": "#E67E22",
    "gpu": "#2ECC71",
    "math": "#3498DB",
    "cf": "#8E44AD",
    "builtin": "#7F8C8D",
    "default": "#3b82f6"
  },
  "edge": {
    "width": 2,
    "selectedWidth": 3,
    "bezierOffset": "100",
    "exec": {
      "color": "#ffffff"
    },
    "data": {
      "defaultColor": "#888888"
    }
  },
  "node": {
    "bg": "#2d2d3d",
    "minWidth": 200,
    "padding": 4,
    "border": {
      "color": "#3d3d4d",
      "width": 1,
      "radius": 8
    },
    "selected": {
      "borderColor": "#60a5fa",
      "borderWidth": 2
    },
    "header": {
      "height": 32
    },
    "pin": {
      "rowHeight": 28
    },
    "handle": {
      "radius": 6,
      "size": 12,
      "offset": "0"
    }
  },
  "nodeType": {
    "entry": "#22c55e",
    "entryMain": "#f59e0b",
    "return": "#ef4444",
    "returnMain": "#dc2626",
    "call": "#a855f7",
    "operation": "#3b82f6"
  },
  "overlay": {
    "bg": "#1f2937",
    "borderColor": "#4b5563",
    "borderWidth": 1,
    "borderRadius": 8,
    "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.3)",
    "padding": 8
  },
  "typeLabel": {
    "width": 60,
    "height": 16,
    "borderRadius": 3,
    "bgAlpha": "0.3",
    "textColor": "#ffffff",
    "fontSize": 10,
    "offsetFromHandle": 16
  },
  "canvas": {
    "bg": "#030712"
  },
  "minimap": {
    "nodeColor": "#4a5568",
    "selectedNodeColor": "#4299e1",
    "viewportColor": "rgba(66, 153, 225, 0.3)",
    "viewportBorderColor": "rgba(66, 153, 225, 0.8)",
    "bg": "#1a1a1a"
  },
  "text": {
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "title": {
      "size": 14,
      "color": "#ffffff",
      "weight": 600
    },
    "subtitle": {
      "size": 12,
      "color": "rgba(255,255,255,0.7)",
      "weight": 500
    },
    "label": {
      "size": 12,
      "color": "#d1d5db"
    },
    "muted": {
      "color": "#6b7280"
    }
  },
  "type": {
    "I1": "#E74C3C",
    "Index": "#50C878",
    "BF16": "#3498DB",
    "AnyType": "#F5F5F5",
    "unsignedInteger": "#50C878",
    "signlessInteger": "#52C878",
    "signedInteger": "#2D8659",
    "float": "#4A90D9",
    "tensorFloat": "#5BA3E8",
    "default": "#95A5A6"
  },
  "ui": {
    "listItemHeight": 28,
    "searchHeight": 28,
    "smallButtonHeight": 24,
    "rowHeight": 28,
    "labelWidth": 80,
    "gap": 8,
    "smallGap": 6,
    "scrollbarWidth": 6,
    "panelWidthNarrow": 240,
    "panelWidthMedium": 280,
    "panelMaxHeight": 320,
    "shadowBlur": 16,
    "shadowColor": "rgba(0, 0, 0, 0.5)",
    "darkBg": "#111827",
    "buttonBg": "#374151",
    "buttonHoverBg": "#4b5563",
    "successColor": "#22c55e",
    "successHoverColor": "#16a34a",
    "cursorBlinkInterval": "530",
    "minScrollbarHeight": 20,
    "closeButtonOffset": 24,
    "closeButtonSize": 10,
    "titleLeftPadding": 12,
    "colorDotRadius": 4,
    "colorDotGap": 16
  }
} as const;

export type Tokens = typeof tokens;
