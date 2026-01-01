/**
 * UIManager - Canvas UI 管理器
 * 
 * 统一管理所有 Canvas UI 组件（TypeSelector、EditableName、AttributeEditor 等）。
 * 负责：
 * - UI 组件的生命周期管理
 * - 事件路由（UI 组件优先）
 * - 渲染协调
 */

import type { UIMouseEvent, UIKeyEvent, UIWheelEvent } from './ui/UIComponent';
import { TypeSelector, type TypeOption, type ConstraintData } from './ui/TypeSelector';
import { EditableName } from './ui/EditableName';
import { AttributeEditor, type AttributeDefinition } from './ui/AttributeEditor';

export interface TypeSelectorState {
  visible: boolean;
  nodeId: string;
  handleId: string;
  screenX: number;
  screenY: number;
  options: TypeOption[];
  currentType?: string;
  constraintData?: ConstraintData;
}

export interface EditableNameState {
  visible: boolean;
  nodeId: string;
  fieldId: string;
  screenX: number;
  screenY: number;
  width: number;
  value: string;
  placeholder?: string;
}

export interface AttributeEditorState {
  visible: boolean;
  nodeId: string;
  screenX: number;
  screenY: number;
  attributes: AttributeDefinition[];
  title?: string;
}

export interface UIManagerCallbacks {
  onTypeSelect?: (nodeId: string, handleId: string, type: string) => void;
  onTypeSelectorClose?: () => void;
  onNameChange?: (nodeId: string, fieldId: string, value: string) => void;
  onNameSubmit?: (nodeId: string, fieldId: string, value: string) => void;
  onEditableNameClose?: () => void;
  onAttributeChange?: (nodeId: string, attrName: string, value: unknown) => void;
  onAttributeEditorClose?: () => void;
}

/**
 * Canvas UI 管理器
 */
export class UIManager {
  private typeSelector: TypeSelector;
  private typeSelectorState: TypeSelectorState = {
    visible: false,
    nodeId: '',
    handleId: '',
    screenX: 0,
    screenY: 0,
    options: [],
  };
  
  private editableName: EditableName;
  private editableNameState: EditableNameState = {
    visible: false,
    nodeId: '',
    fieldId: '',
    screenX: 0,
    screenY: 0,
    width: 80,
    value: '',
  };
  
  private attributeEditor: AttributeEditor;
  private attributeEditorState: AttributeEditorState = {
    visible: false,
    nodeId: '',
    screenX: 0,
    screenY: 0,
    attributes: [],
  };
  
  private callbacks: UIManagerCallbacks = {};
  
  constructor() {
    this.typeSelector = new TypeSelector('type-selector');
    this.typeSelector.visible = false;
    
    this.editableName = new EditableName('editable-name');
    this.editableName.visible = false;
    
    this.attributeEditor = new AttributeEditor('attribute-editor');
    this.attributeEditor.visible = false;
    
    // 设置 TypeSelector 回调
    this.typeSelector.setOnSelect((type) => {
      const { nodeId, handleId } = this.typeSelectorState;
      this.callbacks.onTypeSelect?.(nodeId, handleId, type);
      this.hideTypeSelector();
    });
    
    this.typeSelector.setOnClose(() => {
      this.hideTypeSelector();
    });
    
    // 设置 EditableName 回调
    this.editableName.setOnChange((value) => {
      const { nodeId, fieldId } = this.editableNameState;
      this.callbacks.onNameChange?.(nodeId, fieldId, value);
    });
    
    this.editableName.setOnSubmit((value) => {
      const { nodeId, fieldId } = this.editableNameState;
      this.callbacks.onNameSubmit?.(nodeId, fieldId, value);
      this.hideEditableName();
    });
    
    this.editableName.setOnCancel(() => {
      this.hideEditableName();
    });
    
    // 设置 AttributeEditor 回调
    this.attributeEditor.setOnChange((name, value) => {
      const { nodeId } = this.attributeEditorState;
      this.callbacks.onAttributeChange?.(nodeId, name, value);
    });
    
    this.attributeEditor.setOnClose(() => {
      this.hideAttributeEditor();
    });
  }
  
  /**
   * 挂载到容器（用于隐藏 input 等 DOM 元素）
   */
  mount(container: HTMLElement): void {
    this.typeSelector.mount(container);
    this.editableName.mount(container);
    this.attributeEditor.mount(container);
  }
  
  /**
   * 卸载
   */
  unmount(): void {
    this.typeSelector.unmount();
    this.editableName.unmount();
    this.attributeEditor.unmount();
  }
  
  /**
   * 设置回调
   */
  setCallbacks(callbacks: UIManagerCallbacks): void {
    this.callbacks = callbacks;
  }
  
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
    this.typeSelectorState = {
      visible: true,
      nodeId,
      handleId,
      screenX,
      screenY,
      options,
      currentType,
      constraintData,
    };
    
    // 设置约束数据（优先使用新 API）
    if (constraintData) {
      this.typeSelector.setConstraintData(constraintData);
      this.typeSelector.setCurrentType(currentType || '');
    } else {
      // 兼容旧 API
      this.typeSelector.setOptions(options);
    }
    
    this.typeSelector.setPosition(screenX, screenY);
    this.typeSelector.show();
  }
  
  /**
   * 隐藏类型选择器
   */
  hideTypeSelector(): void {
    this.typeSelectorState.visible = false;
    this.typeSelector.hide();
    this.callbacks.onTypeSelectorClose?.();
  }
  
  /**
   * 类型选择器是否可见
   */
  isTypeSelectorVisible(): boolean {
    return this.typeSelectorState.visible;
  }
  
  /**
   * 获取类型选择器状态
   */
  getTypeSelectorState(): TypeSelectorState {
    return { ...this.typeSelectorState };
  }
  
  /**
   * 渲染所有 UI 组件
   */
  render(ctx: CanvasRenderingContext2D): void {
    // 渲染 EditableName（如果可见）
    if (this.editableName.visible) {
      this.editableName.render(ctx);
    }
    
    // 渲染 AttributeEditor（如果可见）
    if (this.attributeEditor.visible) {
      this.attributeEditor.render(ctx);
    }
    
    // 渲染 TypeSelector（最上层）
    if (this.typeSelector.visible) {
      this.typeSelector.render(ctx);
    }
  }
  
  // ============================================================
  // EditableName 方法
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
    this.editableNameState = {
      visible: true,
      nodeId,
      fieldId,
      screenX,
      screenY,
      width,
      value,
      placeholder,
    };
    
    this.editableName.setPosition(screenX, screenY);
    this.editableName.setSize(width, this.editableName.getBounds().height);
    this.editableName.setValue(value);
    if (placeholder) {
      this.editableName.setPlaceholder(placeholder);
    }
    this.editableName.visible = true;
    this.editableName.startEdit();
  }
  
  /**
   * 隐藏可编辑名称
   */
  hideEditableName(): void {
    if (this.editableName.isEditing()) {
      this.editableName.cancel();
    }
    this.editableNameState.visible = false;
    this.editableName.visible = false;
    this.callbacks.onEditableNameClose?.();
  }
  
  /**
   * 可编辑名称是否可见
   */
  isEditableNameVisible(): boolean {
    return this.editableNameState.visible;
  }
  
  /**
   * 获取可编辑名称状态
   */
  getEditableNameState(): EditableNameState {
    return { ...this.editableNameState };
  }
  
  // ============================================================
  // AttributeEditor 方法
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
    this.attributeEditorState = {
      visible: true,
      nodeId,
      screenX,
      screenY,
      attributes,
      title,
    };
    
    this.attributeEditor.setPosition(screenX, screenY);
    if (title) {
      this.attributeEditor.setTitle(title);
    }
    this.attributeEditor.setAttributes(attributes);
    this.attributeEditor.show();
  }
  
  /**
   * 隐藏属性编辑器
   */
  hideAttributeEditor(): void {
    this.attributeEditorState.visible = false;
    this.attributeEditor.hide();
    this.callbacks.onAttributeEditorClose?.();
  }
  
  /**
   * 属性编辑器是否可见
   */
  isAttributeEditorVisible(): boolean {
    return this.attributeEditorState.visible;
  }
  
  /**
   * 获取属性编辑器状态
   */
  getAttributeEditorState(): AttributeEditorState {
    return { ...this.attributeEditorState };
  }
  
  // ============================================================
  // 事件处理
  // ============================================================
  
  /**
   * 处理鼠标按下事件
   * @returns true 如果事件被 UI 组件处理
   */
  handleMouseDown(event: UIMouseEvent): boolean {
    // TypeSelector 优先级最高
    if (this.typeSelector.visible) {
      // 点击外部关闭
      if (!this.typeSelector.hitTest(event.x, event.y)) {
        this.hideTypeSelector();
        return true; // 消费事件
      }
      return this.typeSelector.onMouseDown?.(event) ?? false;
    }
    
    // AttributeEditor
    if (this.attributeEditor.visible) {
      if (!this.attributeEditor.hitTest(event.x, event.y)) {
        this.hideAttributeEditor();
        return true;
      }
      return this.attributeEditor.onMouseDown(event);
    }
    
    // EditableName
    if (this.editableName.visible) {
      const handled = this.editableName.onMouseDown(event);
      if (handled) return true;
      // 点击外部会自动提交（在 EditableName 内部处理）
    }
    
    return false;
  }
  
  /**
   * 处理鼠标移动事件
   */
  handleMouseMove(event: UIMouseEvent): boolean {
    let handled = false;
    
    if (this.typeSelector.visible) {
      handled = this.typeSelector.onMouseMove?.(event) ?? false;
    }
    
    if (this.attributeEditor.visible) {
      handled = this.attributeEditor.onMouseMove(event) || handled;
    }
    
    if (this.editableName.visible) {
      handled = this.editableName.onMouseMove(event) || handled;
    }
    
    return handled;
  }
  
  /**
   * 处理鼠标抬起事件
   */
  handleMouseUp(event: UIMouseEvent): boolean {
    if (this.typeSelector.visible) {
      return this.typeSelector.onMouseUp?.(event) ?? false;
    }
    return false;
  }
  
  /**
   * 处理滚轮事件
   */
  handleWheel(event: UIWheelEvent): boolean {
    if (this.typeSelector.visible && this.typeSelector.hitTest(event.x, event.y)) {
      return this.typeSelector.onWheel?.(event) ?? false;
    }
    return false;
  }
  
  /**
   * 处理键盘按下事件
   */
  handleKeyDown(event: UIKeyEvent): boolean {
    if (this.typeSelector.visible) {
      return this.typeSelector.onKeyDown?.(event) ?? false;
    }
    if (this.attributeEditor.visible) {
      return this.attributeEditor.onKeyDown(event);
    }
    return false;
  }
  

  
  /**
   * 销毁
   */
  dispose(): void {
    this.typeSelector.dispose();
    this.editableName.dispose();
    this.attributeEditor.dispose();
  }
}
