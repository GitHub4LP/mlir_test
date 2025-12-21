/**
 * App Component
 * 
 * Main application entry point that renders the MainLayout.
 * The MainLayout handles all the core functionality including:
 * - Node editor canvas
 * - Node palette
 * - Function management
 * - Project management
 * - Execution panel
 */

import { useEffect, useMemo } from 'react';
import '@xyflow/react/dist/style.css';
import { MainLayout } from './app/MainLayout';
import { useTypeConstraintStore } from './stores/typeConstraintStore';
import { getStyleCSSVariables } from './editor/core/StyleSystem';

function App() {
  // 启动时加载类型约束数据
  const loadTypeConstraints = useTypeConstraintStore(state => state.loadTypeConstraints);
  
  useEffect(() => {
    loadTypeConstraints();
  }, [loadTypeConstraints]);
  
  // 获取 CSS Variables（主题变化时会更新）
  const cssVariables = useMemo(() => getStyleCSSVariables(), []);
  
  return (
    <div style={cssVariables}>
      <MainLayout />
    </div>
  );
}

export default App;
