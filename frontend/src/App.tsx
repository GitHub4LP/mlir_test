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

import { useEffect } from 'react';
import '@xyflow/react/dist/style.css';
import './editor/adapters/reactflow/styles/nodes.css';
import { MainLayout } from './app/MainLayout';
import { useTypeConstraintStore } from './stores/typeConstraintStore';

function App() {
  // 启动时加载类型约束数据
  const loadTypeConstraints = useTypeConstraintStore(state => state.loadTypeConstraints);
  
  useEffect(() => {
    loadTypeConstraints();
  }, [loadTypeConstraints]);
  
  return (
    <MainLayout />
  );
}

export default App;
