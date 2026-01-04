/**
 * Project toolbar component
 * 
 * Displays project actions (New, Open, Save) and Graph/Code view switcher.
 */

import type { Project } from '../../types';
import { useRendererStore, type ViewMode } from '../../stores/rendererStore';

export interface ProjectToolbarProps {
  project: Project | null;
  onCreateClick: () => void;
  onOpenClick: () => void;
  onSaveClick: () => void;
  /** 切换到 Code 视图后触发的加载回调（异步执行，不阻塞切换） */
  onSwitchToCode?: () => void;
}

export function ProjectToolbar({
  project,
  onCreateClick,
  onOpenClick,
  onSaveClick,
  onSwitchToCode,
}: ProjectToolbarProps) {
  const viewMode = useRendererStore(state => state.viewMode);
  const setViewMode = useRendererStore(state => state.setViewMode);

  const handleViewSwitch = (mode: ViewMode) => {
    if (mode === viewMode) return;
    
    // 立即切换视图
    setViewMode(mode);
    
    // 切换到 Code 时，异步触发加载（不阻塞 UI）
    if (mode === 'code' && onSwitchToCode) {
      // 使用 setTimeout 确保视图先切换
      setTimeout(() => onSwitchToCode(), 0);
    }
  };

  return (
    <div className="h-10 bg-gray-800 border-b border-gray-700 flex items-center px-4 gap-4">
      {/* Project Actions */}
      <div className="flex items-center gap-1">
        <button
          onClick={onCreateClick}
          className="px-2 py-1 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors flex items-center gap-1"
          title="Create new project"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New
        </button>

        <button
          onClick={onOpenClick}
          className="px-2 py-1 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors flex items-center gap-1"
          title="Open existing project"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
          </svg>
          Open
        </button>

        <button
          onClick={onSaveClick}
          disabled={!project}
          className="px-2 py-1 text-xs text-gray-300 hover:text-white hover:bg-gray-700 rounded transition-colors flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
          title="Save project"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
          </svg>
          Save
        </button>
      </div>

      {/* Separator */}
      <div className="h-5 w-px bg-gray-600" />

      {/* Current Project Info */}
      {project && (
        <div className="flex items-center gap-2 text-xs">
          <span className="text-gray-400">{project.name}</span>
          <span className="text-gray-600 text-[10px]">{project.path}</span>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* View Switcher - 最右侧 */}
      <div className="flex items-center rounded overflow-hidden border border-gray-600">
        <button
          onClick={() => handleViewSwitch('graph')}
          className={`px-3 py-1 text-xs font-medium transition-colors ${
            viewMode === 'graph'
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          Graph
        </button>
        <button
          onClick={() => handleViewSwitch('code')}
          className={`px-3 py-1 text-xs font-medium transition-colors ${
            viewMode === 'code'
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          Code
        </button>
      </div>
    </div>
  );
}
