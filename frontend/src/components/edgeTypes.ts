/**
 * Edge Types Configuration
 * 
 * Re-exports from the React Flow adapter for backward compatibility.
 * @deprecated Import from '../editor/adapters/reactflow' instead.
 */

export { edgeTypes } from '../editor/adapters/reactflow';

// 为了兼容 default import
import { edgeTypes as _edgeTypes } from '../editor/adapters/reactflow';
export default _edgeTypes;
