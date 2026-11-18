import { createHttpApi } from './httpApi';
import { createMockApi } from './mockApi';
import { resolveApiMode } from './mode';
import type { StudioApi } from './types';

const mode = resolveApiMode(typeof window !== 'undefined' ? window.location.search : '', import.meta.env.VITE_API_MODE);

export const api: StudioApi = mode === 'mock' ? createMockApi() : createHttpApi();

export type { StudioApi, SegmentRequest, TrackRequest, ExportRequest, GpuJobRequest, GpuJobInfo, TrackComparison } from './types';
