// ⚠️ Phase 1 전용 Mock API — 모델 추론/추적을 수행하지 않습니다.
// UI 흐름(업로드 → 클릭 → 마스크 → 추적 → export, 진행률/오류)을 검증하기 위한 시뮬레이션입니다.
// 오류 UI 테스트: URL에 ?mockFail=segment | track | oom | export 를 붙이세요.
import type { Job, JobError, JobKind, Mask, VideoMeta } from '../types';
import { mockMaskFromPoints } from '../lib/mask';
import { ApiError } from './errors';
import type { StudioApi } from './types';

const MOCK_FPS = 30;

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
const uid = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : Math.random().toString(36).slice(2);

interface MockSession {
  meta: VideoMeta;
  promptMask: Mask | null;
  trackedMask: Mask | null;
}

interface MockJobState {
  job: Job;
  timer: ReturnType<typeof setInterval> | null;
}

export function createMockApi(): StudioApi {
  const failMode = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('mockFail') : null;
  const sessions = new Map<string, MockSession>();
  const jobs = new Map<string, MockJobState>();

  const getSession = (id: string) => {
    const s = sessions.get(id);
    if (!s) throw new ApiError('not_found', 'Session not found.');
    return s;
  };

  function runJob(
    kind: JobKind,
    durationMs: number,
    describe: (progress: number) => string,
    opts: { failAt?: number; failError?: JobError; onSuccess?: () => Partial<Job> } = {},
  ): Job {
    const id = uid();
    const state: MockJobState = { job: { id, kind, status: 'queued', progress: 0, message: 'Queued' }, timer: null };
    jobs.set(id, state);
    const start = performance.now() + 300; // 큐 대기 시뮬레이션
    state.timer = setInterval(() => {
      const j = state.job;
      const stop = () => {
        if (state.timer) clearInterval(state.timer);
        state.timer = null;
      };
      if (j.status === 'cancelled') return stop();
      const t = (performance.now() - start) / durationMs;
      if (t < 0) return;
      const progress = Math.min(1, t);
      if (opts.failAt !== undefined && progress >= opts.failAt) {
        state.job = { ...j, status: 'failed', progress, error: opts.failError, message: 'Failed' };
        return stop();
      }
      if (progress >= 1) {
        state.job = { ...j, status: 'succeeded', progress: 1, message: 'Done', ...opts.onSuccess?.() };
        return stop();
      }
      state.job = { ...j, status: 'running', progress, message: describe(progress) };
    }, 100);
    return { ...state.job };
  }

  return {
    mode: 'mock',
    capabilities: { tracking: true, export: true, gpuJobs: false },

    async getStatus() {
      return { mode: 'mock', modelState: 'ready', deviceName: 'Mock (no model)' };
    },

    async createSession(_file, info) {
      await delay(400);
      const meta: VideoMeta = {
        ...info,
        fps: MOCK_FPS,
        frameCount: Math.max(1, Math.round(info.duration * MOCK_FPS)),
        fpsSource: 'assumed',
      };
      const sessionId = uid();
      sessions.set(sessionId, { meta, promptMask: null, trackedMask: null });
      return { sessionId, meta };
    },

    async segmentFrame(sessionId, req) {
      const s = getSession(sessionId);
      await delay(220);
      if (failMode === 'segment') throw new ApiError('internal', 'Mock segmentation failure (mockFail=segment).');
      const mask = mockMaskFromPoints(req.points, req.maskWidth, req.maskHeight);
      s.promptMask = mask;
      s.trackedMask = null;
      return { mask, metrics: { device: 'mock' } };
    },

    async startPropagation(sessionId) {
      const s = getSession(sessionId);
      if (!s.promptMask) throw new ApiError('validation', 'Select an object before tracking.');
      const n = s.meta.frameCount;
      const failing = failMode === 'track' || failMode === 'oom';
      return runJob(
        'propagate',
        Math.min(4500, Math.max(1500, n * 12)),
        (p) => `Simulated propagation · frame ${Math.floor(p * n)}/${n}`,
        {
          failAt: failing ? 0.4 : undefined,
          failError:
            failMode === 'oom'
              ? { code: 'gpu_oom', message: 'CUDA out of memory (simulated)' }
              : { code: 'internal', message: 'Mock tracking failure (mockFail=track).' },
          onSuccess: () => {
            s.trackedMask = s.promptMask;
            return { message: `Simulated: the same mask was copied to ${n} frames (no real tracking in mock mode)` };
          },
        },
      );
    },

    async getJob(jobId) {
      await delay(30);
      const state = jobs.get(jobId);
      if (!state) throw new ApiError('not_found', 'Job not found.');
      return { ...state.job };
    },

    async cancelJob(jobId) {
      const state = jobs.get(jobId);
      if (state && (state.job.status === 'queued' || state.job.status === 'running')) {
        state.job = { ...state.job, status: 'cancelled', message: 'Cancelled' };
      }
    },

    async fetchTrackedMasks(sessionId) {
      const s = getSession(sessionId);
      await delay(150);
      if (!s.trackedMask) throw new ApiError('not_found', 'No tracking result available.');
      // Mock: 모든 프레임에 같은 마스크 (실제 추적 아님). 동일 객체를 공유하므로 메모리 부담 없음.
      const result = new Map<number, Mask>();
      for (let f = 0; f < s.meta.frameCount; f++) result.set(f, s.trackedMask);
      return result;
    },

    async startExport(sessionId) {
      const s = getSession(sessionId);
      if (!s.trackedMask) throw new ApiError('validation', 'Track the object before exporting.');
      const n = s.meta.frameCount;
      return runJob('export', 2500, (p) => `Simulated rendering · frame ${Math.floor(p * n)}/${n}`, {
        failAt: failMode === 'export' ? 0.6 : undefined,
        failError: { code: 'internal', message: 'Mock export failure (mockFail=export).' },
        onSuccess: () => ({ resultUrl: null, message: 'Mock export finished — real MP4 rendering arrives in Phase 4.' }),
      });
    },

    async deleteSession(sessionId) {
      sessions.delete(sessionId);
    },

    async createGpuJob() {
      throw new ApiError('not_implemented', 'GPU jobs need the live backend (remove ?mode=mock).');
    },
    async importGpuResult() {
      throw new ApiError('not_implemented', 'GPU jobs need the live backend (remove ?mode=mock).');
    },
    async compareTracks() {
      throw new ApiError('not_implemented', 'Not available in mock mode.');
    },
    async warmupRemote() {
      throw new ApiError('not_implemented', 'Remote GPU needs the live backend (remove ?mode=mock).');
    },
  };
}
