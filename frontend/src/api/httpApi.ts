// 실제 FastAPI 백엔드 클라이언트 (Live 모드)
// 실패하면 ApiError를 던집니다. Mock 결과로 대체하지 않습니다.
import { RleBackedMask } from '../lib/lazyMask';
import { rleDecode, type RleMask } from '../lib/rle';
import type { BackendStatus, ErrorCode, InferenceMetrics, Job, Mask, RemoteStatus, VideoMeta } from '../types';
import { ApiError } from './errors';
import type { GpuJobInfo, StudioApi, TrackComparison } from './types';

interface HealthResponse {
  capabilities: { segmentation: boolean; tracking: boolean; export: boolean };
  tracking?: { maxFrames: number; lastAvgFrameMs: number | null };
  ffmpeg?: { version: string; source: string } | null;
  gpuImport?: BackendStatus['gpuImport'];
  remote?: RemoteStatus;
  model: {
    state: BackendStatus['modelState'];
    error: string | null;
    loadMs: number | null;
    warmup: { setImageMs: number; predictMs: number } | null;
    device: null | {
      device: 'cuda' | 'cpu';
      reason: string;
      torchVersion: string;
      gpu: { name: string | null };
    };
  };
}

interface SegmentResponse {
  mask: RleMask;
  metrics: InferenceMetrics;
  candidates?: { mask: RleMask; score: number; coverage: number }[];
  selected?: number;
}

interface TrackMasksResponse {
  width: number;
  height: number;
  frames: { index: number; counts: number[] }[];
}

const pointsBody = (points: { x: number; y: number; label: 0 | 1 }[]) => points.map(({ x, y, label }) => ({ x, y, label }));


export const TIMEOUTS = {
  default: 15_000,
  health: 5_000,
  upload: 180_000,
  /** CPU에서 새 프레임 인코딩은 5~15초 걸릴 수 있음 */
  segment: 120_000,
  /** 원격 GPU: 첫 호출은 영상 업로드 + 콜드 스타트(컨테이너 시작·모델 로드)가 포함됨 */
  remote: 240_000,
};

export function createHttpApi(base = '/api', fetchImpl: typeof fetch = (...a) => fetch(...a)): StudioApi {
  /** 모든 요청에 타임아웃을 둡니다 — 백엔드/프록시가 응답하지 않아도 UI가 멈추지 않도록 */
  async function request<T>(path: string, init: RequestInit = {}, timeoutMs = TIMEOUTS.default): Promise<T> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    let res: Response;
    try {
      res = await fetchImpl(base + path, { ...init, signal: ctrl.signal });
    } catch {
      throw ctrl.signal.aborted
        ? new ApiError('network', `The backend did not respond within ${Math.round(timeoutMs / 1000)}s.`)
        : new ApiError('network', 'Could not reach the backend server.');
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) {
      let code: ErrorCode = res.status === 422 ? 'validation' : 'internal';
      let message = `Request failed (HTTP ${res.status})`;
      try {
        const body = await res.json();
        if (body?.error?.code) {
          code = body.error.code;
          message = body.error.message;
        } else if (body?.detail) {
          // FastAPI 기본 검증 오류 형식
          message = typeof body.detail === 'string' ? body.detail : 'Invalid request.';
        }
      } catch {
        // JSON 오류 본문이 없는 5xx = 백엔드가 아닌 프록시의 응답 (Vite 개발 프록시는 연결 실패 시 빈 500을 반환)
        if (res.status >= 500) {
          throw new ApiError('network', 'Could not reach the backend server.');
        }
      }
      throw new ApiError(code, message);
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  }

  return {
    mode: 'live',
    capabilities: { tracking: true, export: true, gpuJobs: true },

    async getStatus() {
      const h = await request<HealthResponse>('/health', {}, TIMEOUTS.health);
      const d = h.model.device;
      return {
        mode: 'live',
        modelState: h.model.state,
        modelError: h.model.error,
        device: d?.device,
        deviceReason: d?.reason,
        deviceName: d ? (d.device === 'cuda' ? d.gpu.name : 'CPU') : undefined,
        torchVersion: d?.torchVersion,
        loadMs: h.model.loadMs,
        warmup: h.model.warmup,
        tracking: h.tracking,
        ffmpeg: h.ffmpeg ?? null,
        gpuImport: h.gpuImport,
        remote: h.remote ?? { configured: false },
      };
    },

    async createSession(file) {
      // 사전 확인: 개발 프록시는 백엔드가 꺼져 있을 때 대용량 업로드 요청을 끝내지 못하므로 먼저 상태를 확인
      await request('/health', {}, TIMEOUTS.health);
      const fd = new FormData();
      fd.append('file', file, file.name);
      return request<{ sessionId: string; meta: VideoMeta }>('/sessions', { method: 'POST', body: fd }, TIMEOUTS.upload);
    },

    async segmentFrame(sessionId, req) {
      const t0 = performance.now();
      const r = await request<SegmentResponse>(`/sessions/${sessionId}/segment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frameIndex: req.frameIndex,
          points: pointsBody(req.points),
          maskWidth: req.maskWidth,
          maskHeight: req.maskHeight,
          ...(req.backend ? { backend: req.backend } : {}),
        }),
      }, req.backend === 'local' ? TIMEOUTS.segment : TIMEOUTS.remote);
      const mask = rleDecode(r.mask);
      const candidates = r.candidates?.map((c) => ({ mask: rleDecode(c.mask), score: c.score, coverage: c.coverage }));
      for (const m of [mask, ...(candidates ?? []).map((c) => c.mask)]) {
        if (m.width !== req.maskWidth || m.height !== req.maskHeight) {
          throw new ApiError('internal', 'Backend returned a mask with unexpected dimensions.');
        }
      }
      const metrics = { ...r.metrics, roundTripMs: Math.round(performance.now() - t0) };
      return candidates?.length ? { mask, metrics, candidates, selected: r.selected ?? 0 } : { mask, metrics };
    },

    async startPropagation(sessionId, req) {
      return request<Job>(`/sessions/${sessionId}/propagate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frameIndex: req.frameIndex,
          points: pointsBody(req.points),
          maxFrames: req.maxFrames,
          ...(req.backend ? { backend: req.backend } : {}),
          ...(req.maskCandidate !== undefined ? { maskCandidate: req.maskCandidate } : {}),
        }),
      });
    },
    async warmupRemote() {
      return request<RemoteStatus>('/remote/warmup', { method: 'POST' }, TIMEOUTS.remote);
    },
    async getJob(jobId) {
      return request<Job>(`/jobs/${jobId}`);
    },
    async cancelJob(jobId) {
      await request(`/jobs/${jobId}/cancel`, { method: 'POST' });
    },
    async fetchTrackedMasks(sessionId, jobId, size) {
      const r = await request<TrackMasksResponse>(
        `/sessions/${sessionId}/tracks/${jobId}/masks?width=${size.width}&height=${size.height}`,
        {},
        TIMEOUTS.segment,
      );
      // RLE 로 보관하고 표시할 때만 디코딩 (300프레임도 메모리 부담 없음)
      const masks = new Map<number, Mask>();
      for (const f of r.frames) masks.set(f.index, new RleBackedMask({ width: r.width, height: r.height, counts: f.counts }));
      return masks;
    },
    async startExport(sessionId, req) {
      return request<Job>(`/sessions/${sessionId}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trackJobId: req.trackJobId, effects: req.effects }),
      });
    },
    async createGpuJob(sessionId, req) {
      return request<GpuJobInfo>(
        `/sessions/${sessionId}/gpu-jobs`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            frameIndex: req.frameIndex,
            points: pointsBody(req.points),
            maxFrames: req.maxFrames,
            precision: req.precision,
            ...(req.maskCandidate !== undefined ? { maskCandidate: req.maskCandidate } : {}),
          }),
        },
        TIMEOUTS.upload,
      );
    },
    async importGpuResult(sessionId, file) {
      const fd = new FormData();
      fd.append('file', file, file.name);
      return request<Job>(`/sessions/${sessionId}/gpu-results`, { method: 'POST', body: fd }, TIMEOUTS.upload);
    },
    async compareTracks(sessionId, a, b) {
      return request<TrackComparison>(`/sessions/${sessionId}/tracks/compare?a=${a}&b=${b}`, {}, TIMEOUTS.segment);
    },
    async deleteSession(sessionId) {
      await request(`/sessions/${sessionId}`, { method: 'DELETE' }).catch(() => undefined);
    },
  };
}
