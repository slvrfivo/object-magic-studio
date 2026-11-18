// 앱 전역 상태 (zustand)
import { create } from 'zustand';
import { api, type GpuJobInfo, type TrackComparison } from '../api';
import { ApiError, describeError, describeJobError } from '../api/errors';
import { DEFAULT_EFFECTS } from '../lib/effectParams';
import { validateFileBasics, validateVideoInfo } from '../lib/validation';
import { generateThumbnails, loadVideoInfo, previewSize } from '../lib/video';
import type {
  BackendStatus,
  ComputeBackend,
  MaskCandidate,
  EffectId,
  InferenceMetrics,
  EffectSettings,
  Job,
  Mask,
  PointLabel,
  PromptPoint,
  VideoMeta,
  ViewMode,
} from '../types';
import { TERMINAL_JOB_STATUSES } from '../types';

export interface LoadedVideo {
  file: File;
  url: string;
  meta: VideoMeta;
  previewWidth: number;
  previewHeight: number;
}

export type ToastKind = 'info' | 'success' | 'error';
export interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  message?: string;
}

export type Tool = 'positive' | 'negative';

/** 이 세션에서 만든/가져온 추적 결과 (전환 가능, 효과 변경 시 재추론 없음) */
export interface TrackEntry {
  job: Job;
  label: string;
}

export function trackLabel(job: Job): string {
  const st = job.stats;
  if (st?.source === 'gpu-import') return `${st.gpuName ?? st.device ?? 'GPU'} · ${st.precision ?? '?'}`;
  if (st?.source === 'remote-gpu') return `Remote ${st.gpuName ?? 'GPU'} · ${st.precision ?? 'fp16'}`;
  return 'Local CPU · fp32';
}

interface StudioState {
  backendStatus: BackendStatus | null;
  /** Live 모드에서 백엔드 상태 조회 실패 메시지 */
  backendError: string | null;
  /** 분할·추적 실행 위치. 원격 GPU 가 설정되어 있으면 기본 remote, 사용자가 바꾸면 유지 */
  compute: ComputeBackend;
  computeChosen: boolean;
  remoteWarming: boolean;
  uploadState: 'idle' | 'validating';
  video: LoadedVideo | null;
  sessionId: string | null;
  thumbnails: string[];

  currentFrame: number;
  isPlaying: boolean;

  tool: Tool;
  points: PromptPoint[];
  promptFrame: number | null;
  currentMask: Mask | null;
  /** 한 번 클릭의 마스크 후보 3개 (여러 번 클릭하면 null) */
  candidates: MaskCandidate[] | null;
  selectedCandidate: number;
  /** SAM 이 고른 기본 후보 (점수 최고) */
  bestCandidate: number;
  isSegmenting: boolean;
  /** 마지막 분할의 측정값 (장치, 시간, 메모리) */
  lastInference: InferenceMetrics | null;
  /** 이미지 임베딩이 계산된 프레임 (같은 프레임 재클릭 시 빠름) */
  encodedFrame: number | null;

  trackedMasks: Map<number, Mask> | null;
  trackingJob: Job | null;
  /** 한 번에 추적할 프레임 수 (프롬프트 프레임 포함, 정방향) */
  trackFrames: number;
  trackHistory: TrackEntry[];

  gpuPrecision: 'fp16' | 'fp32';
  gpuJob: GpuJobInfo | null;
  gpuBusy: 'creating' | 'importing' | null;
  comparison: (TrackComparison & { a: string; b: string }) | null;

  effects: EffectSettings;
  exportJob: Job | null;

  viewMode: ViewMode;
  splitPosition: number;
  showMask: boolean;

  toasts: Toast[];

  refreshBackendStatus(): Promise<void>;
  loadVideo(file: File): Promise<void>;
  closeVideo(): void;
  setCurrentFrame(frame: number): void;
  stepFrame(delta: number): void;
  setPlaying(playing: boolean): void;
  setTool(tool: Tool): void;
  addPoint(p: { x: number; y: number }, label: PointLabel): Promise<void>;
  removePoint(id: string): Promise<void>;
  undoPoint(): Promise<void>;
  clearPoints(): void;
  setTrackFrames(n: number): void;
  selectCandidate(index: number): void;
  setCompute(c: ComputeBackend): void;
  warmupRemote(): Promise<void>;
  selectTrack(jobId: string): Promise<void>;
  setGpuPrecision(p: 'fp16' | 'fp32'): void;
  createGpuJob(): Promise<GpuJobInfo | null>;
  importGpuResult(file: File): Promise<void>;
  compareTracks(a: string, b: string): Promise<void>;
  startTracking(): Promise<void>;
  cancelJob(kind: 'propagate' | 'export'): Promise<void>;
  updateEffect<K extends EffectId>(id: K, patch: Partial<EffectSettings[K]>): void;
  startExport(): Promise<void>;
  setViewMode(mode: ViewMode): void;
  setSplitPosition(pos: number): void;
  setShowMask(show: boolean): void;
  pushToast(t: Omit<Toast, 'id'>): void;
  dismissToast(id: number): void;
}

/** 현재 프레임에 표시할 마스크: 추적 결과 우선, 없으면 프롬프트 프레임의 단일 프레임 마스크 */
export function maskForFrame(
  s: Pick<StudioState, 'trackedMasks' | 'promptFrame' | 'currentMask'>,
  frame: number,
): Mask | null {
  const tracked = s.trackedMasks?.get(frame);
  if (tracked) return tracked;
  if (s.promptFrame === frame) return s.currentMask;
  return null;
}

export const isJobActive = (job: Job | null) => !!job && !TERMINAL_JOB_STATUSES.has(job.status);

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** 서버가 만든 파일을 브라우저 다운로드로 저장 */
function triggerDownload(url: string) {
  if (typeof document === 'undefined') return;
  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** 작업 상태 폴링. 일시적인 네트워크 오류는 몇 번까지 재시도합니다 (작업은 서버에서 계속 실행 중). */
async function pollJob(jobId: string, onUpdate: (j: Job) => void): Promise<Job> {
  const interval = api.mode === 'live' ? 1000 : 250;
  let failures = 0;
  for (;;) {
    try {
      const job = await api.getJob(jobId);
      failures = 0;
      onUpdate(job);
      if (TERMINAL_JOB_STATUSES.has(job.status)) return job;
    } catch (e) {
      if (!(e instanceof ApiError) || e.code !== 'network' || ++failures > 3) throw e;
    }
    await sleep(interval);
  }
}

let segmentSeq = 0;
let toastSeq = 0;
const newId = () => Math.random().toString(36).slice(2, 10);

const sessionDefaults = {
  thumbnails: [] as string[],
  currentFrame: 0,
  isPlaying: false,
  points: [] as PromptPoint[],
  promptFrame: null as number | null,
  currentMask: null as Mask | null,
  candidates: null as MaskCandidate[] | null,
  selectedCandidate: 0,
  bestCandidate: 0,
  isSegmenting: false,
  lastInference: null as InferenceMetrics | null,
  encodedFrame: null as number | null,
  trackedMasks: null as Map<number, Mask> | null,
  trackingJob: null as Job | null,
  exportJob: null as Job | null,
  trackHistory: [] as TrackEntry[],
  gpuJob: null as GpuJobInfo | null,
  gpuBusy: null as 'creating' | 'importing' | null,
  comparison: null as (TrackComparison & { a: string; b: string }) | null,
};

export const useStudioStore = create<StudioState>((set, get) => {
  const toast = (kind: ToastKind, title: string, message?: string) => get().pushToast({ kind, title, message });

  /** 프롬프트가 바뀌면 기존 추적/export 결과는 무효 */
  const invalidateTracking = () => {
    const s = get();
    if (isJobActive(s.trackingJob)) void api.cancelJob(s.trackingJob!.id);
    if (s.trackedMasks) toast('info', 'Tracking reset', 'Prompts changed — run tracking again.');
    set({ trackedMasks: null, trackingJob: null, exportJob: null });
  };

  type PromptSnapshot = Pick<StudioState, 'points' | 'promptFrame' | 'currentMask' | 'candidates' | 'selectedCandidate' | 'bestCandidate'>;

  /**
   * 현재 점들로 분할 요청. 실패하면 직전 상태(rollback)로 되돌려
   * "점은 추가됐는데 마스크는 예전 것" 같은 불일치나 다른 프레임의 마스크가 남는 일을 막습니다.
   */
  const segment = async (rollback?: PromptSnapshot) => {
    const s = get();
    if (!s.video || !s.sessionId || s.promptFrame === null) return;
    const seq = ++segmentSeq;
    if (!s.points.some((p) => p.label === 1)) {
      set({ currentMask: null, candidates: null, isSegmenting: false });
      return;
    }
    set({ isSegmenting: true });
    try {
      const frameIndex = s.promptFrame;
      const { mask, metrics, candidates, selected } = await api.segmentFrame(s.sessionId, {
        frameIndex: s.promptFrame,
        points: s.points,
        maskWidth: s.video.previewWidth,
        maskHeight: s.video.previewHeight,
        backend: s.compute,
      });
      if (metrics.backend === 'remote') void get().refreshBackendStatus(); // 사용량·비용 표시 갱신
      if (seq === segmentSeq)
        set({
          currentMask: mask,
          // 한 번 클릭: 후보 3개 보관, 기본 선택은 SAM 점수 최고 후보
          candidates: candidates ?? null,
          selectedCandidate: selected ?? 0,
          bestCandidate: selected ?? 0,
          isSegmenting: false,
          lastInference: metrics,
          encodedFrame: frameIndex,
        });
    } catch (e) {
      if (seq !== segmentSeq) return;
      set({ isSegmenting: false, ...(rollback ?? { currentMask: null, candidates: null }) });
      toast('error', 'Segmentation failed', describeError(e));
    }
  };

  const guardEditable = () => {
    if (isJobActive(get().trackingJob)) {
      toast('info', 'Tracking in progress', 'Cancel tracking before editing prompts.');
      return false;
    }
    return true;
  };

  return {
    backendStatus: null,
    backendError: null,
    compute: 'local',
    computeChosen: false,
    remoteWarming: false,
    uploadState: 'idle',
    video: null,
    sessionId: null,
    tool: 'positive',
    trackFrames: 10,
    gpuPrecision: 'fp16',
    effects: DEFAULT_EFFECTS,
    viewMode: 'after',
    splitPosition: 0.5,
    showMask: true,
    toasts: [],
    ...sessionDefaults,

    async refreshBackendStatus() {
      try {
        const status = await api.getStatus();
        const remoteOk = !!status.remote?.configured;
        const s = get();
        // 원격 GPU 가 설정되어 있으면 기본으로 사용, 없어지면 로컬 CPU 로
        const compute: ComputeBackend = !remoteOk ? 'local' : s.computeChosen ? s.compute : 'remote';
        set({ backendStatus: status, backendError: null, compute });
      } catch (e) {
        set({ backendStatus: null, backendError: describeError(e) });
      }
    },

    async loadVideo(file) {
      const basicError = validateFileBasics(file);
      if (basicError) return toast('error', 'Upload rejected', basicError);
      set({ uploadState: 'validating' });
      const url = URL.createObjectURL(file);
      try {
        const info = await loadVideoInfo(url);
        const infoError = validateVideoInfo(info);
        if (infoError) throw new Error(infoError);
        const { sessionId, meta } = await api.createSession(file, info);
        get().closeVideo();
        const preview = previewSize(meta.width, meta.height);
        set({
          ...sessionDefaults,
          uploadState: 'idle',
          sessionId,
          video: { file, url, meta, previewWidth: preview.width, previewHeight: preview.height },
        });
        generateThumbnails(url, meta.duration, 16)
          .then((thumbnails) => {
            if (get().video?.url === url) set({ thumbnails });
          })
          .catch(() => {
            /* 썸네일 실패는 치명적이지 않음 */
          });
      } catch (e) {
        URL.revokeObjectURL(url);
        set({ uploadState: 'idle' });
        // 검증 실패는 "거부", 서버 연결/처리 실패는 "실패"로 구분
        const rejected = !(e instanceof ApiError) || e.code === 'validation';
        toast('error', rejected ? 'Upload rejected' : 'Upload failed', describeError(e));
      }
    },

    closeVideo() {
      const s = get();
      if (isJobActive(s.trackingJob)) void api.cancelJob(s.trackingJob!.id);
      if (isJobActive(s.exportJob)) void api.cancelJob(s.exportJob!.id);
      if (s.sessionId) void api.deleteSession(s.sessionId);
      if (s.video) URL.revokeObjectURL(s.video.url);
      segmentSeq++;
      set({ ...sessionDefaults, video: null, sessionId: null });
    },

    setCurrentFrame(frame) {
      const v = get().video;
      if (!v) return;
      set({ currentFrame: Math.max(0, Math.min(v.meta.frameCount - 1, Math.round(frame))) });
    },

    stepFrame(delta) {
      set({ isPlaying: false });
      get().setCurrentFrame(get().currentFrame + delta);
    },

    setPlaying(isPlaying) {
      if (get().video) set({ isPlaying });
    },

    setTool(tool) {
      set({ tool });
    },

    async addPoint(p, label) {
      const s = get();
      if (!s.video || !guardEditable()) return;
      const rollback: PromptSnapshot = {
        points: s.points,
        promptFrame: s.promptFrame,
        currentMask: s.currentMask,
        candidates: s.candidates,
        selectedCandidate: s.selectedCandidate,
        bestCandidate: s.bestCandidate,
      };
      let points = s.points;
      let currentMask = s.currentMask;
      if (s.promptFrame !== null && s.promptFrame !== s.currentFrame && points.length > 0) {
        points = [];
        currentMask = null; // 다른 프레임의 마스크를 새 프레임에 표시하지 않음
        toast('info', `Prompt moved to frame ${s.currentFrame + 1}`, 'Previous points were cleared.');
      }
      invalidateTracking();
      set({
        isPlaying: false,
        promptFrame: s.currentFrame,
        currentMask,
        points: [...points, { id: newId(), x: p.x, y: p.y, label }],
      });
      await segment(rollback);
    },

    async removePoint(id) {
      if (!guardEditable()) return;
      const s = get();
      const rollback: PromptSnapshot = {
        points: s.points,
        promptFrame: s.promptFrame,
        currentMask: s.currentMask,
        candidates: s.candidates,
        selectedCandidate: s.selectedCandidate,
        bestCandidate: s.bestCandidate,
      };
      invalidateTracking();
      set({ points: s.points.filter((p) => p.id !== id) });
      await segment(rollback);
    },

    async undoPoint() {
      const pts = get().points;
      if (pts.length) await get().removePoint(pts[pts.length - 1].id);
    },

    clearPoints() {
      if (!guardEditable()) return;
      invalidateTracking();
      segmentSeq++;
      set({ points: [], promptFrame: null, currentMask: null, candidates: null, isSegmenting: false });
    },

    async startTracking() {
      const s = get();
      if (!s.video || !s.sessionId || s.promptFrame === null || !s.currentMask || isJobActive(s.trackingJob)) return;
      if (!api.capabilities.tracking) {
        toast('info', 'Not available yet', 'Video tracking with SAM 2.1 arrives in Phase 3.');
        return;
      }
      const sessionId = s.sessionId;
      const size = { width: s.video.previewWidth, height: s.video.previewHeight };
      set({ trackedMasks: null, exportJob: null, isPlaying: false });
      try {
        const job = await api.startPropagation(sessionId, {
          frameIndex: s.promptFrame,
          points: s.points,
          maskWidth: size.width,
          maskHeight: size.height,
          maxFrames: s.trackFrames,
          backend: s.compute,
          // 고른 마스크 후보 (기본 후보면 보내지 않음 → 기존처럼 클릭 점으로 추적)
          ...(s.candidates && s.selectedCandidate !== s.bestCandidate ? { maskCandidate: s.selectedCandidate } : {}),
        });
        set({ trackingJob: job });
        const isCurrent = () => get().trackingJob?.id === job.id;
        const final = await pollJob(job.id, (j) => isCurrent() && set({ trackingJob: j }));
        if (!isCurrent()) return;
        if (final.stats?.source === 'remote-gpu' || s.compute === 'remote') void get().refreshBackendStatus();
        const done = final.stats?.framesDone ?? 0;
        if (final.status === 'succeeded' || (final.status === 'cancelled' && done > 0)) {
          // 취소된 경우에도 이미 실제로 추적된 프레임은 유지
          const masks = await api.fetchTrackedMasks(sessionId, job.id, size);
          if (!isCurrent()) return;
          // 이 결과를 만든 프롬프트를 함께 기록 (나중에 결과를 다시 선택할 때 복원)
          const prompt = { frameIndex: s.promptFrame, points: s.points.map(({ x, y, label }) => ({ x, y, label })) };
          const entryJob: Job = { ...final, stats: { ...final.stats, prompt } };
          set({ trackingJob: entryJob, trackedMasks: masks, trackHistory: [...get().trackHistory, { job: entryJob, label: trackLabel(entryJob) }] });
          const lost = final.stats?.lostFrames?.length ?? 0;
          if (final.status === 'succeeded') {
            toast(
              lost ? 'info' : 'success',
              'Tracking complete',
              (final.message ?? `${masks.size} frames tracked`) + (lost ? ' — check the marked frames on the timeline.' : ''),
            );
          } else {
            toast('info', 'Tracking cancelled', `Kept ${masks.size} frames that were already tracked.`);
          }
        } else if (final.status === 'failed') {
          toast('error', 'Tracking failed', describeJobError(final.error ?? undefined));
        } else {
          toast('info', 'Tracking cancelled');
        }
      } catch (e) {
        set({ trackingJob: null });
        toast('error', 'Tracking failed', describeError(e));
      }
    },

    setTrackFrames(n) {
      set({ trackFrames: Math.max(2, Math.round(n)) });
    },

    selectCandidate(index) {
      const s = get();
      const c = s.candidates?.[index];
      if (!c || index === s.selectedCandidate || !guardEditable()) return;
      invalidateTracking(); // 마스크가 바뀌었으므로 이전 추적 결과는 무효
      set({ selectedCandidate: index, currentMask: c.mask });
    },

    setCompute(c) {
      if (isJobActive(get().trackingJob)) {
        toast('info', 'Tracking in progress', 'Cancel tracking before switching between Local CPU and Remote GPU.');
        return;
      }
      if (c === 'remote' && !get().backendStatus?.remote?.configured) return;
      set({ compute: c, computeChosen: true });
    },

    async warmupRemote() {
      if (get().remoteWarming) return;
      set({ remoteWarming: true });
      try {
        await api.warmupRemote();
        toast('success', 'Remote GPU ready', 'The GPU container is running. It shuts down by itself when idle.');
      } catch (e) {
        toast('error', 'Remote GPU unavailable', describeError(e));
      } finally {
        set({ remoteWarming: false });
        void get().refreshBackendStatus();
      }
    },

    async selectTrack(jobId) {
      const s = get();
      const entry = s.trackHistory.find((e) => e.job.id === jobId);
      if (!entry || !s.sessionId || !s.video || isJobActive(s.trackingJob)) return;
      try {
        const masks = await api.fetchTrackedMasks(s.sessionId, jobId, { width: s.video.previewWidth, height: s.video.previewHeight });
        const prompt = entry.job.stats?.prompt;
        segmentSeq++; // 진행 중인 클릭 분할 응답은 무시
        set({
          trackingJob: entry.job,
          trackedMasks: masks,
          exportJob: null, // 효과 설정은 그대로, 이전 내보내기 결과만 초기화
          isPlaying: false,
          isSegmenting: false,
          ...(prompt
            ? {
                // 가져온 결과를 만든 클릭 프롬프트로 표시를 맞춤
                promptFrame: prompt.frameIndex,
                points: prompt.points.map((p) => ({ id: newId(), x: p.x, y: p.y, label: p.label })),
                currentMask: masks.get(prompt.frameIndex) ?? null,
                // 이전 결과로 돌아가면 마스크 후보는 다시 클릭할 때까지 숨김 (서버의 후보와 어긋나지 않도록)
                candidates: null,
              }
            : {}),
        });
      } catch (e) {
        toast('error', 'Could not load tracking result', describeError(e));
      }
    },

    setGpuPrecision(gpuPrecision) {
      set({ gpuPrecision });
    },

    async createGpuJob() {
      const s = get();
      if (!s.sessionId || !s.video || s.promptFrame === null || !s.points.some((p) => p.label === 1) || s.gpuBusy) return null;
      const limit = s.backendStatus?.gpuImport?.maxFrames ?? 300;
      const maxFrames = Math.min(limit, s.video.meta.frameCount - s.promptFrame);
      set({ gpuBusy: 'creating' });
      try {
        const info = await api.createGpuJob(s.sessionId, {
          frameIndex: s.promptFrame,
          points: s.points,
          maxFrames,
          precision: s.gpuPrecision,
          ...(s.candidates && s.selectedCandidate !== s.bestCandidate ? { maskCandidate: s.selectedCandidate } : {}),
        });
        set({ gpuJob: info, gpuBusy: null });
        triggerDownload(info.downloadUrl);
        toast('success', 'GPU job ready', `${info.numFrames} frames (${info.precision}). Upload the zip to the Colab notebook.`);
        return info;
      } catch (e) {
        set({ gpuBusy: null });
        toast('error', 'Could not create GPU job', describeError(e));
        return null;
      }
    },

    async importGpuResult(file) {
      const s = get();
      if (!s.sessionId || s.gpuBusy || isJobActive(s.trackingJob)) return;
      set({ gpuBusy: 'importing' });
      try {
        const job = await api.importGpuResult(s.sessionId, file);
        set({ gpuBusy: null, trackHistory: [...get().trackHistory, { job, label: trackLabel(job) }] });
        await get().selectTrack(job.id);
        const st = job.stats;
        toast('success', 'GPU result imported', `${st?.framesDone} frames (${st?.rangeStart! + 1}–${st?.rangeEnd! + 1}) · ${trackLabel(job)}`);
      } catch (e) {
        set({ gpuBusy: null });
        toast('error', 'GPU result rejected', describeError(e));
      }
    },

    async compareTracks(a, b) {
      const s = get();
      if (!s.sessionId) return;
      try {
        const r = await api.compareTracks(s.sessionId, a, b);
        set({ comparison: { ...r, a, b } });
      } catch (e) {
        toast('error', 'Comparison failed', describeError(e));
      }
    },

    async cancelJob(kind) {
      const job = kind === 'propagate' ? get().trackingJob : get().exportJob;
      if (job && isJobActive(job)) await api.cancelJob(job.id);
    },

    updateEffect(id, patch) {
      const effects = { ...get().effects, [id]: { ...get().effects[id], ...patch } };
      const exportJob = get().exportJob;
      // 이펙트가 바뀌면 이미 끝난 export 결과는 더 이상 현재 설정과 일치하지 않음
      set({ effects, exportJob: exportJob && isJobActive(exportJob) ? exportJob : null });
    },

    async startExport() {
      const s = get();
      const trackJob = s.trackingJob;
      if (!s.sessionId || !s.trackedMasks || !trackJob || isJobActive(trackJob) || isJobActive(s.exportJob)) return;
      try {
        // 추적 결과(마스크)를 재사용 — 효과만 바꿔 다시 내보내도 재추적하지 않음
        const job = await api.startExport(s.sessionId, { trackJobId: trackJob.id, effects: s.effects });
        set({ exportJob: job });
        const isCurrent = () => get().exportJob?.id === job.id;
        const final = await pollJob(job.id, (j) => isCurrent() && set({ exportJob: j }));
        if (!isCurrent()) return;
        if (final.status === 'succeeded') {
          if (final.resultUrl) {
            const st = final.stats;
            toast(
              'success',
              'Export ready',
              st?.frameCount !== undefined
                ? `${st.frameCount} frames · ${st.duration?.toFixed(2)}s · ${st.outputWidth}×${st.outputHeight}. Ready to download.`
                : 'Your MP4 is ready to download.',
            );
          }
          else toast('info', 'Mock export finished', final.message);
        } else if (final.status === 'failed') {
          toast('error', 'Export failed', describeJobError(final.error ?? undefined));
        } else {
          toast('info', 'Export cancelled');
        }
      } catch (e) {
        set({ exportJob: null });
        toast('error', 'Export failed', describeError(e));
      }
    },

    setViewMode(viewMode) {
      set({ viewMode });
    },
    setSplitPosition(pos) {
      set({ splitPosition: Math.max(0, Math.min(1, pos)) });
    },
    setShowMask(showMask) {
      set({ showMask });
    },

    pushToast(t) {
      const id = ++toastSeq;
      set({ toasts: [...get().toasts.slice(-3), { ...t, id }] });
      setTimeout(() => get().dismissToast(id), t.kind === 'error' ? 8000 : 4500);
    },
    dismissToast(id) {
      set({ toasts: get().toasts.filter((t) => t.id !== id) });
    },
  };
});
