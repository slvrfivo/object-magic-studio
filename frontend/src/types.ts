// 앱 전역에서 공유하는 도메인 타입

/** SAM 규칙과 동일: 1 = positive(포함), 0 = negative(제외) */
export type PointLabel = 0 | 1;

/** 클릭 프롬프트. x, y는 프레임 기준 [0, 1] 정규화 좌표 (해상도 독립) */
export interface PromptPoint {
  id: string;
  x: number;
  y: number;
  label: PointLabel;
}

export interface BasicVideoInfo {
  width: number;
  height: number;
  duration: number;
}

export interface VideoMeta extends BasicVideoInfo {
  fps: number;
  frameCount: number;
  /** 'probed' = 백엔드가 실제 측정, 'assumed' = Mock 기본값 */
  fpsSource: 'probed' | 'assumed';
}

/** 바이너리 마스크 (0/1), 행 우선(row-major) */
export interface Mask {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface NeonSettings {
  enabled: boolean;
  intensity: number; // 0..1
  color: string; // #rrggbb
  thickness: number; // 1..10
}

export interface ColorPopSettings {
  enabled: boolean;
  intensity: number;
}

export interface BlurSettings {
  enabled: boolean;
  intensity: number;
}

export interface EffectSettings {
  neon: NeonSettings;
  colorPop: ColorPopSettings;
  blur: BlurSettings;
}

export type EffectId = keyof EffectSettings;

export type ViewMode = 'before' | 'split' | 'after';

export type ErrorCode =
  | 'validation'
  | 'gpu_oom'
  | 'not_found'
  | 'cancelled'
  | 'network'
  | 'internal'
  | 'model_unavailable'
  | 'inference_failed'
  | 'not_implemented'
  | 'busy'
  | 'invalid_result'
  | 'export_unavailable'
  | 'export_failed'
  | 'remote_unavailable'
  | 'remote_failed'
  | 'budget_exceeded';

/** 분할·추적을 실행할 위치. remote = 서버리스 GPU (Phase 5B-2) */
export type ComputeBackend = 'local' | 'remote';

/** 분할 1회의 측정값. Mock 모드에서는 device === 'mock' */
export interface InferenceMetrics {
  device: 'cuda' | 'cpu' | 'mock';
  deviceName?: string;
  dtype?: string;
  embeddingCached?: boolean;
  /** 원격 GPU: 이 프레임의 특징을 추적과 공유하는 캐시에서 재사용 */
  embeddingShared?: boolean;
  backend?: ComputeBackend;
  /** 원격 GPU: 영상 업로드 시간 (처음 한 번), 원격 왕복 시간, 이번 호출의 추정 비용 */
  uploadMs?: number;
  remoteRoundTripMs?: number;
  estimatedCostUSD?: number;
  coldStart?: boolean;
  setImageMs?: number;
  predictMs?: number;
  inferenceMs?: number;
  /** 브라우저에서 측정한 왕복 시간 (업로드/네트워크/디코딩 포함) */
  roundTripMs?: number;
  score?: number;
  multimask?: boolean;
  peakAllocatedMB?: number;
  reservedMB?: number;
  processRssMB?: number;
  coverage?: number;
}

/** 한 번 클릭일 때 SAM 이 낸 마스크 후보 (부분 / 전체 / 더 넓은 영역 등) */
export interface MaskCandidate {
  mask: Mask;
  score: number;
  /** 원본 해상도 기준 면적 비율 */
  coverage: number;
}

export interface SegmentResult {
  mask: Mask;
  metrics: InferenceMetrics;
  /** 한 번 클릭일 때만: 후보 3개와 기본 선택(점수 최고) 번호 */
  candidates?: MaskCandidate[];
  selected?: number;
}

export type ModelState = 'not_loaded' | 'loading' | 'ready' | 'error';

export interface BackendStatus {
  mode: 'mock' | 'live';
  modelState: ModelState;
  modelError?: string | null;
  device?: 'cuda' | 'cpu';
  deviceReason?: string;
  deviceName?: string | null;
  torchVersion?: string;
  loadMs?: number | null;
  warmup?: { setImageMs: number; predictMs: number } | null;
  /** 추적 한도와 최근 실측 프레임당 시간 (ETA 계산용) */
  tracking?: { maxFrames: number; lastAvgFrameMs: number | null };
  /** FFmpeg 가 없으면 null → 내보내기 불가 */
  ffmpeg?: { version: string; source: string } | null;
  /** 외부 GPU 결과 가져오기 한도 (CPU 추적 한도와 별도) */
  gpuImport?: { maxFrames: number; precisions: ('fp16' | 'fp32')[]; defaultPrecision: 'fp16' | 'fp32' };
  /** 원격 GPU 연결 (Phase 5B-2). configured=false 면 로컬 CPU 만 */
  remote?: RemoteStatus;
}

export interface RemoteUsage {
  todayBillableS: number;
  todayCalls: number;
  todayTrackedFrames: number;
  monthBillableS: number;
  monthEstimatedUSD: number;
  dailyLimitS: number;
  monthlyLimitUSD: number;
  costPerSecondUSD: number;
  idleWindowS: number;
  coldStartEstimateS: number;
  likelyWarm: boolean;
}

export interface RemoteStatus {
  configured: boolean;
  provider?: 'modal' | 'inprocess';
  gpu?: string;
  precision?: string;
  /** idle = 아직 호출 안 함, ready = 마지막 호출 성공, error = 마지막 호출 실패 */
  state?: 'idle' | 'ready' | 'error';
  lastError?: string | null;
  maxFrames?: number;
  usage?: RemoteUsage;
}

export interface JobError {
  code: ErrorCode;
  message: string;
}

export type JobKind = 'propagate' | 'export';
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

/** 추적 작업의 실측 통계 (백엔드가 프레임마다 갱신) */
export interface TrackStats {
  rangeStart: number;
  rangeEnd: number;
  framesRequested: number;
  framesDone: number;
  trackedFrames: number[];
  frameMs: number[];
  coverage: number[];
  /** 마스크가 비어 있는 프레임 (객체가 보이지 않음) */
  lostFrames: number[];
  /** 면적이 직전 대비 급변한 프레임 (마스크 오류 가능성) */
  unstableFrames: number[];
  initMs?: number;
  promptMs?: number;
  avgFrameMs?: number;
  medianFrameMs?: number;
  maxFrameMs?: number;
  fps?: number;
  peakRssMB?: number;
  peakAllocatedMB?: number;
  totalMs?: number;
  cancelled?: boolean;
  /** 'gpu-import' = Colab 등 외부 GPU 에서 추적해 가져온 결과, 'remote-gpu' = 원격 GPU 자동 연결 */
  source?: 'cpu' | 'gpu-import' | 'remote-gpu';
  /** 원격 GPU 추적: 영상 업로드 시간, 추정 비용, 분할 때 계산한 특징 재사용 여부 */
  uploadMs?: number;
  estimatedCostUSD?: number;
  coldStart?: boolean;
  promptFeaturesReused?: boolean;
  /** 추적 시작 방식: points = 클릭 점, mask = 사용자가 고른 마스크 후보 */
  promptType?: 'points' | 'mask';
  precision?: 'fp16' | 'fp32';
  device?: 'cuda' | 'cpu';
  gpuName?: string | null;
  gpuJobId?: string;
  /** 가져온 결과를 만든 클릭 프롬프트 */
  prompt?: { frameIndex: number; points: { x: number; y: number; label: 0 | 1 }[] };
}

/** 내보내기 작업의 실측 통계 및 ffprobe 검증 결과 */
export interface ExportStats {
  rangeStart: number;
  rangeEnd: number;
  framesRequested: number;
  framesDone: number;
  fpsRational: string;
  width: number;
  height: number;
  audio: boolean;
  renderMs: number[];
  avgRenderMs?: number;
  totalMs?: number;
  fileSizeBytes?: number;
  // ffprobe 로 확인한 출력값
  codec?: string;
  pixFmt?: string;
  outputWidth?: number;
  outputHeight?: number;
  frameCount?: number;
  outputFps?: number;
  duration?: number;
  hasAudio?: boolean;
}

export interface Job {
  id: string;
  kind: JobKind;
  status: JobStatus;
  progress: number; // 0..1
  message?: string;
  error?: JobError | null;
  resultUrl?: string | null;
  stats?: Partial<TrackStats> & Partial<Omit<ExportStats, keyof TrackStats>>;
}

export const TERMINAL_JOB_STATUSES: ReadonlySet<JobStatus> = new Set(['succeeded', 'failed', 'cancelled']);
