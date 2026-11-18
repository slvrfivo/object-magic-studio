// 프론트엔드 ↔ 백엔드 계약. Mock 구현과 실제 HTTP 구현이 같은 인터페이스를 따릅니다.
import type {
  BackendStatus,
  BasicVideoInfo,
  ComputeBackend,
  EffectSettings,
  Job,
  Mask,
  PromptPoint,
  RemoteStatus,
  SegmentResult,
  VideoMeta,
} from '../types';

export interface SegmentRequest {
  frameIndex: number;
  /** 정규화 좌표 [0,1] — 백엔드가 원본 해상도 픽셀 좌표로 변환 */
  points: PromptPoint[];
  /** 프론트엔드 미리보기용 마스크 해상도 */
  maskWidth: number;
  maskHeight: number;
  /** 실행 위치 (생략하면 서버 기본값: 원격 GPU 가 설정되어 있으면 원격) */
  backend?: ComputeBackend;
}

export interface TrackRequest extends SegmentRequest {
  /** 프롬프트 프레임부터 정방향으로 추적할 최대 프레임 수 (프롬프트 프레임 포함) */
  maxFrames: number;
  /** 한 번 클릭의 마스크 후보 중 고른 번호 (기본 후보면 서버가 클릭 점으로 추적) */
  maskCandidate?: number;
}

export interface ExportRequest {
  /** 마스크를 재사용할 추적 작업 (SAM 2 를 다시 실행하지 않음) */
  trackJobId: string;
  effects: EffectSettings;
}

export interface GpuJobRequest {
  frameIndex: number;
  points: PromptPoint[];
  /** 프롬프트 프레임부터 정방향 프레임 수 (최대 300, 영상 끝에서 잘림) */
  maxFrames: number;
  precision: 'fp16' | 'fp32';
  maskCandidate?: number;
}

export interface GpuJobInfo {
  jobId: string;
  startFrame: number;
  numFrames: number;
  precision: 'fp16' | 'fp32';
  packageBytes: number;
  downloadUrl: string;
}

export interface TrackComparison {
  frames: number;
  minIoU: number;
  meanIoU: number;
  worstFrame: number;
  worstDiffPx: number;
}

export interface ApiCapabilities {
  tracking: boolean;
  export: boolean;
  /** 오프라인 GPU 작업 (작업 파일 내보내기 / 결과 가져오기) */
  gpuJobs: boolean;
}

export interface StudioApi {
  readonly mode: 'mock' | 'live';
  readonly capabilities: ApiCapabilities;
  getStatus(): Promise<BackendStatus>;
  createSession(file: File, info: BasicVideoInfo): Promise<{ sessionId: string; meta: VideoMeta }>;
  segmentFrame(sessionId: string, req: SegmentRequest): Promise<SegmentResult>;
  startPropagation(sessionId: string, req: TrackRequest): Promise<Job>;
  getJob(jobId: string): Promise<Job>;
  cancelJob(jobId: string): Promise<void>;
  /** 추적 결과 마스크 (취소된 작업은 완료된 프레임만) */
  fetchTrackedMasks(sessionId: string, jobId: string, size: { width: number; height: number }): Promise<Map<number, Mask>>;
  startExport(sessionId: string, req: ExportRequest): Promise<Job>;
  deleteSession(sessionId: string): Promise<void>;
  createGpuJob(sessionId: string, req: GpuJobRequest): Promise<GpuJobInfo>;
  /** Colab 결과 zip 업로드 → 서버 검증 → 추적 결과(Job)로 등록 */
  importGpuResult(sessionId: string, file: File): Promise<Job>;
  compareTracks(sessionId: string, a: string, b: string): Promise<TrackComparison>;
  /** 원격 GPU 컨테이너를 미리 켜고 상태 확인 (요금 발생) */
  warmupRemote(): Promise<RemoteStatus>;
}
