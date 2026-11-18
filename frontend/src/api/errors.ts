import type { ErrorCode, JobError } from '../types';

export class ApiError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
  }
}

const FRIENDLY: Partial<Record<ErrorCode, string>> = {
  gpu_oom: 'The GPU ran out of memory. Try a shorter or lower-resolution clip, or retry in a moment.',
  network: 'Could not reach the server. Check that the backend is running.',
  not_found: 'This session has expired. Please upload the video again.',
};

/** 원격 GPU 오류는 서버 메시지(원인과 해결 방법 포함)를 그대로 보여 주고, 로컬 CPU 로 전환할 수 있음을 알림 */
const REMOTE_HINT: Partial<Record<ErrorCode, string>> = {
  remote_unavailable: ' Switch to Local CPU to keep working.',
  remote_failed: ' Switch to Local CPU to keep working.',
  budget_exceeded: '',
};

function withHint(code: ErrorCode, message: string): string {
  const hint = REMOTE_HINT[code];
  return hint === undefined ? message : message + hint;
}

export function describeJobError(e: JobError | null | undefined): string {
  if (!e) return 'Unknown error';
  return FRIENDLY[e.code] ?? withHint(e.code, e.message);
}

export function describeError(err: unknown): string {
  if (err instanceof ApiError) return FRIENDLY[err.code] ?? withHint(err.code, err.message);
  if (err instanceof Error) return err.message;
  return 'Unexpected error';
}
