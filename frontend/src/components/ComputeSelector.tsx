// 분할·추적 실행 위치 선택 (Local CPU / Remote GPU) + 원격 GPU 사용량·추정 비용 (Phase 5B-2)
import { Cloud, Cpu, Power } from 'lucide-react';
import { api } from '../api';
import type { RemoteStatus } from '../types';
import { isJobActive, useStudioStore } from '../store/studioStore';
import { Spinner } from './ui';

export function fmtUSD(v: number) {
  return v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
}

/** 추적 1회의 추정 청구 시간(초): 콜드 스타트(꺼져 있으면) + 프레임 처리 + 대기 시간 */
export function estimateRemoteTrackSeconds(remote: RemoteStatus, frames: number, perFrameS = 0.25): number {
  const u = remote.usage;
  if (!u) return frames * perFrameS;
  return (u.likelyWarm ? 0 : u.coldStartEstimateS) + frames * perFrameS + u.idleWindowS;
}

export function ComputeSelector() {
  const status = useStudioStore((s) => s.backendStatus);
  const compute = useStudioStore((s) => s.compute);
  const warming = useStudioStore((s) => s.remoteWarming);
  const tracking = useStudioStore((s) => isJobActive(s.trackingJob));
  const { setCompute, warmupRemote } = useStudioStore.getState();
  const remote = status?.remote;
  if (api.mode === 'mock' || !remote?.configured) return null;

  const u = remote.usage;
  const options = [
    { id: 'remote' as const, label: `Remote GPU · ${remote.gpu ?? 'GPU'}`, icon: Cloud },
    { id: 'local' as const, label: 'Local CPU', icon: Cpu },
  ];
  const stateText =
    remote.state === 'error' ? 'last call failed' : u?.likelyWarm ? 'running' : remote.state === 'ready' ? 'idle (starts on demand)' : 'not started';

  return (
    <div className="mb-4" data-testid="compute-selector">
      <div className="grid grid-cols-2 gap-1 rounded-lg bg-white/[0.03] p-1 ring-1 ring-white/5">
        {options.map((o) => (
          <button
            key={o.id}
            disabled={tracking}
            onClick={() => setCompute(o.id)}
            className={`flex items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs transition-colors disabled:opacity-50 ${
              compute === o.id ? 'bg-violet-500/20 text-violet-100 ring-1 ring-violet-400/40' : 'text-zinc-400 hover:text-zinc-100'
            }`}
          >
            <o.icon className="h-3.5 w-3.5" />
            {o.label}
          </button>
        ))}
      </div>
      {compute === 'remote' && (
        <div className="mt-2 space-y-1 text-[11px] text-zinc-500" data-testid="remote-usage">
          <div className="flex items-center justify-between gap-2">
            <span>
              {remote.provider === 'inprocess' ? 'In-process test worker' : 'Serverless GPU'} · {remote.precision ?? 'fp16'} ·{' '}
              <span className={remote.state === 'error' ? 'text-rose-300' : u?.likelyWarm ? 'text-emerald-300' : ''}>{stateText}</span>
            </span>
            {!u?.likelyWarm && (
              <button
                onClick={() => void warmupRemote()}
                disabled={warming}
                title="Start the GPU container now so the first click is fast. It is billed while running and stops by itself when idle."
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-zinc-400 ring-1 ring-white/10 hover:text-zinc-100 disabled:opacity-50"
              >
                {warming ? <Spinner className="h-3 w-3" /> : <Power className="h-3 w-3" />}
                Start GPU
              </button>
            )}
          </div>
          {u && (
            <p>
              Today {Math.round(u.todayBillableS / 60)} / {Math.round(u.dailyLimitS / 60)} min · this month ≈ {fmtUSD(u.monthEstimatedUSD)} /{' '}
              {fmtUSD(u.monthlyLimitUSD)} <span className="text-zinc-600">(estimate)</span>
            </p>
          )}
          {remote.state === 'error' && remote.lastError && <p className="text-rose-300/90">{remote.lastError}</p>}
          <p className="text-zinc-600">The first request after idle includes a cold start (up to about a minute).</p>
        </div>
      )}
    </div>
  );
}
