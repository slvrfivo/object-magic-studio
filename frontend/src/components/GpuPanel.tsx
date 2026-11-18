// 오프라인 GPU 작업 패널: 작업 파일 내보내기 → Colab 실행 → 결과 가져오기, 추적 결과 전환/비교
import { CheckCircle2, Cloud, Download, GitCompare, Upload } from 'lucide-react';
import { useRef } from 'react';
import { api } from '../api';
import { isJobActive, useStudioStore } from '../store/studioStore';
import { Button, Panel, Spinner } from './ui';

function fmtMB(n: number) {
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function GpuPanel() {
  const video = useStudioStore((s) => s.video);
  const promptFrame = useStudioStore((s) => s.promptFrame);
  const hasPositive = useStudioStore((s) => s.points.some((p) => p.label === 1));
  const precision = useStudioStore((s) => s.gpuPrecision);
  const gpuJob = useStudioStore((s) => s.gpuJob);
  const busy = useStudioStore((s) => s.gpuBusy);
  const history = useStudioStore((s) => s.trackHistory);
  const active = useStudioStore((s) => s.trackingJob);
  const comparison = useStudioStore((s) => s.comparison);
  const limit = useStudioStore((s) => s.backendStatus?.gpuImport?.maxFrames ?? 300);
  const { setGpuPrecision, createGpuJob, importGpuResult, selectTrack, compareTracks } = useStudioStore.getState();
  const fileRef = useRef<HTMLInputElement>(null);

  if (!api.capabilities.gpuJobs) {
    return (
      <Panel title="GPU tracking (Colab)" icon={<Cloud className="h-3.5 w-3.5" />}>
        <p className="text-[11px] text-zinc-500">Available with the live backend only.</p>
      </Panel>
    );
  }

  const tracking = isJobActive(active);
  const start = promptFrame ?? 0;
  const n = video ? Math.min(limit, video.meta.frameCount - start) : 0;
  const canCreate = !!video && promptFrame !== null && hasPositive && !busy && !tracking && n >= 2;

  return (
    <Panel title="GPU tracking (Colab)" icon={<Cloud className="h-3.5 w-3.5" />}>
      <p className="mb-3 text-[11px] leading-relaxed text-zinc-500">
        Track up to {limit} frames on a Colab GPU: download a job, run <span className="font-mono text-zinc-400">colab/phase5b_gpu_tracking.ipynb</span>, then import the result zip.
      </p>

      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="text-zinc-400">Precision</span>
        <div className="flex rounded-md bg-ink-800 p-0.5 ring-1 ring-white/5">
          {(['fp16', 'fp32'] as const).map((p) => (
            <button
              key={p}
              onClick={() => setGpuPrecision(p)}
              className={`rounded px-2.5 py-0.5 font-mono text-[11px] transition-colors ${precision === p ? 'bg-violet-500 text-white' : 'text-zinc-400 hover:text-zinc-100'}`}
            >
              {p}
              {p === 'fp16' && <span className="ml-1 opacity-70">(default)</span>}
            </button>
          ))}
        </div>
      </div>
      <p className="mb-2 text-[11px] text-zinc-500">
        {video && promptFrame !== null ? `Frames ${start + 1}–${start + n} (${n}) from the prompt frame, forward.` : 'Click the object first.'}
      </p>

      <Button className="w-full" disabled={!canCreate} onClick={() => void createGpuJob()} data-testid="gpu-job-button">
        {busy === 'creating' ? <Spinner className="h-3.5 w-3.5" /> : <Download className="h-4 w-4" />}
        1. Download GPU job
      </Button>
      {gpuJob && (
        <p className="mt-1.5 animate-fade-in text-[11px] text-zinc-500" data-testid="gpu-job-info">
          Job <span className="font-mono text-zinc-300">{gpuJob.jobId.slice(0, 8)}</span> · {gpuJob.numFrames} frames · {gpuJob.precision} · {fmtMB(gpuJob.packageBytes)}{' '}
          <a className="text-violet-300 underline-offset-2 hover:underline" href={gpuJob.downloadUrl} download>
            download again
          </a>
        </p>
      )}

      <input
        ref={fileRef}
        type="file"
        accept=".zip,application/zip"
        className="sr-only"
        data-testid="gpu-result-input"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void importGpuResult(f);
          e.target.value = '';
        }}
      />
      <Button className="mt-2 w-full" disabled={!video || !!busy || tracking} onClick={() => fileRef.current?.click()}>
        {busy === 'importing' ? <Spinner className="h-3.5 w-3.5" /> : <Upload className="h-4 w-4" />}
        2. Import GPU result (.zip)
      </Button>

      {history.length > 0 && (
        <div className="mt-4" data-testid="track-history">
          <p className="mb-1.5 text-[10px] font-semibold tracking-wider text-zinc-500 uppercase">Tracking results</p>
          <ul className="space-y-1">
            {history.map(({ job, label }) => {
              const st = job.stats;
              const isActive = active?.id === job.id;
              return (
                <li key={job.id} className={`rounded-lg px-2.5 py-2 ring-1 transition-colors ${isActive ? 'bg-violet-500/10 ring-violet-400/40' : 'ring-white/5 hover:bg-white/[0.03]'}`}>
                  <button className="flex w-full items-center gap-2 text-left" onClick={() => void selectTrack(job.id)} disabled={isActive || tracking}>
                    {isActive ? <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-violet-300" /> : <span className="h-3.5 w-3.5 shrink-0 rounded-full ring-1 ring-zinc-600" />}
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs text-zinc-200">{label}</span>
                      <span className="block font-mono text-[10px] text-zinc-500">
                        {/* 요청 범위(rangeEnd)가 아니라 실제로 추적된 프레임 (취소된 작업은 일부만) */}
                        frames {(st?.rangeStart ?? 0) + 1}–{(st?.trackedFrames?.at(-1) ?? st?.rangeEnd ?? 0) + 1}
                        {job.status === 'cancelled' ? ' (cancelled)' : ''}
                        {st?.avgFrameMs ? ` · ${st.avgFrameMs >= 1000 ? `${(st.avgFrameMs / 1000).toFixed(1)} s` : `${Math.round(st.avgFrameMs)} ms`}/frame` : ''}
                      </span>
                    </span>
                  </button>
                  {!isActive && active && active.status === 'succeeded' && (
                    <button
                      className="mt-1 flex items-center gap-1 text-[10px] text-zinc-500 hover:text-zinc-200"
                      onClick={() => void compareTracks(active.id, job.id)}
                    >
                      <GitCompare className="h-3 w-3" /> Compare with active
                    </button>
                  )}
                  {comparison && ((comparison.a === active?.id && comparison.b === job.id) || (comparison.b === active?.id && comparison.a === job.id)) && (
                    <p className="mt-1 animate-fade-in font-mono text-[10px] text-zinc-400" data-testid="track-comparison">
                      IoU min {comparison.minIoU.toFixed(4)} · mean {comparison.meanIoU.toFixed(4)} · {comparison.frames} frames · worst frame {comparison.worstFrame + 1} ({comparison.worstDiffPx}px)
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </Panel>
  );
}
