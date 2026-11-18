// 객체 선택(클릭 프롬프트) + 추적 패널
import { Crosshair, Gauge, MinusCircle, PlusCircle, Route, Trash2, Undo2 } from 'lucide-react';
import { api } from '../api';
import type { InferenceMetrics, TrackStats } from '../types';
import { maskCoverage } from '../lib/mask';
import { isJobActive, useStudioStore } from '../store/studioStore';
import { ComputeSelector, estimateRemoteTrackSeconds, fmtUSD } from './ComputeSelector';
import { JobProgress } from './JobProgress';
import { MaskCandidates } from './MaskCandidates';
import { Button, Panel } from './ui';

function fmtMs(ms?: number) {
  if (ms === undefined) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

/** 마지막 분할의 측정값: 장치, 인코딩/디코딩 시간, 메모리 */
function InferenceStats({ m }: { m: InferenceMetrics }) {
  if (m.device === 'mock') {
    return (
      <p className="mt-2 rounded-lg bg-amber-500/5 px-3 py-2 text-[11px] text-amber-300/80 ring-1 ring-amber-500/15">
        Mock mask — no model inference was run.
      </p>
    );
  }
  const remote = m.backend === 'remote';
  const encoder = m.embeddingCached ? 'cached (same frame)' : m.embeddingShared ? 'reused (shared cache)' : fmtMs(m.setImageMs);
  const rows: [string, string][] = [
    ['Device', `${remote ? 'Remote ' : ''}${m.device.toUpperCase()}${m.deviceName && m.device === 'cuda' ? ` · ${m.deviceName}` : ''} · ${m.dtype ?? ''}`],
    ['Image encoder', encoder],
    ['Mask decoder', fmtMs(m.predictMs)],
    [remote ? 'GPU worker total' : 'Server total', fmtMs(m.inferenceMs)],
  ];
  if (remote) {
    if (m.uploadMs) rows.push(['Video upload (once)', fmtMs(m.uploadMs)]);
    rows.push(['Remote round trip', `${fmtMs(m.remoteRoundTripMs)}${m.coldStart ? ' (cold start?)' : ''}`]);
    if (m.estimatedCostUSD !== undefined) rows.push(['Est. cost (incl. idle)', fmtUSD(m.estimatedCostUSD)]);
  }
  rows.push(['Round trip', fmtMs(m.roundTripMs)]);
  if (m.peakAllocatedMB !== undefined) rows.push(['GPU peak', `${m.peakAllocatedMB} MB (reserved ${m.reservedMB} MB)`]);
  if (m.processRssMB !== undefined) rows.push(['Process RAM', `${m.processRssMB} MB`]);
  if (m.score !== undefined) rows.push(['Mask score', m.score.toFixed(3)]);
  return (
    <div className="mt-3 animate-fade-in rounded-lg bg-white/[0.03] p-2.5 ring-1 ring-white/5" data-testid="inference-stats">
      <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold tracking-wider text-zinc-500 uppercase">
        <Gauge className="h-3 w-3" /> SAM 2.1 inference
      </p>
      <dl className="space-y-0.5 text-[11px]">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2">
            <dt className="text-zinc-500">{k}</dt>
            <dd className="text-right font-mono text-zinc-300">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function fmtDuration(sec: number) {
  if (sec < 90) return `${Math.round(sec)} s`;
  return `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} s`;
}

/** 추적 프레임 수 선택 + 실측 기반 예상 시간 */
function TrackBudget() {
  const video = useStudioStore((s) => s.video);
  const promptFrame = useStudioStore((s) => s.promptFrame);
  const trackFrames = useStudioStore((s) => s.trackFrames);
  const status = useStudioStore((s) => s.backendStatus);
  const compute = useStudioStore((s) => s.compute);
  const setTrackFrames = useStudioStore((s) => s.setTrackFrames);
  if (!video || api.mode === 'mock') return null;

  const remote = compute === 'remote' && status?.remote?.configured ? status.remote : null;
  const maxFrames = remote ? (remote.maxFrames ?? 300) : (status?.tracking?.maxFrames ?? 60);
  const remaining = video.meta.frameCount - (promptFrame ?? 0);
  const limit = Math.min(remaining, maxFrames);
  const options = [...new Set([10, 30, 60, 150, 300].filter((n) => n < limit).concat(limit))].filter((n) => n >= 2);
  const effective = Math.min(trackFrames, limit);
  const avgMs = status?.tracking?.lastAvgFrameMs;

  return (
    <div className="mt-4" data-testid="track-budget">
      <div className="mb-1.5 flex items-baseline justify-between text-xs">
        <span className="text-zinc-400">Frames to track</span>
        <span className="text-[11px] text-zinc-500">forward from frame {(promptFrame ?? 0) + 1}</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {options.map((n) => (
          <button
            key={n}
            onClick={() => setTrackFrames(n)}
            className={`rounded-md px-2.5 py-1 font-mono text-xs ring-1 transition-colors ${
              effective === n ? 'bg-violet-500/20 text-violet-200 ring-violet-400/40' : 'text-zinc-400 ring-white/10 hover:text-zinc-100'
            }`}
          >
            {n === remaining ? `${n} (to end)` : n}
          </button>
        ))}
      </div>
      {remote ? (
        <p className="mt-1.5 text-[11px] text-zinc-500" data-testid="remote-estimate">
          {`≈ ${fmtDuration(effective * 0.21)} on ${remote.gpu ?? 'GPU'} (T4 fp16 measured ≈ 0.2 s/frame) + upload/start-up. `}
          {`Est. cost ≈ ${fmtUSD(estimateRemoteTrackSeconds(remote, effective) * (remote.usage?.costPerSecondUSD ?? 0))} incl. ${remote.usage?.likelyWarm ? '' : 'cold start and '}idle time.`}
          {remaining > maxFrames && ` Limit: ${maxFrames} frames per job.`}
        </p>
      ) : (
        <p className="mt-1.5 text-[11px] text-zinc-500">
          {avgMs
            ? `≈ ${fmtDuration(((effective - 1) * avgMs) / 1000)} at the last measured ${(avgMs / 1000).toFixed(1)} s/frame (${status?.device?.toUpperCase()})`
            : `Speed is measured on the first run (${status?.device?.toUpperCase() ?? '…'}).`}
          {status?.tracking && remaining > status.tracking.maxFrames && ` Limit on this device: ${status.tracking.maxFrames} frames per job.`}
        </p>
      )}
    </div>
  );
}

export function SelectionPanel() {
  const video = useStudioStore((s) => s.video);
  const tool = useStudioStore((s) => s.tool);
  const points = useStudioStore((s) => s.points);
  const promptFrame = useStudioStore((s) => s.promptFrame);
  const currentMask = useStudioStore((s) => s.currentMask);
  const trackingJob = useStudioStore((s) => s.trackingJob);
  const trackedMasks = useStudioStore((s) => s.trackedMasks);
  const lastInference = useStudioStore((s) => s.lastInference);
  const { setTool, undoPoint, clearPoints, startTracking, cancelJob, setCurrentFrame } = useStudioStore.getState();

  const tracking = isJobActive(trackingJob);
  const positives = points.filter((p) => p.label === 1).length;

  return (
    <Panel title="Select object" icon={<Crosshair className="h-3.5 w-3.5" />}>
      <ComputeSelector />
      <div className="grid grid-cols-2 gap-1 rounded-lg bg-ink-800 p-1">
        {(
          [
            { id: 'positive', label: 'Include', icon: PlusCircle, active: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30' },
            { id: 'negative', label: 'Exclude', icon: MinusCircle, active: 'bg-rose-500/15 text-rose-300 ring-rose-500/30' },
          ] as const
        ).map((t) => (
          <button
            key={t.id}
            onClick={() => setTool(t.id)}
            disabled={!video}
            className={`flex items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium ring-1 transition-all disabled:opacity-40 ${
              tool === t.id ? t.active : 'text-zinc-400 ring-transparent hover:text-zinc-200'
            }`}
          >
            <t.icon className="h-3.5 w-3.5" />
            {t.label}
          </button>
        ))}
      </div>

      <div className="mt-3 flex items-center justify-between text-xs">
        <span className="text-zinc-400">
          {points.length === 0 ? (
            video ? 'Click the object in the preview' : 'Upload a video first'
          ) : (
            <>
              <span className="text-emerald-400">{positives}</span> include ·{' '}
              <span className="text-rose-400">{points.length - positives}</span> exclude
              {promptFrame !== null && (
                <button className="ml-1 text-zinc-500 underline-offset-2 hover:text-zinc-200 hover:underline" onClick={() => setCurrentFrame(promptFrame)}>
                  @ frame {promptFrame + 1}
                </button>
              )}
            </>
          )}
        </span>
        <div className="flex gap-1">
          <Button size="sm" variant="ghost" onClick={() => void undoPoint()} disabled={!points.length || tracking} title="Undo last point">
            <Undo2 className="h-3.5 w-3.5" />
          </Button>
          <Button size="sm" variant="ghost" onClick={clearPoints} disabled={!points.length || tracking} title="Clear all points">
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {currentMask && (
        <div className="mt-2 animate-fade-in text-[11px] text-zinc-500">
          Mask covers <span className="font-mono text-zinc-300">{(maskCoverage(currentMask) * 100).toFixed(1)}%</span> of the frame
        </div>
      )}

      {currentMask && <MaskCandidates />}

      {lastInference && currentMask && <InferenceStats m={lastInference} />}

      {currentMask && <TrackBudget />}

      <Button
        variant="primary"
        className="mt-4 w-full"
        disabled={!currentMask || tracking || !api.capabilities.tracking}
        onClick={() => void startTracking()}
      >
        <Route className="h-4 w-4" />
        {trackedMasks ? 'Re-track object' : 'Track across video'}
      </Button>

      {trackingJob && (
        <div className="mt-3">
          <JobProgress job={trackingJob} label="Tracking" onCancel={() => void cancelJob('propagate')} />
        </div>
      )}
      {trackingJob?.stats?.avgFrameMs !== undefined && <TrackStatsCard stats={trackingJob.stats} />}
    </Panel>
  );
}

/** 추적 실측 통계 */
function TrackStatsCard({ stats }: { stats: Partial<TrackStats> }) {
  const rows: [string, string][] = [
    ['Frames', `${stats.framesDone ?? 0} / ${stats.framesRequested ?? '—'} (${(stats.rangeStart ?? 0) + 1}–${(stats.rangeEnd ?? 0) + 1})`],
    ['Per frame (avg)', fmtMs(stats.avgFrameMs)],
    ['Per frame (median / max)', `${fmtMs(stats.medianFrameMs)} / ${fmtMs(stats.maxFrameMs)}`],
    ['Video state init', fmtMs(stats.initMs)],
  ];
  if (stats.totalMs !== undefined) rows.push(['Total', fmtMs(stats.totalMs)]);
  if (stats.peakRssMB !== undefined) rows.push(['Peak process RAM', `${Math.round(stats.peakRssMB)} MB`]);
  if (stats.peakAllocatedMB !== undefined) rows.push(['GPU peak', `${stats.peakAllocatedMB} MB`]);
  if (stats.source === 'remote-gpu') {
    rows.unshift(['Ran on', `Remote ${stats.gpuName ?? 'GPU'} · ${stats.precision ?? 'fp16'}`]);
    if (stats.uploadMs) rows.push(['Video upload', fmtMs(stats.uploadMs)]);
    if (stats.promptFeaturesReused !== undefined) rows.push(['Prompt frame encoding', stats.promptFeaturesReused ? 'reused from segmentation' : 'computed']);
    if (stats.estimatedCostUSD !== undefined) rows.push(['Est. cost (incl. idle)', fmtUSD(stats.estimatedCostUSD)]);
  }
  const lost = stats.lostFrames?.length ?? 0;
  const unstable = stats.unstableFrames?.length ?? 0;
  return (
    <div className="mt-3 animate-fade-in rounded-lg bg-white/[0.03] p-2.5 ring-1 ring-white/5" data-testid="track-stats">
      <p className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold tracking-wider text-zinc-500 uppercase">
        <Gauge className="h-3 w-3" /> Tracking measurements
      </p>
      <dl className="space-y-0.5 text-[11px]">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2">
            <dt className="text-zinc-500">{k}</dt>
            <dd className="text-right font-mono text-zinc-300">{v}</dd>
          </div>
        ))}
      </dl>
      {(lost > 0 || unstable > 0) && (
        <p className="mt-2 text-[11px] text-amber-300/90">
          {lost > 0 && `Object not visible in ${lost} frame(s). `}
          {unstable > 0 && `Mask area jumped in ${unstable} frame(s) — review them.`}
        </p>
      )}
    </div>
  );
}
