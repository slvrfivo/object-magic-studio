// Export 패널: 추적된 구간만 MP4 로 내보내기 (마스크 재사용, 재추적 없음)
import { Download, Film } from 'lucide-react';
import { api } from '../api';
import { anyEffectEnabled } from '../lib/effectParams';
import { formatTime } from '../lib/frames';
import { isJobActive, useStudioStore } from '../store/studioStore';
import { JobProgress } from './JobProgress';
import { Button, Panel } from './ui';

function fmtBytes(n?: number) {
  if (n === undefined) return '—';
  return n > 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(2)} MB` : `${Math.round(n / 1024)} KB`;
}

export function ExportPanel() {
  const video = useStudioStore((s) => s.video);
  const trackedMasks = useStudioStore((s) => s.trackedMasks);
  const trackingJob = useStudioStore((s) => s.trackingJob);
  const effects = useStudioStore((s) => s.effects);
  const exportJob = useStudioStore((s) => s.exportJob);
  const backend = useStudioStore((s) => s.backendStatus);
  const { startExport, cancelJob } = useStudioStore.getState();

  const active = isJobActive(exportJob);
  const tracking = isJobActive(trackingJob);
  const frames = trackedMasks ? [...trackedMasks.keys()].sort((a, b) => a - b) : [];
  const first = frames[0];
  const last = frames[frames.length - 1];
  const fps = video?.meta.fps ?? 30;
  const effectNames = [effects.neon.enabled && 'Neon', effects.colorPop.enabled && 'Color Pop', effects.blur.enabled && 'Blur'].filter(Boolean);
  const ffmpegMissing = api.mode === 'live' && backend?.modelState !== undefined && backend.ffmpeg === null;
  const canExport = frames.length > 0 && anyEffectEnabled(effects) && !active && !tracking && !ffmpegMissing;

  const rows: [string, string][] = [
    [
      'Export range',
      frames.length ? `frames ${first + 1}–${last + 1} · ${formatTime(frames.length / fps)}` : '—',
    ],
    ['Effects', effectNames.length ? effectNames.join(', ') : 'None'],
    ['Output', video ? `H.264 MP4 · ${video.meta.width}×${video.meta.height} · ${video.meta.fps} fps` : '—'],
  ];

  const st = exportJob?.status === 'succeeded' ? exportJob.stats : undefined;

  return (
    <Panel title="Export" icon={<Film className="h-3.5 w-3.5" />}>
      <dl className="space-y-1.5 text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3">
            <dt className="text-zinc-500">{k}</dt>
            <dd className="truncate text-right font-mono text-zinc-300">{v}</dd>
          </div>
        ))}
      </dl>
      {frames.length > 0 && video && frames.length < video.meta.frameCount && (
        <p className="mt-2 text-[11px] text-zinc-500">Only tracked frames are exported. Untracked parts of the video are not included.</p>
      )}

      <Button variant="primary" size="lg" className="mt-4 w-full" disabled={!canExport} onClick={() => void startExport()} data-testid="export-button">
        <Download className="h-4 w-4" />
        Export MP4
      </Button>
      {!trackedMasks && video && <p className="mt-2 text-center text-[11px] text-zinc-500">Track the object before exporting.</p>}
      {trackedMasks && !anyEffectEnabled(effects) && <p className="mt-2 text-center text-[11px] text-zinc-500">Enable at least one effect.</p>}
      {ffmpegMissing && <p className="mt-2 text-center text-[11px] text-rose-300">FFmpeg is not installed on the server.</p>}

      {exportJob && (
        <div className="mt-3 space-y-2">
          <JobProgress job={exportJob} label="Exporting" onCancel={() => void cancelJob('export')} />
          {st && st.frameCount !== undefined && (
            <dl className="animate-fade-in space-y-0.5 rounded-lg bg-white/[0.03] p-2.5 text-[11px] ring-1 ring-white/5" data-testid="export-stats">
              {(
                [
                  ['Frames', `${st.frameCount}`],
                  ['Duration', `${st.duration?.toFixed(3)} s`],
                  ['Frame rate', `${st.outputFps} fps (${st.fpsRational})`],
                  ['Resolution', `${st.outputWidth}×${st.outputHeight} · ${st.codec}/${st.pixFmt}`],
                  ['Audio', st.hasAudio ? 'kept (AAC)' : 'none in source'],
                  ['File size', fmtBytes(st.fileSizeBytes)],
                  ['Effects per frame', `${st.avgRenderMs} ms`],
                  ['Total time', `${((st.totalMs ?? 0) / 1000).toFixed(1)} s`],
                ] as [string, string][]
              ).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2">
                  <dt className="text-zinc-500">{k}</dt>
                  <dd className="text-right font-mono text-zinc-300">{v}</dd>
                </div>
              ))}
            </dl>
          )}
          {exportJob.status === 'succeeded' && exportJob.resultUrl && (
            <a
              href={exportJob.resultUrl}
              download
              data-testid="download-link"
              className="flex h-10 w-full animate-slide-up items-center justify-center gap-2 rounded-lg bg-emerald-500/15 text-sm font-medium text-emerald-300 ring-1 ring-emerald-500/30 transition-colors hover:bg-emerald-500/25"
            >
              <Download className="h-4 w-4" /> Download MP4
            </a>
          )}
        </div>
      )}
    </Panel>
  );
}
