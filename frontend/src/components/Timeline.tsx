// 타임라인: 썸네일 스트립, 스크러빙, 프레임 단위 이동, 추적 구간 표시
import { ChevronLeft, ChevronRight, Pause, Play } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { computeRanges, formatTime } from '../lib/frames';
import { useStudioStore } from '../store/studioStore';
import { Button } from './ui';

export function Timeline() {
  const video = useStudioStore((s) => s.video)!;
  const thumbnails = useStudioStore((s) => s.thumbnails);
  const currentFrame = useStudioStore((s) => s.currentFrame);
  const isPlaying = useStudioStore((s) => s.isPlaying);
  const promptFrame = useStudioStore((s) => s.promptFrame);
  const trackedMasks = useStudioStore((s) => s.trackedMasks);
  const lostFrames = useStudioStore((s) => (s.trackedMasks ? s.trackingJob?.stats?.lostFrames : undefined));
  const unstableFrames = useStudioStore((s) => (s.trackedMasks ? s.trackingJob?.stats?.unstableFrames : undefined));
  const { setCurrentFrame, setPlaying, stepFrame } = useStudioStore.getState();
  const { frameCount, fps, fpsSource } = video.meta;

  const trackRef = useRef<HTMLDivElement>(null);
  const [scrubbing, setScrubbing] = useState(false);

  const ranges = useMemo(() => (trackedMasks ? computeRanges(trackedMasks.keys()) : []), [trackedMasks]);

  const frameFromPointer = (e: ReactPointerEvent) => {
    const r = trackRef.current!.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    return Math.min(frameCount - 1, Math.floor(ratio * frameCount));
  };

  const pct = (f: number) => `${(f / frameCount) * 100}%`;

  return (
    <div className="rounded-2xl border border-white/5 bg-ink-900/80 p-3">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          size="sm"
          className="!h-9 !w-9 !rounded-full !px-0"
          onClick={() => setPlaying(!isPlaying)}
          aria-label={isPlaying ? 'Pause' : 'Play'}
          title="Play / Pause (Space)"
        >
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
        </Button>
        <Button variant="ghost" size="sm" onClick={() => stepFrame(-1)} aria-label="Previous frame" title="Previous frame (←)">
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="sm" onClick={() => stepFrame(1)} aria-label="Next frame" title="Next frame (→)">
          <ChevronRight className="h-4 w-4" />
        </Button>
        <div className="font-mono text-xs text-zinc-300 tabular-nums">
          {formatTime(currentFrame / fps)} <span className="text-zinc-600">/ {formatTime(video.meta.duration)}</span>
        </div>
        <div className="ml-auto flex items-center gap-3 text-xs text-zinc-500">
          <span className="font-mono tabular-nums">
            Frame <span className="text-zinc-200">{currentFrame + 1}</span> / {frameCount}
          </span>
          <span title={fpsSource === 'assumed' ? 'Frame rate is assumed in mock mode; the backend will probe the real value.' : 'Probed frame rate'}>
            {fps} fps{fpsSource === 'assumed' && ' (assumed)'}
          </span>
        </div>
      </div>

      <div
        ref={trackRef}
        className="relative h-14 cursor-pointer touch-none overflow-hidden rounded-lg bg-ink-800 ring-1 ring-white/5 select-none"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          setScrubbing(true);
          setPlaying(false);
          setCurrentFrame(frameFromPointer(e));
        }}
        onPointerMove={(e) => scrubbing && setCurrentFrame(frameFromPointer(e))}
        onPointerUp={() => setScrubbing(false)}
        data-testid="timeline-track"
      >
        <div className="absolute inset-0 flex">
          {thumbnails.length > 0
            ? thumbnails.map((src, i) => (
                <img key={i} src={src} alt="" draggable={false} className="h-full min-w-0 flex-1 animate-fade-in object-cover opacity-70" />
              ))
            : Array.from({ length: 16 }, (_, i) => <div key={i} className="h-full flex-1 animate-pulse border-r border-ink-900 bg-ink-700/50" />)}
        </div>

        {/* 추적 완료 구간 */}
        {ranges.map(([a, b]) => (
          <div
            key={a}
            className="absolute bottom-0 h-1.5 bg-gradient-to-r from-violet-500 to-fuchsia-500"
            style={{ left: pct(a), width: pct(b - a + 1) }}
          />
        ))}

        {/* 객체 소실(빨강) / 면적 급변(주황) 프레임 */}
        {lostFrames?.map((f) => (
          <div key={`l${f}`} className="absolute bottom-0 h-3 bg-rose-500" style={{ left: pct(f), width: `max(2px, ${pct(1)})` }} title={`Frame ${f + 1}: object not visible`} />
        ))}
        {unstableFrames?.map((f) => (
          <div key={`u${f}`} className="absolute bottom-0 h-3 bg-amber-400" style={{ left: pct(f), width: `max(2px, ${pct(1)})` }} title={`Frame ${f + 1}: mask area jumped`} />
        ))}

        {/* 프롬프트 프레임 마커 */}
        {promptFrame !== null && (
          <div
            className="absolute top-1 h-2.5 w-2.5 -translate-x-1/2 rotate-45 bg-emerald-400 shadow ring-1 ring-black/40"
            style={{ left: pct(promptFrame + 0.5) }}
            title={`Prompt frame ${promptFrame + 1}`}
          />
        )}

        {/* 재생 헤드 */}
        <div className="pointer-events-none absolute top-0 bottom-0 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_6px_rgba(255,255,255,0.7)]" style={{ left: pct(currentFrame + 0.5) }}>
          <div className="absolute -top-0.5 left-1/2 h-2 w-2.5 -translate-x-1/2 rounded-b-sm bg-white" />
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-4 text-[11px] text-zinc-500">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rotate-45 bg-emerald-400" /> Prompt frame
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-1.5 w-3 rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500" /> Tracked frames
        </span>
        {!!lostFrames?.length && (
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-1 bg-rose-500" /> Object not visible ({lostFrames.length})
          </span>
        )}
        {!!unstableFrames?.length && (
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-1 bg-amber-400" /> Mask jump ({unstableFrames.length})
          </span>
        )}
      </div>
    </div>
  );
}
