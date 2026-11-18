// 중앙 비디오 미리보기: 프레임 렌더링, 클릭 프롬프트, Before/After 비교
import { Eye, EyeOff, Minus, Plus } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react';
import { clientToNormalized, fitContain } from '../lib/coords';
import { frameToTime, timeToFrame } from '../lib/frames';
import { FrameRenderer } from '../lib/renderer';
import { maskForFrame, useStudioStore } from '../store/studioStore';
import type { ViewMode } from '../types';
import { Spinner } from './ui';

const VIEW_MODES: { id: ViewMode; label: string }[] = [
  { id: 'before', label: 'Before' },
  { id: 'split', label: 'Split' },
  { id: 'after', label: 'After' },
];

export function VideoStage() {
  const video = useStudioStore((s) => s.video)!;
  const currentFrame = useStudioStore((s) => s.currentFrame);
  const isPlaying = useStudioStore((s) => s.isPlaying);
  const points = useStudioStore((s) => s.points);
  const promptFrame = useStudioStore((s) => s.promptFrame);
  const isSegmenting = useStudioStore((s) => s.isSegmenting);
  const encodedFrame = useStudioStore((s) => s.encodedFrame);
  const viewMode = useStudioStore((s) => s.viewMode);
  const splitPosition = useStudioStore((s) => s.splitPosition);
  const showMask = useStudioStore((s) => s.showMask);
  const tool = useStudioStore((s) => s.tool);
  const { setCurrentFrame, setPlaying, addPoint, removePoint, setViewMode, setSplitPosition, setShowMask } =
    useStudioStore.getState();

  const { fps, frameCount, duration } = video.meta;
  const containerRef = useRef<HTMLDivElement>(null);
  const frameBoxRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FrameRenderer | null>(null);
  const [box, setBox] = useState({ width: 0, height: 0 });
  const [draggingSplit, setDraggingSplit] = useState(false);

  // 컨테이너 크기 추적 → 레터박스 계산
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setBox({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const fit = fitContain(box, { width: video.previewWidth, height: video.previewHeight });

  const draw = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c || v.readyState < 2) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const s = useStudioStore.getState();
    rendererRef.current ??= new FrameRenderer();
    rendererRef.current.render(ctx, v, {
      mask: maskForFrame(s, s.currentFrame),
      effects: s.effects,
      view: s.viewMode,
      split: s.splitPosition,
      showMask: s.showMask,
    });
  }, []);

  // 일시정지 상태에서 상태가 바뀌면 다시 그림 (마스크/이펙트/뷰 모드 등)
  useEffect(
    () =>
      useStudioStore.subscribe(() => {
        if (!useStudioStore.getState().isPlaying) draw();
      }),
    [draw],
  );

  // 프레임 이동 → seek (재생 중에는 rAF 루프가 프레임을 갱신)
  useEffect(() => {
    const v = videoRef.current;
    if (!v || isPlaying) return;
    const t = frameToTime(currentFrame, fps, duration);
    if (Math.abs(v.currentTime - t) > 0.25 / fps) v.currentTime = t;
    else draw();
  }, [currentFrame, isPlaying, fps, duration, draw]);

  // 재생 루프
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (!isPlaying) {
      v.pause();
      return;
    }
    if (useStudioStore.getState().currentFrame >= frameCount - 1) v.currentTime = 0;
    v.play().catch(() => setPlaying(false));
    let raf = 0;
    const tick = () => {
      const f = timeToFrame(v.currentTime, fps, frameCount);
      if (f !== useStudioStore.getState().currentFrame) setCurrentFrame(f);
      draw();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, fps, frameCount, draw, setCurrentFrame, setPlaying]);

  const handleStageClick = (e: ReactMouseEvent, forceNegative = false) => {
    if (draggingSplit || !frameBoxRef.current) return;
    const p = clientToNormalized(e.clientX, e.clientY, frameBoxRef.current.getBoundingClientRect());
    if (!p) return;
    const negative = forceNegative || e.shiftKey || tool === 'negative';
    void addPoint(p, negative ? 0 : 1);
  };

  const updateSplit = (e: ReactPointerEvent) => {
    const r = frameBoxRef.current?.getBoundingClientRect();
    if (r) setSplitPosition((e.clientX - r.left) / r.width);
  };

  const showPoints = promptFrame === currentFrame && !isPlaying;

  return (
    <div className="relative flex min-h-[240px] flex-1 flex-col overflow-hidden rounded-2xl border border-white/5 stage-bg sm:min-h-[320px]">
      {/* 상단 툴바 */}
      <div className="absolute top-3 right-3 left-3 z-20 flex items-center justify-between gap-2">
        <div className="flex rounded-lg bg-ink-900/80 p-0.5 ring-1 ring-white/10 backdrop-blur">
          {VIEW_MODES.map((m) => (
            <button
              key={m.id}
              onClick={() => setViewMode(m.id)}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                viewMode === m.id ? 'bg-violet-500 text-white shadow' : 'text-zinc-400 hover:text-zinc-100'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <button
          onClick={() => setShowMask(!showMask)}
          className="flex items-center gap-1.5 rounded-lg bg-ink-900/80 px-2.5 py-1.5 text-xs text-zinc-300 ring-1 ring-white/10 backdrop-blur transition-colors hover:text-white"
        >
          {showMask ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
          Mask overlay
        </button>
      </div>

      <div ref={containerRef} className="relative flex-1">
        <video ref={videoRef} src={video.url} muted playsInline preload="auto" className="hidden" onSeeked={draw} onLoadedData={draw} onEnded={() => setPlaying(false)} />
        <div
          ref={frameBoxRef}
          className="absolute overflow-hidden rounded-md shadow-2xl shadow-black/60 ring-1 ring-white/10"
          style={{ left: fit.offsetX, top: fit.offsetY, width: fit.width, height: fit.height }}
        >
          <canvas
            ref={canvasRef}
            width={video.previewWidth}
            height={video.previewHeight}
            className="block h-full w-full"
          />
          {/* 클릭 캡처 레이어 */}
          <div
            className={`absolute inset-0 ${tool === 'negative' ? 'cursor-crosshair' : 'cursor-cell'}`}
            onClick={(e) => handleStageClick(e)}
            onContextMenu={(e) => {
              e.preventDefault();
              handleStageClick(e, true);
            }}
            data-testid="stage-click-layer"
          />

          {viewMode === 'split' && (
            <>
              <span className="pointer-events-none absolute bottom-3 left-3 rounded bg-black/60 px-2 py-0.5 text-[10px] font-semibold tracking-wider text-zinc-200 uppercase">Before</span>
              <span className="pointer-events-none absolute right-3 bottom-3 rounded bg-violet-500/80 px-2 py-0.5 text-[10px] font-semibold tracking-wider text-white uppercase">After</span>
              <div
                className="absolute top-0 bottom-0 z-10 w-6 -translate-x-1/2 cursor-ew-resize touch-none"
                style={{ left: `${splitPosition * 100}%` }}
                onClick={(e) => e.stopPropagation()}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  e.currentTarget.setPointerCapture(e.pointerId);
                  setDraggingSplit(true);
                }}
                onPointerMove={(e) => draggingSplit && updateSplit(e)}
                onPointerUp={() => setTimeout(() => setDraggingSplit(false), 0)}
              >
                <div className="mx-auto h-full w-0.5 bg-white/90 shadow-[0_0_8px_rgba(0,0,0,0.6)]" />
                <div className="absolute top-1/2 left-1/2 grid h-7 w-7 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-[10px] font-bold text-ink-900 shadow-lg">
                  ⇆
                </div>
              </div>
            </>
          )}

          {showPoints &&
            points.map((p) => (
              <button
                key={p.id}
                title="Remove point"
                onClick={(e) => {
                  e.stopPropagation();
                  void removePoint(p.id);
                }}
                onContextMenu={(e) => e.stopPropagation()}
                className={`absolute z-10 grid h-5 w-5 -translate-x-1/2 -translate-y-1/2 animate-fade-in place-items-center rounded-full text-white shadow-lg ring-2 ring-white transition-transform hover:scale-125 ${
                  p.label === 1 ? 'bg-emerald-500' : 'bg-rose-500'
                }`}
                style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
              >
                {p.label === 1 ? <Plus className="h-3 w-3" strokeWidth={3} /> : <Minus className="h-3 w-3" strokeWidth={3} />}
              </button>
            ))}

          {isSegmenting && (
            <div className="pointer-events-none absolute top-14 left-1/2 flex -translate-x-1/2 animate-fade-in items-center gap-2 rounded-full bg-ink-900/90 px-3 py-1.5 text-xs text-zinc-200 ring-1 ring-white/10">
              <Spinner className="h-3 w-3" />
              {encodedFrame === promptFrame ? 'Segmenting…' : 'Encoding frame with SAM 2.1…'}
            </div>
          )}
        </div>
      </div>

      <p className="pointer-events-none px-4 pb-3 text-center text-[11px] text-zinc-500">
        Click to add a <span className="text-emerald-400">positive</span> point · Shift-click or right-click for{' '}
        <span className="text-rose-400">negative</span> · Click a point to remove it
      </p>
    </div>
  );
}
