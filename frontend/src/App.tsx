import { useEffect } from 'react';
import { EffectsPanel } from './components/EffectsPanel';
import { ExportPanel } from './components/ExportPanel';
import { GpuPanel } from './components/GpuPanel';
import { Header } from './components/Header';
import { SelectionPanel } from './components/SelectionPanel';
import { Timeline } from './components/Timeline';
import { Toasts } from './components/Toasts';
import { UploadDropzone } from './components/UploadDropzone';
import { VideoStage } from './components/VideoStage';
import { useStudioStore } from './store/studioStore';

function useKeyboardShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target;
      if (target instanceof HTMLElement && target.closest('input, textarea, select, [contenteditable]')) return;
      const s = useStudioStore.getState();
      if (!s.video) return;
      if (e.code === 'Space') {
        e.preventDefault();
        s.setPlaying(!s.isPlaying);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        s.stepFrame(e.shiftKey ? -10 : -1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        s.stepFrame(e.shiftKey ? 10 : 1);
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        void s.undoPoint();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

/** 백엔드/모델 상태 폴링: 로딩 중이거나 오프라인이면 2초, 준비되면 15초 간격 */
function useBackendStatus() {
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let alive = true;
    const tick = async () => {
      await useStudioStore.getState().refreshBackendStatus();
      if (!alive) return;
      const s = useStudioStore.getState();
      const settled = s.backendStatus && (s.backendStatus.modelState === 'ready' || s.backendStatus.modelState === 'error');
      timer = setTimeout(tick, settled ? 15000 : 2000);
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);
}

export default function App() {
  const hasVideo = useStudioStore((s) => !!s.video);
  useKeyboardShortcuts();
  useBackendStatus();

  return (
    <div className="flex min-h-dvh flex-col lg:h-dvh">
      <Header />
      <main className="flex flex-1 flex-col lg:min-h-0 lg:flex-row">
        <section className="flex min-h-[70vh] min-w-0 flex-1 flex-col gap-3 p-3 lg:min-h-0">
          {hasVideo ? (
            <>
              <VideoStage />
              <Timeline />
            </>
          ) : (
            <UploadDropzone />
          )}
        </section>
        <aside className="scrollbar-thin w-full shrink-0 border-t border-white/5 bg-ink-900/60 lg:w-[340px] lg:overflow-y-auto lg:border-t-0 lg:border-l">
          <SelectionPanel />
          <GpuPanel />
          <EffectsPanel />
          <ExportPanel />
        </aside>
      </main>
      <Toasts />
    </div>
  );
}
