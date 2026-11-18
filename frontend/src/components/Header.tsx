import { Check, RotateCcw, Wand2 } from 'lucide-react';
import { anyEffectEnabled } from '../lib/effectParams';
import { useStudioStore } from '../store/studioStore';
import { ModeBadge } from './ModeBadge';
import { Button } from './ui';

export function Header() {
  const video = useStudioStore((s) => s.video);
  const hasMask = useStudioStore((s) => !!s.currentMask);
  const tracked = useStudioStore((s) => !!s.trackedMasks);
  const effectsOn = useStudioStore((s) => anyEffectEnabled(s.effects));
  const exported = useStudioStore((s) => s.exportJob?.status === 'succeeded');
  const closeVideo = useStudioStore((s) => s.closeVideo);

  const steps = [
    { label: 'Upload', done: !!video },
    { label: 'Select', done: hasMask },
    { label: 'Track', done: tracked },
    { label: 'Style', done: tracked && effectsOn },
    { label: 'Export', done: exported },
  ];
  const activeIndex = steps.findIndex((s) => !s.done);

  return (
    <header className="flex h-14 shrink-0 items-center gap-4 border-b border-white/5 bg-ink-900/80 px-4 backdrop-blur">
      <div className="flex items-center gap-2.5">
        <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg shadow-violet-900/50">
          <Wand2 className="h-4 w-4 text-white" />
        </div>
        <h1 className="hidden text-sm font-semibold tracking-tight text-zinc-100 sm:block">Object Magic Studio</h1>
      </div>

      <ol className="mx-auto hidden items-center gap-1 md:flex">
        {steps.map((s, i) => (
          <li key={s.label} className="flex items-center gap-1">
            <span
              className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition-colors duration-300 ${
                s.done ? 'text-violet-200' : i === activeIndex ? 'bg-white/5 text-zinc-100 ring-1 ring-white/10' : 'text-zinc-500'
              }`}
            >
              <span
                className={`grid h-4 w-4 place-items-center rounded-full text-[9px] font-bold transition-colors duration-300 ${
                  s.done ? 'bg-violet-500 text-white' : i === activeIndex ? 'bg-zinc-200 text-ink-900' : 'bg-ink-700 text-zinc-400'
                }`}
              >
                {s.done ? <Check className="h-2.5 w-2.5" strokeWidth={4} /> : i + 1}
              </span>
              {s.label}
            </span>
            {i < steps.length - 1 && <span className="h-px w-3 bg-white/10" />}
          </li>
        ))}
      </ol>

      <div className="ml-auto flex items-center gap-2 md:ml-0">
        <ModeBadge />
        {video && (
          <Button size="sm" variant="ghost" onClick={closeVideo} title="Start over with a new video">
            <RotateCcw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">New video</span>
          </Button>
        )}
      </div>
    </header>
  );
}
