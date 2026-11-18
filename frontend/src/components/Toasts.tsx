import { CheckCircle2, CircleAlert, Info, X } from 'lucide-react';
import { useStudioStore } from '../store/studioStore';

const ICONS = {
  info: <Info className="h-4 w-4 text-sky-400" />,
  success: <CheckCircle2 className="h-4 w-4 text-emerald-400" />,
  error: <CircleAlert className="h-4 w-4 text-rose-400" />,
};

export function Toasts() {
  const toasts = useStudioStore((s) => s.toasts);
  const dismiss = useStudioStore((s) => s.dismissToast);
  return (
    <div className="pointer-events-none fixed right-4 bottom-4 left-4 z-50 flex flex-col items-end gap-2 sm:left-auto">
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.kind === 'error' ? 'alert' : 'status'}
          className={`pointer-events-auto flex w-full max-w-sm animate-slide-up gap-3 rounded-xl border bg-ink-800/95 p-3 shadow-2xl backdrop-blur ${
            t.kind === 'error' ? 'border-rose-500/30' : 'border-white/10'
          }`}
        >
          <div className="mt-0.5">{ICONS[t.kind]}</div>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-zinc-100">{t.title}</p>
            {t.message && <p className="mt-0.5 text-xs leading-relaxed text-zinc-400">{t.message}</p>}
          </div>
          <button onClick={() => dismiss(t.id)} className="h-fit text-zinc-500 hover:text-zinc-200" aria-label="Dismiss">
            <X className="h-4 w-4" />
          </button>
        </div>
      ))}
    </div>
  );
}
