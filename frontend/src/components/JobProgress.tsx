import { CheckCircle2, CircleAlert, CircleSlash, X } from 'lucide-react';
import { describeJobError } from '../api/errors';
import type { Job } from '../types';
import { Button, ProgressBar, Spinner } from './ui';

export function JobProgress({ job, label, onCancel }: { job: Job; label: string; onCancel?: () => void }) {
  const active = job.status === 'queued' || job.status === 'running';
  const tone = job.status === 'failed' ? 'rose' : job.status === 'succeeded' ? 'emerald' : 'violet';
  return (
    <div className="animate-slide-up rounded-xl border border-white/5 bg-ink-800/70 p-3">
      <div className="mb-2 flex items-center gap-2 text-sm">
        {active && <Spinner className="h-3.5 w-3.5" />}
        {job.status === 'succeeded' && <CheckCircle2 className="h-4 w-4 text-emerald-400" />}
        {job.status === 'failed' && <CircleAlert className="h-4 w-4 text-rose-400" />}
        {job.status === 'cancelled' && <CircleSlash className="h-4 w-4 text-zinc-500" />}
        <span className="font-medium text-zinc-200">{label}</span>
        <span className="ml-auto font-mono text-xs text-zinc-400 tabular-nums">{Math.round(job.progress * 100)}%</span>
        {active && onCancel && (
          <Button size="sm" variant="ghost" className="!h-6 !px-1.5" onClick={onCancel} aria-label="Cancel">
            <X className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
      <ProgressBar value={job.progress} tone={tone} />
      <p className={`mt-2 text-xs ${job.status === 'failed' ? 'text-rose-300' : 'text-zinc-500'}`}>
        {job.status === 'failed' ? describeJobError(job.error) : job.status === 'queued' ? 'Waiting in queue…' : job.message}
      </p>
    </div>
  );
}
