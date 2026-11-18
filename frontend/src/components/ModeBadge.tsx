// Mock / Live(SAM 2.1) 모드와 백엔드 상태를 명확히 표시하는 배지
import { Cloud, Cpu, FlaskConical, MonitorCog, PlugZap, TriangleAlert } from 'lucide-react';
import type { ReactNode } from 'react';
import { api } from '../api';
import { useStudioStore } from '../store/studioStore';
import { Spinner } from './ui';

function Badge({ tone, icon, children, title }: { tone: string; icon: ReactNode; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ring-1 ${tone}`} data-testid="mode-badge">
      {icon}
      {children}
    </span>
  );
}

export function ModeBadge() {
  const status = useStudioStore((s) => s.backendStatus);
  const error = useStudioStore((s) => s.backendError);
  const compute = useStudioStore((s) => s.compute);

  if (api.mode === 'mock') {
    return (
      <Badge
        tone="bg-amber-500/10 text-amber-300 ring-amber-500/25"
        icon={<FlaskConical className="h-3 w-3" />}
        title="Mock mode: masks and tracking are simulated, no model inference. Remove ?mode=mock to use SAM 2.1."
      >
        Mock mode
      </Badge>
    );
  }
  if (error) {
    return (
      <Badge tone="bg-rose-500/10 text-rose-300 ring-rose-500/25" icon={<PlugZap className="h-3 w-3" />} title={error}>
        Backend offline
      </Badge>
    );
  }
  if (!status || status.modelState === 'loading' || status.modelState === 'not_loaded') {
    return (
      <Badge tone="bg-sky-500/10 text-sky-300 ring-sky-500/25" icon={<Spinner className="h-3 w-3" />}>
        Loading SAM 2.1…
      </Badge>
    );
  }
  if (status.modelState === 'error') {
    return (
      <Badge tone="bg-rose-500/10 text-rose-300 ring-rose-500/25" icon={<TriangleAlert className="h-3 w-3" />} title={status.modelError ?? ''}>
        Model failed to load
      </Badge>
    );
  }
  const DeviceIcon = status.device === 'cuda' ? MonitorCog : Cpu;
  const local = (
    <Badge
      tone="bg-emerald-500/10 text-emerald-300 ring-emerald-500/25"
      icon={<DeviceIcon className="h-3 w-3" />}
      title={`SAM 2.1 Hiera Tiny · float32 · ${status.deviceName ?? ''}\n${status.deviceReason ?? ''}\nPyTorch ${status.torchVersion ?? ''}`}
    >
      SAM 2.1 · {status.device?.toUpperCase()}
    </Badge>
  );
  const remote = status.remote;
  if (!remote?.configured) return local;
  const tone =
    remote.state === 'error'
      ? 'bg-rose-500/10 text-rose-300 ring-rose-500/25'
      : compute === 'remote'
        ? 'bg-violet-500/10 text-violet-200 ring-violet-500/30'
        : 'bg-white/5 text-zinc-400 ring-white/10';
  return (
    <span className="flex items-center gap-1.5">
      {local}
      <span
        title={`Remote GPU (${remote.provider}) · SAM 2.1 Hiera Tiny ${remote.precision ?? ''}\n${remote.lastError ?? (compute === 'remote' ? 'Selected for segmentation and tracking' : 'Available — select it in the Select object panel')}`}
        className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ring-1 ${tone}`}
        data-testid="remote-badge"
      >
        <Cloud className="h-3 w-3" />
        Remote {remote.gpu ?? 'GPU'}
        {remote.state === 'error' ? ' · error' : compute === 'remote' ? ' · in use' : ''}
      </span>
    </span>
  );
}
