// 공용 UI 프리미티브
import type { ButtonHTMLAttributes, CSSProperties, ReactNode } from 'react';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';

const VARIANTS: Record<Variant, string> = {
  primary:
    'bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-violet-900/40 hover:brightness-110 active:brightness-95',
  secondary: 'bg-ink-700 text-zinc-100 hover:bg-ink-600 border border-white/5',
  ghost: 'text-zinc-300 hover:bg-white/5 hover:text-white',
  danger: 'bg-rose-500/15 text-rose-300 hover:bg-rose-500/25 border border-rose-500/20',
};

export function Button({
  variant = 'secondary',
  size = 'md',
  className = '',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' | 'lg' }) {
  const sizes = { sm: 'h-8 px-2.5 text-xs gap-1.5', md: 'h-9 px-3.5 text-sm gap-2', lg: 'h-11 px-5 text-sm gap-2' };
  return (
    <button
      {...props}
      className={`inline-flex items-center justify-center rounded-lg font-medium transition-all duration-150 disabled:pointer-events-none disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-violet-400 ${sizes[size]} ${VARIANTS[variant]} ${className}`}
    />
  );
}

export function Panel({ title, icon, actions, children }: { title: string; icon?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="border-b border-white/5 px-4 py-4">
      <header className="mb-3 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-zinc-400">
          {icon}
          {title}
        </h2>
        {actions}
      </header>
      {children}
    </section>
  );
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={`relative h-5 w-9 shrink-0 rounded-full transition-colors duration-200 focus-visible:outline-2 focus-visible:outline-violet-400 ${checked ? 'bg-violet-500' : 'bg-ink-600'}`}
    >
      <span
        className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200 ${checked ? 'translate-x-4' : ''}`}
      />
    </button>
  );
}

export function Slider({
  label,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  format = (v) => `${Math.round(v * 100)}%`,
  onChange,
  disabled,
  fill,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
  disabled?: boolean;
  fill?: string;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="block">
      <div className="mb-1.5 flex justify-between text-xs">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono text-zinc-300 tabular-nums">{format(value)}</span>
      </div>
      <input
        type="range"
        className="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ '--val': `${pct}%`, ...(fill ? { '--range-fill': fill } : {}) } as CSSProperties}
      />
    </label>
  );
}

export function ProgressBar({ value, tone = 'violet' }: { value: number; tone?: 'violet' | 'rose' | 'emerald' }) {
  const tones = {
    violet: 'from-violet-500 to-fuchsia-500',
    rose: 'from-rose-500 to-rose-400',
    emerald: 'from-emerald-500 to-teal-400',
  };
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-700">
      <div
        className={`h-full rounded-full bg-gradient-to-r ${tones[tone]} transition-[width] duration-300 ease-out`}
        style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }}
      />
    </div>
  );
}

export function Spinner({ className = 'h-4 w-4' }: { className?: string }) {
  return <span className={`inline-block animate-spin rounded-full border-2 border-white/20 border-t-white ${className}`} />;
}
