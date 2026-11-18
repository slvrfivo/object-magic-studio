// 이펙트 사이드바: Neon Outline / Color Pop / Background Blur
import { Aperture, Palette, Sparkles, Zap } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';
import { NEON_PRESETS } from '../lib/effectParams';
import { useStudioStore } from '../store/studioStore';
import type { EffectId } from '../types';
import { Panel, Slider, Switch } from './ui';

function EffectCard({
  id,
  name,
  description,
  icon: Icon,
  accent,
  children,
}: {
  id: EffectId;
  name: string;
  description: string;
  icon: LucideIcon;
  accent: string;
  children: ReactNode;
}) {
  const enabled = useStudioStore((s) => s.effects[id].enabled);
  const updateEffect = useStudioStore((s) => s.updateEffect);
  return (
    <div
      className={`rounded-xl border transition-all duration-200 ${
        enabled ? 'border-violet-400/30 bg-ink-800 shadow-lg shadow-violet-950/30' : 'border-white/5 bg-ink-850 hover:border-white/10'
      }`}
    >
      <div className="flex items-center gap-3 p-3">
        <div className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg ${accent}`}>
          <Icon className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-zinc-100">{name}</p>
          <p className="truncate text-[11px] text-zinc-500">{description}</p>
        </div>
        <Switch checked={enabled} label={`Toggle ${name}`} onChange={(v) => updateEffect(id, { enabled: v })} />
      </div>
      <div className={`grid transition-[grid-template-rows] duration-300 ease-out ${enabled ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}`}>
        <div className="overflow-hidden">
          <div className="space-y-3 px-3 pb-3">{children}</div>
        </div>
      </div>
    </div>
  );
}

export function EffectsPanel() {
  const effects = useStudioStore((s) => s.effects);
  const hasMask = useStudioStore((s) => !!s.currentMask || !!s.trackedMasks);
  const updateEffect = useStudioStore((s) => s.updateEffect);

  return (
    <Panel title="Effects" icon={<Sparkles className="h-3.5 w-3.5" />}>
      {!hasMask && (
        <p className="mb-3 rounded-lg bg-white/[0.03] px-3 py-2 text-[11px] text-zinc-500 ring-1 ring-white/5">
          Select an object to preview effects. Settings can be adjusted at any time.
        </p>
      )}
      <div className="space-y-2">
        <EffectCard id="neon" name="Neon Outline" description="Glowing outline around the object" icon={Zap} accent="bg-cyan-500/15 text-cyan-300">
          <Slider label="Intensity" value={effects.neon.intensity} onChange={(v) => updateEffect('neon', { intensity: v })} fill={effects.neon.color} />
          <Slider
            label="Thickness"
            value={effects.neon.thickness}
            min={1}
            max={10}
            step={1}
            format={(v) => `${v}`}
            onChange={(v) => updateEffect('neon', { thickness: v })}
            fill={effects.neon.color}
          />
          <div>
            <p className="mb-1.5 text-xs text-zinc-400">Color</p>
            <div className="flex items-center gap-2">
              {NEON_PRESETS.map((c) => (
                <button
                  key={c}
                  aria-label={`Neon color ${c}`}
                  onClick={() => updateEffect('neon', { color: c })}
                  className={`h-6 w-6 rounded-full transition-transform hover:scale-110 ${
                    effects.neon.color === c ? 'ring-2 ring-white ring-offset-2 ring-offset-ink-800' : ''
                  }`}
                  style={{ background: c, boxShadow: `0 0 10px ${c}80` }}
                />
              ))}
              <label className="relative ml-auto h-6 w-6 cursor-pointer overflow-hidden rounded-full ring-1 ring-white/20" title="Custom color">
                <input
                  type="color"
                  value={effects.neon.color}
                  onChange={(e) => updateEffect('neon', { color: e.target.value })}
                  className="absolute -inset-2 h-10 w-10 cursor-pointer"
                />
              </label>
            </div>
          </div>
        </EffectCard>

        <EffectCard id="colorPop" name="Color Pop" description="Object in color, background in grayscale" icon={Palette} accent="bg-fuchsia-500/15 text-fuchsia-300">
          <Slider label="Intensity" value={effects.colorPop.intensity} onChange={(v) => updateEffect('colorPop', { intensity: v })} />
        </EffectCard>

        <EffectCard id="blur" name="Background Blur" description="Soft focus behind the object" icon={Aperture} accent="bg-sky-500/15 text-sky-300">
          <Slider label="Intensity" value={effects.blur.intensity} onChange={(v) => updateEffect('blur', { intensity: v })} />
        </EffectCard>
      </div>
    </Panel>
  );
}
