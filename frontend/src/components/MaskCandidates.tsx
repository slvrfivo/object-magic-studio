// 한 번 클릭의 SAM 2.1 마스크 후보 3개를 나란히 보여 주고 선택 (선택한 마스크로 추적 시작)
import { Layers } from 'lucide-react';
import { useEffect, useRef } from 'react';
import type { MaskCandidate } from '../types';
import { isJobActive, useStudioStore } from '../store/studioStore';

const THUMB_W = 112;

/** 현재 프레임 위에 후보 마스크를 칠한 작은 미리보기 */
function CandidateThumb({ c, aspect }: { c: MaskCandidate; aspect: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    const ctx = cv?.getContext('2d');
    if (!cv || !ctx) return;
    const { width: w, height: h } = cv;
    ctx.fillStyle = '#18181b';
    ctx.fillRect(0, 0, w, h);
    // 메인 화면의 영상(프롬프트 프레임에 맞춰져 있음)을 배경으로 사용
    const v = document.querySelector('video');
    if (v && v.readyState >= 2) {
      try {
        ctx.globalAlpha = 0.6;
        ctx.drawImage(v, 0, 0, w, h);
        ctx.globalAlpha = 1;
      } catch {
        /* 배경 없이 마스크만 표시 */
      }
    }
    const m = c.mask;
    const tmp = document.createElement('canvas');
    tmp.width = m.width;
    tmp.height = m.height;
    const t = tmp.getContext('2d');
    if (!t) return;
    const img = t.createImageData(m.width, m.height);
    for (let i = 0; i < m.data.length; i++) {
      if (m.data[i]) {
        img.data[i * 4] = 167;
        img.data[i * 4 + 1] = 139;
        img.data[i * 4 + 2] = 250;
        img.data[i * 4 + 3] = 210;
      }
    }
    t.putImageData(img, 0, 0);
    ctx.drawImage(tmp, 0, 0, w, h);
  }, [c]);
  return <canvas ref={ref} width={THUMB_W} height={Math.round(THUMB_W / aspect)} className="block w-full rounded" />;
}

export function MaskCandidates() {
  const candidates = useStudioStore((s) => s.candidates);
  const selected = useStudioStore((s) => s.selectedCandidate);
  const best = useStudioStore((s) => s.bestCandidate);
  const video = useStudioStore((s) => s.video);
  const tracking = useStudioStore((s) => isJobActive(s.trackingJob));
  const selectCandidate = useStudioStore((s) => s.selectCandidate);
  if (!candidates || candidates.length < 2 || !video) return null;
  const aspect = video.previewWidth / video.previewHeight;

  return (
    <div className="mt-3 animate-fade-in" data-testid="mask-candidates">
      <p className="mb-1 flex items-center gap-1.5 text-[10px] font-semibold tracking-wider text-zinc-500 uppercase">
        <Layers className="h-3 w-3" /> Mask options · single click
      </p>
      <p className="mb-2 text-[11px] text-zinc-500">One click can mean a part or the whole object. Pick the mask you mean — tracking starts from it.</p>
      <div className="grid grid-cols-3 gap-1.5">
        {candidates.map((c, i) => (
          <button
            key={i}
            onClick={() => selectCandidate(i)}
            disabled={tracking}
            aria-pressed={selected === i}
            title={`Option ${i + 1}: covers ${(c.coverage * 100).toFixed(1)}% of the frame · SAM score ${c.score.toFixed(2)}`}
            className={`rounded-md p-1 text-left ring-1 transition-colors disabled:opacity-50 ${
              selected === i ? 'bg-violet-500/15 ring-violet-400/60' : 'ring-white/10 hover:bg-white/[0.04]'
            }`}
            data-testid={`mask-candidate-${i}`}
          >
            <CandidateThumb c={c} aspect={aspect} />
            <span className="mt-1 flex items-center justify-between gap-1 px-0.5 font-mono text-[10px]">
              <span className={selected === i ? 'text-violet-200' : 'text-zinc-400'}>{(c.coverage * 100).toFixed(1)}%</span>
              {i === best ? <span className="rounded bg-white/10 px-1 text-[9px] text-zinc-300">SAM pick</span> : <span className="text-zinc-600">{c.score.toFixed(2)}</span>}
            </span>
          </button>
        ))}
      </div>
      {selected !== best && <p className="mt-1.5 text-[11px] text-violet-300/90">Tracking will start from option {selected + 1} instead of SAM's pick.</p>}
    </div>
  );
}
