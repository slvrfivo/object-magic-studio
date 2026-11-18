import { Film, UploadCloud } from 'lucide-react';
import { useRef, useState } from 'react';
import { useStudioStore } from '../store/studioStore';
import { Spinner } from './ui';

const CHIPS = ['MP4 (H.264)', '≤ 10 seconds', '≤ 1920×1080', '≤ 100 MB'];

export function UploadDropzone() {
  const loadVideo = useStudioStore((s) => s.loadVideo);
  const busy = useStudioStore((s) => s.uploadState === 'validating');
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | null) => {
    const f = files?.[0];
    if (f && !busy) void loadVideo(f);
  };

  return (
    <div className="stage-bg flex flex-1 items-center justify-center rounded-2xl border border-white/5 p-4 sm:p-8">
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`group flex w-full max-w-xl cursor-pointer flex-col items-center rounded-3xl border-2 border-dashed px-6 py-12 text-center transition-all duration-200 sm:py-16 ${
          dragging
            ? 'scale-[1.01] border-violet-400 bg-violet-500/10'
            : 'border-white/10 bg-ink-900/60 hover:border-violet-400/50 hover:bg-ink-850'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,.mp4"
          className="sr-only"
          disabled={busy}
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = '';
          }}
        />
        <div className="mb-5 grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 ring-1 ring-violet-400/30 transition-transform duration-300 group-hover:-translate-y-1">
          {busy ? <Spinner className="h-6 w-6" /> : <UploadCloud className="h-7 w-7 text-violet-300" />}
        </div>
        <h2 className="text-lg font-semibold text-zinc-100">{busy ? 'Checking your video…' : 'Drop a short video to begin'}</h2>
        <p className="mt-1 text-sm text-zinc-400">
          or <span className="text-violet-300 underline-offset-4 group-hover:underline">browse your files</span>
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {CHIPS.map((c) => (
            <span key={c} className="flex items-center gap-1.5 rounded-full bg-white/5 px-3 py-1 text-xs text-zinc-400 ring-1 ring-white/5">
              <Film className="h-3 w-3" />
              {c}
            </span>
          ))}
        </div>
      </label>
    </div>
  );
}
