import { describe, expect, it, vi } from 'vitest';
import { ApiError } from './errors';
import { createHttpApi } from './httpApi';
import { resolveApiMode } from './mode';

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const req = {
  frameIndex: 3,
  points: [
    { id: 'a', x: 0.5, y: 0.25, label: 1 as const },
    { id: 'b', x: 0.1, y: 0.9, label: 0 as const },
  ],
  maskWidth: 4,
  maskHeight: 2,
};

describe('createHttpApi.segmentFrame', () => {
  it('sends normalized points and decodes the RLE mask', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      json(200, { mask: { width: 4, height: 2, counts: [1, 2, 3, 2] }, metrics: { device: 'cpu', inferenceMs: 120 } }),
    );
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    const r = await api.segmentFrame('sid', req);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/sessions/sid/segment');
    expect(JSON.parse(init!.body as string)).toEqual({
      frameIndex: 3,
      points: [
        { x: 0.5, y: 0.25, label: 1 },
        { x: 0.1, y: 0.9, label: 0 },
      ],
      maskWidth: 4,
      maskHeight: 2,
    });
    expect(Array.from(r.mask.data)).toEqual([0, 1, 1, 0, 0, 0, 1, 1]);
    expect(r.metrics.device).toBe('cpu');
    expect(r.metrics.roundTripMs).toBeGreaterThanOrEqual(0);
  });

  it('rejects masks with unexpected dimensions', async () => {
    const api = createHttpApi('/api', (async () => json(200, { mask: { width: 2, height: 1, counts: [2] }, metrics: { device: 'cpu' } })) as unknown as typeof fetch);
    await expect(api.segmentFrame('sid', req)).rejects.toThrow(/unexpected dimensions/);
  });
});

describe('createHttpApi errors (no silent mock fallback)', () => {
  it('maps backend error codes', async () => {
    const api = createHttpApi('/api', (async () =>
      json(503, { error: { code: 'gpu_oom', message: 'GPU out of memory' } })) as unknown as typeof fetch);
    const err = await api.segmentFrame('sid', req).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe('gpu_oom');
  });

  it('maps model_unavailable', async () => {
    const api = createHttpApi('/api', (async () =>
      json(503, { error: { code: 'model_unavailable', message: 'still loading' } })) as unknown as typeof fetch);
    await expect(api.segmentFrame('sid', req)).rejects.toMatchObject({ code: 'model_unavailable', message: 'still loading' });
  });

  it('maps network failures and proxy 502s to network errors', async () => {
    const down = createHttpApi('/api', (async () => {
      throw new TypeError('fetch failed');
    }) as unknown as typeof fetch);
    await expect(down.getStatus()).rejects.toMatchObject({ code: 'network' });
    const proxy = createHttpApi('/api', (async () => new Response('Bad Gateway', { status: 502 })) as unknown as typeof fetch);
    await expect(proxy.getStatus()).rejects.toMatchObject({ code: 'network' });
    // Vite 개발 프록시: 백엔드 연결 실패 시 본문 없는 500
    const vite = createHttpApi('/api', (async () => new Response('', { status: 500 })) as unknown as typeof fetch);
    await expect(vite.getStatus()).rejects.toMatchObject({ code: 'network' });
    // 백엔드가 직접 보낸 JSON 500은 그대로 전달
    const backend500 = createHttpApi('/api', (async () =>
      json(500, { error: { code: 'inference_failed', message: 'boom' } })) as unknown as typeof fetch);
    await expect(backend500.getStatus()).rejects.toMatchObject({ code: 'inference_failed' });
  });

  it('supports tracking and export in live mode', async () => {
    const api = createHttpApi('/api', vi.fn() as unknown as typeof fetch);
    expect(api.capabilities).toEqual({ tracking: true, export: true, gpuJobs: true });
  });
});

describe('createHttpApi tracking', () => {
  it('starts propagation with the prompt and frame budget', async () => {
    const fetchMock = vi.fn(async (_u: string, _i?: RequestInit) => json(200, { id: 'j1', kind: 'propagate', status: 'queued', progress: 0 }));
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    const job = await api.startPropagation('sid', { ...req, maxFrames: 30 });
    expect(job.id).toBe('j1');
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/sessions/sid/propagate');
    expect(JSON.parse(init!.body as string)).toEqual({
      frameIndex: 3,
      points: [
        { x: 0.5, y: 0.25, label: 1 },
        { x: 0.1, y: 0.9, label: 0 },
      ],
      maxFrames: 30,
    });
  });

  it('polls and cancels jobs via the jobs endpoints', async () => {
    const fetchMock = vi.fn(async (_u: string, _i?: RequestInit) => json(200, { id: 'j1', kind: 'propagate', status: 'running', progress: 0.5 }));
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    expect((await api.getJob('j1')).progress).toBe(0.5);
    await api.cancelJob('j1');
    expect(fetchMock.mock.calls.map((c) => [c[0], c[1]?.method])).toEqual([
      ['/api/jobs/j1', undefined],
      ['/api/jobs/j1/cancel', 'POST'],
    ]);
  });

  it('decodes per-frame tracked masks keyed by frame index', async () => {
    const fetchMock = vi.fn(async (_u: string) =>
      json(200, { width: 2, height: 1, frames: [{ index: 7, counts: [1, 1] }, { index: 8, counts: [0, 2] }] }),
    );
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    const masks = await api.fetchTrackedMasks('sid', 'j1', { width: 2, height: 1 });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/sessions/sid/tracks/j1/masks?width=2&height=1');
    expect([...masks.keys()]).toEqual([7, 8]);
    expect(Array.from(masks.get(7)!.data)).toEqual([0, 1]);
    expect(Array.from(masks.get(8)!.data)).toEqual([1, 1]);
  });
});

describe('createHttpApi export', () => {
  it('exports by reusing the tracking job masks', async () => {
    const fetchMock = vi.fn(async (_u: string, _i?: RequestInit) => json(200, { id: 'e1', kind: 'export', status: 'queued', progress: 0 }));
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    const effects = {
      neon: { enabled: true, intensity: 0.5, color: '#22d3ee', thickness: 3 },
      colorPop: { enabled: false, intensity: 1 },
      blur: { enabled: false, intensity: 0.5 },
    };
    await api.startExport('sid', { trackJobId: 't1', effects });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/sessions/sid/export');
    expect(JSON.parse(init!.body as string)).toEqual({ trackJobId: 't1', effects });
  });
});

describe('createHttpApi GPU jobs', () => {
  it('creates a GPU job with the prompt, frame budget and precision', async () => {
    const fetchMock = vi.fn(async (_u: string, _i?: RequestInit) =>
      json(200, { jobId: 'j', startFrame: 3, numFrames: 297, precision: 'fp16', packageBytes: 10, downloadUrl: '/api/gpu-jobs/j/package' }),
    );
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    const info = await api.createGpuJob('sid', { frameIndex: 3, points: req.points, maxFrames: 297, precision: 'fp16' });
    expect(info.numFrames).toBe(297);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/sessions/sid/gpu-jobs');
    expect(JSON.parse(init!.body as string)).toMatchObject({ frameIndex: 3, maxFrames: 297, precision: 'fp16' });
  });

  it('uploads a GPU result and surfaces validation errors', async () => {
    const ok = vi.fn(async (_u: string, _i?: RequestInit) => json(200, { id: 't', kind: 'propagate', status: 'succeeded', progress: 1 }));
    const file = new File([new Uint8Array(4)], 'r.zip', { type: 'application/zip' });
    await createHttpApi('/api', ok as unknown as typeof fetch).importGpuResult('sid', file);
    expect(ok.mock.calls[0][0]).toBe('/api/sessions/sid/gpu-results');
    expect(ok.mock.calls[0][1]?.body).toBeInstanceOf(FormData);
    const bad = createHttpApi('/api', (async () =>
      json(422, { error: { code: 'invalid_result', message: 'Result was produced for a different video (sha256 mismatch).' } })) as unknown as typeof fetch);
    await expect(bad.importGpuResult('sid', file)).rejects.toMatchObject({ message: expect.stringContaining('different video') });
  });
});

describe('createHttpApi timeouts and preflight', () => {
  it('aborts requests that never respond', async () => {
    vi.useFakeTimers();
    const hang = (async (_u: string, init?: RequestInit) =>
      new Promise((_, reject) => init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))))) as unknown as typeof fetch;
    const p = createHttpApi('/api', hang).getStatus().catch((e) => e);
    await vi.advanceTimersByTimeAsync(6000);
    const err = await p;
    vi.useRealTimers();
    expect(err).toMatchObject({ code: 'network' });
    expect(err.message).toMatch(/did not respond/);
  });

  it('checks /health before uploading so an offline backend fails fast', async () => {
    const calls: string[] = [];
    const f = (async (url: string) => {
      calls.push(url);
      throw new TypeError('fetch failed');
    }) as unknown as typeof fetch;
    const file = new File([new Uint8Array(10)], 'a.mp4', { type: 'video/mp4' });
    await expect(createHttpApi('/api', f).createSession(file, { width: 1, height: 1, duration: 1 })).rejects.toMatchObject({ code: 'network' });
    expect(calls).toEqual(['/api/health']);
  });
});

describe('createHttpApi.getStatus', () => {
  it('maps health to BackendStatus', async () => {
    const api = createHttpApi('/api', (async () =>
      json(200, {
        capabilities: { segmentation: true, tracking: false, export: false },
        model: {
          state: 'ready',
          error: null,
          loadMs: 900,
          warmup: { setImageMs: 1500, predictMs: 60 },
          device: { device: 'cpu', reason: 'CUDA not used', torchVersion: '2.x', gpu: { name: 'GeForce MX150' } },
        },
      })) as unknown as typeof fetch);
    const s = await api.getStatus();
    expect(s).toMatchObject({ mode: 'live', modelState: 'ready', device: 'cpu', deviceName: 'CPU' });
  });
});

describe('resolveApiMode', () => {
  it('prefers URL, then env, defaulting to live', () => {
    expect(resolveApiMode('?mode=mock', 'live')).toBe('mock');
    expect(resolveApiMode('', 'mock')).toBe('mock');
    expect(resolveApiMode('?mode=bogus', undefined)).toBe('live');
    expect(resolveApiMode('', undefined)).toBe('live');
  });
});

describe('remote GPU (Phase 5B-2)', () => {
  it('sends the chosen backend for segmentation and tracking', async () => {
    const fetchMock = vi.fn(async (url: string, _init?: RequestInit) =>
      url.endsWith('/segment')
        ? json(200, { mask: { width: 4, height: 2, counts: [8] }, metrics: { device: 'cuda', backend: 'remote', uploadMs: 900 } })
        : json(200, { id: 'j', kind: 'propagate', status: 'queued', progress: 0 }),
    );
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    const r = await api.segmentFrame('sid', { ...req, backend: 'remote' });
    expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string).backend).toBe('remote');
    expect(r.metrics.backend).toBe('remote');
    await api.startPropagation('sid', { ...req, maxFrames: 300, backend: 'local' });
    expect(JSON.parse(fetchMock.mock.calls[1][1]!.body as string)).toMatchObject({ maxFrames: 300, backend: 'local' });
  });

  it('omits backend when not chosen (server default)', async () => {
    const fetchMock = vi.fn(async () => json(200, { mask: { width: 4, height: 2, counts: [8] }, metrics: { device: 'cpu' } }));
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    await api.segmentFrame('sid', req);
    expect(JSON.parse((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body as string)).not.toHaveProperty('backend');
  });

  it('reports remote status from health and defaults to not configured', async () => {
    const health = (remote?: unknown) => ({
      capabilities: { segmentation: true, tracking: true, export: true },
      model: { state: 'ready', error: null, loadMs: 1, warmup: null, device: null },
      ...(remote ? { remote } : {}),
    });
    const remote = { configured: true, provider: 'modal', gpu: 'T4', state: 'idle', maxFrames: 300 };
    let api = createHttpApi('/api', (async () => json(200, health(remote))) as unknown as typeof fetch);
    expect((await api.getStatus()).remote).toEqual(remote);
    api = createHttpApi('/api', (async () => json(200, health())) as unknown as typeof fetch);
    expect((await api.getStatus()).remote).toEqual({ configured: false });
  });

  it('surfaces remote errors with a hint to switch to Local CPU', async () => {
    const { describeError } = await import('./errors');
    const api = createHttpApi('/api', (async () =>
      json(503, { error: { code: 'remote_unavailable', message: 'Modal authentication failed.' } })) as unknown as typeof fetch);
    const err = await api.segmentFrame('sid', { ...req, backend: 'remote' }).catch((e) => e);
    expect(err.code).toBe('remote_unavailable');
    expect(describeError(err)).toMatch(/Modal authentication failed\. Switch to Local CPU/);
    const budget = new ApiError('budget_exceeded', 'Daily remote GPU limit reached.');
    expect(describeError(budget)).toBe('Daily remote GPU limit reached.');
  });

  it('warms up the remote GPU', async () => {
    const fetchMock = vi.fn(async () => json(200, { configured: true, state: 'ready' }));
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    expect((await api.warmupRemote()).state).toBe('ready');
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/remote/warmup');
    expect(init.method).toBe('POST');
  });
});

describe('mask candidates (single click)', () => {
  it('decodes three candidates and the default selection', async () => {
    const fetchMock = vi.fn(async () =>
      json(200, {
        mask: { width: 4, height: 2, counts: [1, 2, 3, 2] },
        metrics: { device: 'cpu' },
        candidates: [
          { mask: { width: 4, height: 2, counts: [3, 1, 4] }, score: 0.6, coverage: 0.1 },
          { mask: { width: 4, height: 2, counts: [1, 2, 3, 2] }, score: 0.9, coverage: 0.4 },
          { mask: { width: 4, height: 2, counts: [0, 8] }, score: 0.3, coverage: 1 },
        ],
        selected: 1,
      }),
    );
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    const r = await api.segmentFrame('sid', req);
    expect(r.selected).toBe(1);
    expect(r.candidates).toHaveLength(3);
    expect(Array.from(r.candidates![0].mask.data)).toEqual([0, 0, 0, 1, 0, 0, 0, 0]);
    expect(Array.from(r.candidates![2].mask.data)).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('rejects candidates with wrong dimensions', async () => {
    const api = createHttpApi('/api', (async () =>
      json(200, {
        mask: { width: 4, height: 2, counts: [8] },
        metrics: { device: 'cpu' },
        candidates: [{ mask: { width: 2, height: 1, counts: [2] }, score: 1, coverage: 0 }],
        selected: 0,
      })) as unknown as typeof fetch);
    await expect(api.segmentFrame('sid', req)).rejects.toThrow(/unexpected dimensions/);
  });

  it('sends the chosen candidate for tracking and GPU jobs only when given', async () => {
    const fetchMock = vi.fn(async () => json(200, { id: 'j', kind: 'propagate', status: 'queued', progress: 0, jobId: 'x' }));
    const api = createHttpApi('/api', fetchMock as unknown as typeof fetch);
    const calls = fetchMock.mock.calls as unknown as [string, RequestInit][];
    await api.startPropagation('sid', { ...req, maxFrames: 30, maskCandidate: 2 });
    expect(JSON.parse(calls[0][1].body as string).maskCandidate).toBe(2);
    await api.startPropagation('sid', { ...req, maxFrames: 30 });
    expect(JSON.parse(calls[1][1].body as string)).not.toHaveProperty('maskCandidate');
    await api.createGpuJob('sid', { frameIndex: 0, points: req.points, maxFrames: 30, precision: 'fp16', maskCandidate: 0 });
    expect(JSON.parse(calls[2][1].body as string).maskCandidate).toBe(0);
  });
});
