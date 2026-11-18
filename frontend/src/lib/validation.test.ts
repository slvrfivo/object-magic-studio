import { describe, expect, it } from 'vitest';
import { validateFileBasics, validateVideoInfo } from './validation';

describe('validateFileBasics', () => {
  it('accepts mp4 by mime type or extension', () => {
    expect(validateFileBasics({ name: 'a.mp4', type: 'video/mp4', size: 1000 })).toBeNull();
    expect(validateFileBasics({ name: 'clip.MP4', type: '', size: 1000 })).toBeNull();
  });

  it('rejects other types, empty and oversized files', () => {
    expect(validateFileBasics({ name: 'a.mov', type: 'video/quicktime', size: 1000 })).toMatch(/MP4/);
    expect(validateFileBasics({ name: 'a.mp4', type: 'video/mp4', size: 0 })).toMatch(/empty/);
    expect(validateFileBasics({ name: 'a.mp4', type: 'video/mp4', size: 200 * 1024 * 1024 })).toMatch(/too large/);
  });
});

describe('validateVideoInfo', () => {
  it('accepts a 10s 1080p landscape or portrait clip', () => {
    expect(validateVideoInfo({ duration: 10.04, width: 1920, height: 1080 })).toBeNull();
    expect(validateVideoInfo({ duration: 5, width: 1080, height: 1920 })).toBeNull();
  });

  it('rejects long, unknown-duration, oversized and tiny videos', () => {
    expect(validateVideoInfo({ duration: 12, width: 640, height: 360 })).toMatch(/Maximum length/);
    expect(validateVideoInfo({ duration: Infinity, width: 640, height: 360 })).toMatch(/duration/);
    expect(validateVideoInfo({ duration: 5, width: 3840, height: 2160 })).toMatch(/too high/);
    expect(validateVideoInfo({ duration: 5, width: 1440, height: 1440 })).toMatch(/too high/);
    expect(validateVideoInfo({ duration: 5, width: 32, height: 32 })).toMatch(/too small/);
  });
});
