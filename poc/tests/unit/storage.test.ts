import { describe, it, expect } from 'vitest';
import sharp from 'sharp';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { validateImage, LocalImageStorage, imageFilename } from '../../backend/src/services/imageStorage.service';

describe('image validation and storage', () => {
  it.each(['png', 'jpeg', 'webp'] as const)('validates and preserves original %s bytes', async format => {
    const bytes = await sharp({ create: { width: 512, height: 512, channels: 3, background: '#abc' } }).toFormat(format).toBuffer();
    const directory = await mkdtemp(path.join(os.tmpdir(), 'prism-storage-'));
    try {
      const stored = await new LocalImageStorage(directory, 1000000).save(bytes, `image/${format}`);
      expect(stored.metadata.width).toBe(512);
      const filename = stored.imageUrl.split('/').pop()!;
      expect(imageFilename.test(filename)).toBe(true);
      expect(await readFile(path.join(directory, filename))).toEqual(bytes);
      expect(stored.imageUrl).not.toContain(directory);
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
  it.each([Buffer.alloc(0), Buffer.from('<html>Login required</html>'), Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])])('rejects empty, HTML, and truncated images', async bytes => {
    await expect(validateImage(bytes, 'image/png', 10000)).rejects.toMatchObject({ code: 'INVALID_IMAGE' });
  });
  it('rejects MIME mismatch, oversized files, and thumbnails', async () => {
    const bytes = await sharp({ create: { width: 64, height: 64, channels: 3, background: '#abc' } }).png().toBuffer();
    await expect(validateImage(bytes, 'image/jpeg', 10000)).rejects.toThrow();
    await expect(validateImage(bytes, 'image/png', 1)).rejects.toThrow();
    await expect(validateImage(bytes, 'image/png', 10000)).rejects.toThrow();
  });
});