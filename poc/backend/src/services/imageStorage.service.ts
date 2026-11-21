import { mkdir, writeFile, rename, readFile } from 'node:fs/promises';
import path from 'node:path';
import { randomUUID, createHash } from 'node:crypto';
import sharp from 'sharp';
import { GenerationError } from '../errors';
import type { GenerationJob } from '../../../shared/types';

export interface StoredImage { imageUrl: string; metadata: NonNullable<GenerationJob['metadata']> }
export interface ImageStorage { save(bytes: Buffer, mime: string): Promise<StoredImage> }
export const imageFilename = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.(png|jpg|webp)$/;
export async function validateImage(bytes: Buffer, mime: string, maxBytes: number) {
  if (!bytes.length || bytes.length > maxBytes) throw new GenerationError('INVALID_IMAGE', 'The image is empty or exceeds the configured size limit.');
  let format: 'png' | 'jpeg' | 'webp' | undefined;
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) format = 'png';
  else if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) format = 'jpeg';
  else if (bytes.toString('ascii', 0, 4) === 'RIFF' && bytes.toString('ascii', 8, 12) === 'WEBP') format = 'webp';
  const expected = format === 'jpeg' ? 'image/jpeg' : `image/${format}`;
  if (!format || mime.toLowerCase().split(';')[0].trim() !== expected) throw new GenerationError('INVALID_IMAGE', 'The download is not a supported image or its MIME type does not match.');
  try {
    const image = sharp(bytes, { limitInputPixels: 100000000, failOn: 'warning' });
    const metadata = await image.metadata();
    await image.stats();
    if (!metadata.width || !metadata.height || metadata.width < 256 || metadata.height < 256) throw new Error('too small');
    return { format, mime: expected, width: metadata.width, height: metadata.height };
  } catch { throw new GenerationError('INVALID_IMAGE', 'The downloaded image is damaged or too small to be an original generated image.'); }
}
export class LocalImageStorage implements ImageStorage {
  constructor(private directory: string, private maxBytes: number) {}
  async save(bytes: Buffer, mime: string): Promise<StoredImage> {
    const info = await validateImage(bytes, mime, this.maxBytes);
    const filename = `${randomUUID()}.${info.format === 'jpeg' ? 'jpg' : info.format}`;
    const target = path.join(this.directory, filename);
    try {
      await mkdir(this.directory, { recursive: true, mode: 0o700 });
      await writeFile(`${target}.tmp`, bytes, { flag: 'wx', mode: 0o600 });
      await rename(`${target}.tmp`, target);
      const saved = await readFile(target);
      if (!saved.equals(bytes)) throw new Error('verification failed');
    } catch { throw new GenerationError('STORAGE_FAILED', 'The validated image could not be saved. Check storage permissions and available space.'); }
    return { imageUrl: `/api/generated-images/${filename}`, metadata: { mime: info.mime, bytes: bytes.length, width: info.width, height: info.height, sha256: createHash('sha256').update(bytes).digest('hex'), source: 'gemini' } };
  }
}