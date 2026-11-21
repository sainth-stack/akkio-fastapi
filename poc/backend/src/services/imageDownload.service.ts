import type { Locator, Page } from 'playwright';
import { GenerationError } from '../errors';
import { validateImage } from './imageStorage.service';

export async function downloadOriginal(page: Page, image: Locator, response: Locator, timeout: number, maxBytes: number) {
  await image.hover();
  let scope: Locator | Page = response;
  let button = response.getByRole('button', { name: /download(?: (?:full[- ]size|original|image))?/i }).first();
  if (!await button.isVisible()) {
    await image.click();
    scope = page.getByRole('dialog').filter({ has: page.getByRole('img') }).first();
    if (!await scope.isVisible()) throw new GenerationError('DOWNLOAD_CONTROL_NOT_FOUND', 'Gemini’s original-image download control could not be found. No thumbnail will be substituted.');
    button = scope.getByRole('button', { name: /download(?: (?:full[- ]size|original|image))?/i }).first();
  }
  if (!await button.isVisible()) throw new GenerationError('DOWNLOAD_CONTROL_NOT_FOUND', 'Gemini’s original-image download control could not be found. Its interface may have changed.');
  try {
    const [download] = await Promise.all([page.waitForEvent('download', { timeout }), button.click({ timeout })]);
    const stream = await download.createReadStream();
    if (!stream) throw new Error('no stream');
    const chunks: Buffer[] = []; let size = 0;
    for await (const chunk of stream) {
      size += chunk.length;
      if (size > maxBytes) { stream.destroy(); await download.cancel(); throw new GenerationError('INVALID_IMAGE', 'The downloaded image exceeds the configured size limit.'); }
      chunks.push(Buffer.from(chunk));
    }
    if (await download.failure()) throw new Error('download failure');
    const bytes = Buffer.concat(chunks);
    const extension = download.suggestedFilename().split('.').pop()?.toLowerCase();
    const mime = ({ png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp' } as Record<string, string>)[extension || ''] || 'application/octet-stream';
    await validateImage(bytes, mime, maxBytes);
    return { bytes, mime };
  } catch (error) {
    if (error instanceof GenerationError) throw error;
    throw new GenerationError('DOWNLOAD_FAILED', 'The original generated image could not be downloaded. No image was saved.');
  }
}