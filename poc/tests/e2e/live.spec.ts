import { test, expect } from '@playwright/test';
import { validateImage } from '../../backend/src/services/imageStorage.service';

test('LIVE: real Gemini generation, saved original, and frontend preview', async ({ page, request }) => {
  test.skip(process.env.LIVE_GEMINI !== '1', 'Explicit LIVE_GEMINI=1 and manually authenticated dedicated profile required');
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await page.getByRole('textbox').fill('Generate an image of a small glass greenhouse on a distant planet beneath a lavender sky, cinematic architectural photography, no text.');
  await page.getByRole('button', { name: 'Generate image', exact: true }).click();
  await expect.poll(() => page.evaluate(() => localStorage.getItem('prism.jobId'))).not.toBeNull();
  const id = await page.evaluate(() => localStorage.getItem('prism.jobId'));
  let result;
  await expect.poll(async () => {
    result = await (await request.get(`/api/image-generation/${id}`)).json();
    if (result.status === 'FAILED') throw new Error(`${result.error.code}: ${result.error.message}`);
    return result.status;
  }, { timeout: 240000, intervals: [1500] }).toBe('COMPLETED');
  const job = await (await request.get(`/api/image-generation/${id}`)).json();
  const imageResponse = await request.get(job.imageUrl);
  expect(imageResponse.ok()).toBe(true);
  const bytes = await imageResponse.body();
  await validateImage(bytes, imageResponse.headers()['content-type'], 26214400);
  await test.info().attach('actual-gemini-image', { body: bytes, contentType: job.metadata.mime });
  await expect(page.getByRole('img', { name: /generated image:/i })).toBeVisible();
  await page.screenshot({ path: 'test-results/live-studio.png', fullPage: true });
});