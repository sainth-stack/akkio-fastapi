import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.route('https://fonts.googleapis.com/**', route => route.abort()); });

test('empty prompt is disabled and rejected by API', async ({ page, request }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('button', { name: /generate image/i })).toBeDisabled();
  expect((await request.post('/api/image-generation', { data: { prompt: '' } })).status()).toBe(400);
});
test('full fixture workflow, concurrent rejection, refresh recovery, display, and original download', async ({ page, request }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await page.getByRole('textbox').fill('An ethereal city under a lavender sky');
  await page.getByRole('button', { name: /generate image/i }).click();
  await expect.poll(() => page.evaluate(() => localStorage.getItem('prism.jobId'))).not.toBeNull();
  const id = await page.evaluate(() => localStorage.getItem('prism.jobId'));
  expect((await request.post('/api/image-generation', { data: { prompt: 'Another city' } })).status()).toBe(409);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('img', { name: /generated image:/i })).toBeVisible({ timeout: 20000 });
  const job = await (await request.get(`/api/image-generation/${id}`)).json();
  expect(job.status).toBe('COMPLETED'); expect(job.metadata.sha256).toHaveLength(64);
  await expect(page.getByRole('img', { name: /generated image:/i })).toHaveAttribute('src', job.imageUrl);
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /download image/i }).click()]);
  expect(await download.failure()).toBeNull();
  await page.screenshot({ path: 'screenshots/fixture-studio.png', fullPage: true });
});
for (const [marker, message] of [['AUTH_FAILURE', 'manual sign-in'], ['TIMEOUT', 'before the timeout'], ['DOWNLOAD_FAILURE', 'could not be downloaded'], ['INVALID_IMAGE', 'not a supported image']] as const) {
  test(`shows actionable ${marker} failure without fake success`, async ({ page }) => {
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await page.getByRole('textbox').fill(`${marker}: a city`);
    await page.getByRole('button', { name: /generate image/i }).click();
    await expect(page.getByRole('alert')).toContainText(message, { timeout: 45000 });
    await expect(page.getByRole('button', { name: /retry generation/i })).toBeVisible();
    await expect(page.getByRole('img', { name: /generated image:/i })).toHaveCount(0);
  });
}