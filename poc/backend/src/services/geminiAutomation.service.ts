import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { Page } from 'playwright';
import type { BrowserContext } from 'playwright';
import type { Config } from '../config';
import type { Generator } from '../jobs/generationJobManager';
import type { JobStatus } from '../../../shared/types';
import type { ImageStorage } from './imageStorage.service';
import { BrowserManager } from '../automation/browserManager';
import { GeminiPage, GEMINI_URL } from '../automation/geminiPage';
import { downloadOriginal } from './imageDownload.service';
import { GenerationError, logStep, publicError } from '../errors';

export class GeminiAutomationService implements Generator {
  private browser: { launch(): Promise<BrowserContext>; close(): Promise<void> };
  constructor(private config: Config, private storage: ImageStorage, browser?: { launch(): Promise<BrowserContext>; close(): Promise<void> }) { this.browser = browser || new BrowserManager(config); }
  async generate(prompt: string, jobId: string, progress: (status: JobStatus) => Promise<void>) {
    let page: Page | undefined;
    try {
      await progress('OPENING_GEMINI');
      const context = await this.browser.launch();
      page = context.pages()[0] || await context.newPage();
      page.on('dialog', dialog => { logStep(jobId, 'DIALOG_DISMISSED', { type: dialog.type() }); void dialog.dismiss().catch(() => {}); });
      page.on('crash', () => logStep(jobId, 'BROWSER_CRASHED'));
      context.on('page', popup => { if (popup !== page) void popup.close().catch(() => {}); });
      for (let attempt = 1; attempt <= 2; attempt++) {
        try { await page.goto(GEMINI_URL, { waitUntil: 'domcontentloaded' }); break; }
        catch { logStep(jobId, 'NAVIGATION_RETRY', { attempt }); if (attempt === 2) throw new GenerationError('GEMINI_UNAVAILABLE', 'The Gemini website could not be reached. Check your network connection and try again.'); }
      }
      logStep(jobId, 'GEMINI_OPENED');
      const gemini = new GeminiPage(page, this.config.generationTimeout);
      await gemini.ready(); logStep(jobId, 'AUTHENTICATION_VERIFIED');
      const prior = await gemini.submit(prompt); logStep(jobId, 'PROMPT_SUBMITTED');
      await progress('GENERATING');
      const generated = await gemini.waitForGeneratedImage(prior); logStep(jobId, 'IMAGE_DETECTED');
      await progress('DOWNLOADING');
      const downloaded = await downloadOriginal(page, generated.image, generated.response, this.config.downloadTimeout, this.config.maxImageBytes);
      logStep(jobId, 'IMAGE_VALIDATED', { bytes: downloaded.bytes.length });
      await progress('SAVING');
      return await this.storage.save(downloaded.bytes, downloaded.mime);
    } catch (error) {
      await mkdir(this.config.diagnosticsDir, { recursive: true, mode: 0o700 }).catch(() => {});
      if (page && !page.isClosed()) {
        await page.screenshot({ path: path.join(this.config.diagnosticsDir, `${jobId}.png`), fullPage: true, timeout: 5000 }).catch(() => {});
      }
      await writeFile(path.join(this.config.diagnosticsDir, `${jobId}.json`), JSON.stringify({ jobId, at: new Date().toISOString(), error: publicError(error) }), { mode: 0o600 }).catch(() => {});
      throw error;
    } finally { await this.browser.close().catch(() => logStep(jobId, 'BROWSER_CLEANUP_FAILED')); }
  }
  async close() { await this.browser.close(); }
}