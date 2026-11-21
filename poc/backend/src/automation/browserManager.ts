import { chromium, type BrowserContext } from 'playwright';
import { mkdir, open, unlink, type FileHandle } from 'node:fs/promises';
import path from 'node:path';
import type { Config } from '../config';
import { GenerationError } from '../errors';

export class BrowserManager {
  private context: BrowserContext | null = null;
  private lock: FileHandle | null = null;
  constructor(private config: Config) {}
  async launch(headless = this.config.headless) {
    await mkdir(this.config.profileDir, { recursive: true, mode: 0o700 });
    try { this.lock = await open(path.join(this.config.profileDir, '.poc-lock'), 'wx', 0o600); }
    catch { throw new GenerationError('PROFILE_BUSY', 'The Gemini browser profile is already in use. Close the sign-in browser or other POC process first.'); }
    try {
      this.context = await chromium.launchPersistentContext(this.config.profileDir, {
        channel: 'chrome',
        headless,
        acceptDownloads: true,
        viewport: { width: 1440, height: 1000 },
        locale: 'en-US',
        timeout: this.config.navigationTimeout,
        ignoreDefaultArgs: ['--enable-automation'],
      });
      this.context.setDefaultTimeout(15000);
      this.context.setDefaultNavigationTimeout(this.config.navigationTimeout);
      return this.context;
    } catch (error) {
      await this.close();
      const detail = error instanceof Error ? error.message : '';
      throw new GenerationError(
        'BROWSER_LAUNCH_FAILED',
        /executable|channel|chrome/i.test(detail)
          ? 'Google Chrome could not launch. Install Google Chrome (not Chromium) and retry sign-in.'
          : 'Google Chrome could not launch. Quit any POC Chrome window, then retry sign-in.',
      );
    }
  }
  async close() {
    const context = this.context; this.context = null;
    try { await context?.close(); }
    finally {
      if (this.lock) { await this.lock.close(); this.lock = null; await unlink(path.join(this.config.profileDir, '.poc-lock')).catch(() => {}); }
    }
  }
}