import type { BrowserContext, Page } from 'playwright';
import type { GeminiSession } from '../../../shared/types';
import type { Config } from '../config';
import { BrowserManager } from '../automation/browserManager';
import { GeminiPage, GEMINI_URL } from '../automation/geminiPage';
import { GenerationError, publicError } from '../errors';

interface SessionBrowser { launch(headless: boolean): Promise<BrowserContext>; close(): Promise<void> }
export class GeminiSessionService {
  private state: GeminiSession = { status: 'IDLE', error: null };
  private browser: SessionBrowser;
  private context: BrowserContext | null = null;
  private operation: Promise<void> | null = null;
  private closing = false;
  constructor(config: Config, browser?: SessionBrowser, private verify: (page: Page) => Promise<void> = page => new GeminiPage(page, config.generationTimeout).ready()) {
    this.browser = browser || new BrowserManager(config);
  }
  get busy() { return ['OPENING', 'WAITING_FOR_LOGIN', 'VERIFYING'].includes(this.state.status); }
  get() { return structuredClone(this.state); }
  start() {
    if (this.closing) throw new GenerationError('SESSION_BUSY', 'The sign-in browser is closing. Try again shortly.');
    if (this.busy) return this.get();
    this.state = { status: 'OPENING', error: null };
    this.operation = this.open();
    return this.get();
  }
  private async open() {
    try {
      this.context = await this.browser.launch(false);
      this.context.on('close', () => {
        if (!this.closing) {
          this.state = { status: 'FAILED', error: { code: 'LOGIN_WINDOW_CLOSED', message: 'The Google sign-in window was closed. Open it again and verify sign-in before closing it.' } };
          this.closing = true;
          this.operation = this.browser.close().catch(() => {}).finally(() => { this.context = null; this.closing = false; });
        }
      });
      const page = this.context.pages()[0] || await this.context.newPage();
      await page.goto(GEMINI_URL, { waitUntil: 'domcontentloaded' });
      if (!this.closing && this.state.status === 'OPENING') this.state = { status: 'WAITING_FOR_LOGIN', error: null };
    } catch (error) {
      if (!this.closing) this.state = { status: 'FAILED', error: publicError(error) };
      await this.browser.close().catch(() => {});
    }
  }
  async confirm() {
    if (this.state.status !== 'WAITING_FOR_LOGIN' || this.closing) throw new GenerationError('SESSION_BUSY', 'Wait for the Google sign-in window to open before verifying.');
    this.state = { status: 'VERIFYING', error: null };
    this.operation = this.check();
    await this.operation;
    return this.get();
  }
  private async check() {
    try {
      const page = this.context?.pages().find(page => { try { return new URL(page.url()).origin === 'https://gemini.google.com'; } catch { return false; } });
      if (!page) throw new GenerationError('AUTH_REQUIRED', 'Finish signing in on Google’s page and return to Gemini before verifying.');
      await this.verify(page);
    } catch (error) {
      if (!this.closing && this.state.status === 'VERIFYING') this.state = { status: 'WAITING_FOR_LOGIN', error: publicError(error) };
      return;
    }
    this.closing = true;
    try {
      await this.browser.close();
      this.context = null;
      this.state = { status: 'AUTHENTICATED', error: null };
    } catch { this.state = { status: 'FAILED', error: { code: 'SESSION_SAVE_FAILED', message: 'The sign-in browser could not close safely. Close it manually and sign in again.' } }; }
    finally { this.closing = false; }
  }
  async cancel() {
    if (this.state.status === 'VERIFYING') throw new GenerationError('SESSION_BUSY', 'Sign-in verification is running. Wait for it to finish.');
    this.closing = true;
    try { await this.operation; await this.browser.close(); this.context = null; this.state = { status: 'IDLE', error: null }; }
    finally { this.closing = false; }
    return this.get();
  }
  async close() { await this.operation; await this.cancel(); }
}