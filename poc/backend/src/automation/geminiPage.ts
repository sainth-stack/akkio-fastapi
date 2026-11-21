import { type Page, type Locator } from 'playwright';
import { setTimeout as delay } from 'node:timers/promises';
import { GenerationError } from '../errors';

export const GEMINI_URL = 'https://gemini.google.com/app';
export class GeminiPage {
  constructor(readonly page: Page, private timeout: number) {}
  promptInput() { return this.page.getByRole('textbox', { name: /enter a prompt|ask gemini|message gemini/i }).or(this.page.locator('[contenteditable="true"][role="textbox"]')).first(); }
  async verifyAuthentication() {
    if (this.page.isClosed()) throw new GenerationError('BROWSER_CRASHED', 'The browser closed unexpectedly. Retry the generation.');
    const signIn = this.page.getByRole('button', { name: /^sign in$/i }).or(this.page.getByRole('link', { name: /^sign in$/i }));
    let visibleSignIn = false;
    for (let i = 0; i < await signIn.count(); i++) visibleSignIn ||= await signIn.nth(i).isVisible();
    if (/accounts\.google\.com/.test(this.page.url()) || visibleSignIn) {
      throw new GenerationError('AUTH_REQUIRED', 'Gemini requires manual sign-in. Click Sign in to Gemini in the app, enter your credentials on Google’s page, then click I have signed in to verify. Retry generation afterward.');
    }
    if (await this.page.locator('iframe[src*="recaptcha"], iframe[title*="reCAPTCHA"]').first().isVisible() || await this.page.getByText(/verify (?:that )?you are human|unusual traffic|verify it.s you/i).first().isVisible()) {
      throw new GenerationError('SECURITY_CHECK', 'Google requires a manual security check. Automation has stopped; complete it manually without bypassing Google security.');
    }
  }
  async ready() {
    try {
      await this.page.locator('[role="textbox"]:visible, textarea:visible, a[href*="accounts.google.com"]:visible, button:visible').first().waitFor({ timeout: 60000 });
    } catch { throw new GenerationError('GEMINI_UI_UNAVAILABLE', 'Gemini’s interface did not load within 60 seconds. Check the private screenshot diagnostics and network connection.'); }
    await this.verifyAuthentication();
    try { await this.promptInput().waitFor({ state: 'visible', timeout: 15000 }); }
    catch { await this.verifyAuthentication(); throw new GenerationError('PROMPT_NOT_FOUND', 'Gemini’s prompt input could not be found. Its interface may have changed.'); }
    const account = this.page.getByRole('button', { name: /google account|account.*@/i }).or(this.page.getByRole('link', { name: /google account|account.*@/i })).first();
    if (!await account.isVisible()) throw new GenerationError('AUTH_UNVERIFIED', 'An authenticated Google account could not be verified. Click Sign in to Gemini in the app and verify your sign-in before generating.');
  }
  responses() { return this.page.locator('model-response, [data-message-author-role="assistant"], [data-response-role="model"]'); }
  async submit(prompt: string) {
    const before = await this.responses().count();
    await this.promptInput().fill(`Generate an image for this prompt. Return the generated image, not just a description.\n\n${prompt}`);
    const send = this.page.getByRole('button', { name: /^send(?: message| prompt)?$|^submit$/i }).first();
    try {
      await send.waitFor({ state: 'visible', timeout: 10000 });
      await send.click({ timeout: 10000, noWaitAfter: true });
    } catch { throw new GenerationError('SUBMISSION_UNCERTAIN', 'The prompt submission could not be confirmed. It will not be retried automatically to avoid duplicate generations. Check Gemini before retrying.'); }
    return before;
  }
  async waitForGeneratedImage(previousResponses: number): Promise<{ image: Locator; response: Locator }> {
    let result: { image: Locator; response: Locator } | undefined;
    try {
      const check = async () => {
        await this.verifyAuthentication();
        const count = await this.responses().count();
        if (count <= previousResponses) return false;
        const response = this.responses().nth(count - 1);
        if (await response.getByText(/usage limit|try again tomorrow|can.t generate|cannot generate|not available in your country/i).first().isVisible()) {
          throw new GenerationError('GEMINI_RESTRICTION', 'Gemini declined the request or a usage restriction was reached. Automation has stopped.');
        }
        if (await this.page.getByRole('button', { name: /stop (?:response|generat)/i }).first().isVisible()) return false;
        const images = response.getByRole('img');
        for (let i = 0; i < await images.count(); i++) {
          const image = images.nth(i);
          const info = await image.evaluate((element: HTMLImageElement) => ({ complete: element.complete, width: element.naturalWidth, height: element.naturalHeight, alt: element.alt }));
          if (info.complete && info.width >= 256 && info.height >= 256 && !/avatar|logo|profile|icon/i.test(info.alt)) { result = { image, response }; return true; }
        }
        const completedText = await response.innerText();
        const hasActions = await response.getByRole('button', { name: /good response|bad response|copy/i }).count();
        if (completedText.trim() && hasActions && await images.count() === 0) throw new GenerationError('TEXT_ONLY_RESPONSE', 'Gemini returned text instead of an image. Try an explicit image-generation prompt.');
        return false;
      };
      const deadline = Date.now() + this.timeout;
      while (Date.now() < deadline) {
        if (await check()) return result!;
        await delay(Math.min(500, Math.max(1, deadline - Date.now())));
      }
      throw new GenerationError('GENERATION_TIMEOUT', 'Gemini did not finish generating an image before the timeout. Check Gemini before retrying to avoid duplicate requests.');
    } catch (error) {
      if (error instanceof GenerationError) throw error;
      await this.verifyAuthentication();
      throw new GenerationError('GENERATION_TIMEOUT', 'Gemini did not finish generating an image before the timeout. Check Gemini before retrying to avoid duplicate requests.');
    }
  }
}