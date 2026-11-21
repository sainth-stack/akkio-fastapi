import { config } from './config';
import { BrowserManager } from './automation/browserManager';
import { GeminiPage, GEMINI_URL } from './automation/geminiPage';

const manager = new BrowserManager(config);
try {
  const context = await manager.launch(false);
  const page = context.pages()[0] || await context.newPage();
  await page.goto(GEMINI_URL, { waitUntil: 'domcontentloaded' });
  console.info('Sign in manually in Google Chrome. Never enter credentials in this terminal. After Gemini opens, return here and press Enter to verify and close the browser.');
  process.stdin.resume();
  await new Promise<void>(resolve => process.stdin.once('data', () => resolve()));
  await new GeminiPage(page, config.generationTimeout).ready();
  console.info('Authenticated Gemini session verified and saved to the private browser profile.');
} catch { console.error('Authentication was not verified. Finish manual sign-in and run npm run auth again.'); process.exitCode = 1; }
finally { await manager.close(); process.stdin.pause(); }