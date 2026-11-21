import { chromium, type BrowserContext, type Browser } from 'playwright';
import sharp from 'sharp';

export class FixtureBrowser {
  private browser?: Browser;
  private context?: BrowserContext;
  constructor(private scenario: 'prompt' | 'success' | 'auth' | 'timeout' | 'download' | 'invalid' | 'text' = 'prompt') {}
  async launch() {
    const png = await sharp({ create: { width: 512, height: 512, channels: 3, background: '#b1a1cf' } }).png().toBuffer();
    this.browser = await chromium.launch({ headless: true });
    this.context = await this.browser.newContext({ acceptDownloads: true });
    await this.context.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.hostname !== 'gemini.google.com') { await route.abort(); return; }
      if (url.pathname === '/fixture-preview.png') { await route.fulfill({ contentType: 'image/png', body: png }); return; }
      if (url.pathname === '/fixture-original.png') {
        if (url.searchParams.get('scenario') === 'download') { await route.abort(); return; }
        await route.fulfill({ contentType: 'image/png', headers: { 'content-disposition': 'attachment; filename="gemini-original.png"' }, body: url.searchParams.get('scenario') === 'invalid' ? Buffer.from('<html>error</html>') : png }); return;
      }
      await route.fulfill({ contentType: 'text/html', body: `<!doctype html><html lang="en"><body>
      ${this.scenario === 'auth' ? '<button>Sign in</button>' : '<button aria-label="Google Account: Test Account">Account</button>'}
      <textarea aria-label="Enter a prompt for Gemini"></textarea><button aria-label="Send message" onclick="send()">Send</button>
      <script>
        const scenario = ${JSON.stringify(this.scenario)};
        function send() {
          const value = document.querySelector('textarea').value;
          const mode = scenario === 'prompt' ? (/AUTH_FAILURE/.test(value) ? 'auth' : /TIMEOUT/.test(value) ? 'timeout' : /DOWNLOAD_FAILURE/.test(value) ? 'download' : /INVALID_IMAGE/.test(value) ? 'invalid' : /TEXT_ONLY/.test(value) ? 'text' : 'success') : scenario;
          if (mode === 'auth') { document.body.innerHTML = '<button>Sign in</button>'; return; }
          const stop = document.createElement('button'); stop.textContent = 'Stop response'; document.body.append(stop);
          if (mode === 'timeout') return;
          setTimeout(() => {
            stop.remove();
            const response = document.createElement('model-response');
            response.innerHTML = mode === 'text' ? '<p>A description, not an image.</p><button>Copy</button>' : '<img alt="Generated image" src="/fixture-preview.png"/><button aria-label="Download full size">Download</button><button>Good response</button>';
            if (mode !== 'text') response.querySelector('button').onclick = () => { location.href = '/fixture-original.png?scenario=' + mode; };
            document.body.append(response);
          }, 1200);
        }
      </script></body></html>` });
    });
    return this.context;
  }
  async close() { await this.context?.close(); await this.browser?.close(); }
}