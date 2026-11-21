import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e', timeout: 60000, fullyParallel: false, workers: 1,
  outputDir: process.env.LIVE_GEMINI === '1' ? 'live-test-results' : 'test-results',
  reporter: [['list'], ['html', { open: 'never' }]],
  use: { baseURL: 'http://127.0.0.1:5180', screenshot: 'only-on-failure', trace: 'retain-on-failure' },
  projects: [
    { name: 'fixture', testMatch: 'fixture.spec.ts' },
    { name: 'live', testMatch: 'live.spec.ts', timeout: 300000 },
  ],
  webServer: process.env.LIVE_GEMINI === '1' ? undefined : [
    { command: 'node --import tsx tests/fixtures/server.ts', url: 'http://127.0.0.1:5181/api/health', reuseExistingServer: false, timeout: 90000 },
    { command: 'npm run dev:frontend', url: 'http://127.0.0.1:5180', reuseExistingServer: false, timeout: 90000 },
  ],
});