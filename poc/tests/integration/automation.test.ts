import { describe, it, expect } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { config } from '../../backend/src/config';
import { GeminiAutomationService } from '../../backend/src/services/geminiAutomation.service';
import { LocalImageStorage } from '../../backend/src/services/imageStorage.service';
import { FixtureBrowser } from '../fixtures/browser';

describe('real Chromium automation against intercepted Gemini fixture', () => {
  it.each(['success', 'auth', 'timeout', 'download', 'invalid', 'text'] as const)('handles %s without contacting Google', async scenario => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'prism-automation-'));
    const settings = { ...config, generationTimeout: 3500, downloadTimeout: 1500, diagnosticsDir: path.join(directory, 'diagnostics') };
    const service = new GeminiAutomationService(settings, new LocalImageStorage(path.join(directory, 'images'), settings.maxImageBytes), new FixtureBrowser(scenario));
    const stages: string[] = [];
    try {
      const generation = service.generate('A city', 'fixture-job', async status => { stages.push(status); });
      if (scenario === 'success') {
        const result = await generation;
        expect(result.metadata.width).toBe(512);
        expect(stages).toEqual(['OPENING_GEMINI', 'GENERATING', 'DOWNLOADING', 'SAVING']);
      } else {
        const codes = { auth: 'AUTH_REQUIRED', timeout: 'GENERATION_TIMEOUT', download: 'DOWNLOAD_FAILED', invalid: 'INVALID_IMAGE', text: 'TEXT_ONLY_RESPONSE' };
        await expect(generation).rejects.toMatchObject({ code: codes[scenario] });
      }
    } finally { await service.close(); await rm(directory, { recursive: true, force: true }); }
  });
});