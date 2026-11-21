import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { config } from '../../backend/src/config';
import { createApp } from '../../backend/src/app';
import { GenerationJobManager } from '../../backend/src/jobs/generationJobManager';
import { GeminiAutomationService } from '../../backend/src/services/geminiAutomation.service';
import { LocalImageStorage } from '../../backend/src/services/imageStorage.service';
import { FixtureBrowser } from './browser';

const directory = await mkdtemp(path.join(os.tmpdir(), 'prism-e2e-'));
const settings = { ...config, imageDir: path.join(directory, 'images'), jobDir: path.join(directory, 'jobs'), diagnosticsDir: path.join(directory, 'diagnostics'), generationTimeout: 5000, downloadTimeout: 2500, rateLimit: 100 };
const manager = new GenerationJobManager(new GeminiAutomationService(settings, new LocalImageStorage(settings.imageDir, settings.maxImageBytes), new FixtureBrowser()), settings.jobDir);
await manager.initialize();
const server = createApp(manager, settings).listen(settings.port, '127.0.0.1', () => console.info('TEST FIXTURE SERVER — no live Gemini requests'));
for (const signal of ['SIGINT', 'SIGTERM'] as const) process.once(signal, () => { server.close(); void manager.close().then(() => rm(directory, { recursive: true, force: true })).finally(() => process.exit(0)); });