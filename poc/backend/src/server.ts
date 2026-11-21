import { config } from './config';
import { createApp } from './app';
import { GenerationJobManager } from './jobs/generationJobManager';
import { GeminiAutomationService } from './services/geminiAutomation.service';
import { LocalImageStorage } from './services/imageStorage.service';
import { GeminiSessionService } from './services/geminiSession.service';

const manager = new GenerationJobManager(new GeminiAutomationService(config, new LocalImageStorage(config.imageDir, config.maxImageBytes)), config.jobDir);
const session = new GeminiSessionService(config);
await manager.initialize();
const server = createApp(manager, config, session).listen(config.port, '127.0.0.1', () => console.info(`Gemini POC backend: http://127.0.0.1:${config.port}`));
for (const signal of ['SIGINT', 'SIGTERM'] as const) process.once(signal, () => {
  server.close();
  void Promise.all([manager.close(), session.close()]).finally(() => process.exit(0));
});