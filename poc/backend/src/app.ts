import express from 'express';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { z } from 'zod';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { projectRoot, type Config } from './config';
import type { GenerationJobManager } from './jobs/generationJobManager';
import { GenerationError, publicError } from './errors';
import { imageFilename } from './services/imageStorage.service';
import type { GeminiSessionService } from './services/geminiSession.service';

export function createApp(manager: GenerationJobManager, config: Config, session?: GeminiSessionService) {
  const app = express();
  app.disable('x-powered-by');
  app.use(helmet({ crossOriginResourcePolicy: { policy: 'same-site' } }));
  app.use((req, res, next) => {
    if (!['127.0.0.1', 'localhost'].includes(req.hostname)) { res.status(403).json({ error: { code: 'HOST_FORBIDDEN', message: 'Only local requests are permitted.' } }); return; }
    if (req.headers.origin && ![config.origin, `http://127.0.0.1:${config.port}`].includes(req.headers.origin)) {
      res.status(403).json({ error: { code: 'ORIGIN_FORBIDDEN', message: 'This origin is not allowed.' } }); return;
    }
    next();
  });
  app.use(express.json({ limit: '20kb' }));
  app.get('/api/gemini-session', (_req, res) => res.set('Cache-Control', 'no-store').json(session?.get() || { status: 'IDLE', error: null }));
  app.post('/api/gemini-session/:action', rateLimit({ windowMs: 60000, limit: config.rateLimit, legacyHeaders: false }), async (req, res, next) => {
    try {
      if (!session) { res.status(503).json({ error: { code: 'SESSION_UNAVAILABLE', message: 'Browser sign-in is not available on this server.' } }); return; }
      if (Object.keys(req.body || {}).length) { res.status(400).json({ error: { code: 'INVALID_SESSION_REQUEST', message: 'Enter credentials only on Google’s page. This endpoint does not accept credentials.' } }); return; }
      if (manager.busy) throw new GenerationError('JOB_BUSY', 'Wait for image generation to finish before opening sign-in.');
      switch (req.params.action) {
        case 'start': res.status(202).json(session.start()); return;
        case 'confirm': res.json(await session.confirm()); return;
        case 'cancel': res.json(await session.cancel()); return;
        default: res.status(404).json({ error: { code: 'NOT_FOUND', message: 'Sign-in action not found.' } });
      }
    } catch (error) { next(error); }
  });
  app.get('/api/health', (_req, res) => res.json({ status: 'ok', busy: manager.busy, provider: 'gemini-browser', authentication: 'verified-per-job' }));
  app.get('/api/image-generation', (_req, res) => { res.set('Cache-Control', 'no-store').json({ jobs: manager.list() }); });
  app.post('/api/image-generation', rateLimit({ windowMs: 60000, limit: config.rateLimit, standardHeaders: 'draft-7', legacyHeaders: false, message: { error: { code: 'RATE_LIMITED', message: 'Too many requests. Try again in a minute.' } } }), async (req, res, next) => {
    try {
      if (!req.is('application/json')) { res.status(415).json({ error: { code: 'INVALID_CONTENT_TYPE', message: 'Send an application/json request.' } }); return; }
      const body = z.object({ prompt: z.string().trim().min(1).max(config.maxPromptLength) }).strict().safeParse(req.body);
      if (!body.success) { res.status(400).json({ error: { code: 'INVALID_PROMPT', message: `Enter a prompt between 1 and ${config.maxPromptLength} characters.` } }); return; }
      const key = req.get('Idempotency-Key') || randomUUID();
      if (!z.string().uuid().safeParse(key).success) { res.status(400).json({ error: { code: 'INVALID_REQUEST_KEY', message: 'Idempotency-Key must be a UUID.' } }); return; }
      if (session?.busy) throw new GenerationError('SESSION_BUSY', 'Finish and verify Google sign-in before generating an image.');
      const job = await manager.submit(body.data.prompt, key);
      res.status(202).json({ jobId: job.jobId, status: job.status });
    } catch (error) { next(error); }
  });
  app.get('/api/image-generation/:jobId', (req, res) => {
    const job = manager.get(req.params.jobId);
    res.set('Cache-Control', 'no-store');
    if (!job) { res.status(404).json({ error: { code: 'JOB_NOT_FOUND', message: 'Generation job not found.' } }); return; }
    res.json(job);
  });
  app.get('/api/generated-images/:filename', (req, res, next) => {
    const filename = req.params.filename;
    if (!imageFilename.test(filename) || !manager.hasImage(filename)) {
      res.status(404).json({ error: { code: 'IMAGE_NOT_FOUND', message: 'Generated image not found.' } }); return;
    }
    res.set('Cache-Control', 'private, max-age=31536000, immutable');
    if (req.query.download === '1') res.attachment(filename);
    res.sendFile(filename, { root: config.imageDir, dotfiles: 'deny' }, error => { if (error) next(error); });
  });
  app.use(express.static(path.join(projectRoot, 'dist/frontend')));
  app.get('/', (_req, res, next) => res.sendFile(path.join(projectRoot, 'dist/frontend/index.html'), error => { if (error) next(error); }));
  app.use((_req, res) => res.status(404).json({ error: { code: 'NOT_FOUND', message: 'Route not found.' } }));
  app.use(((error, _req, res, _next) => {
    const parsed = publicError(error);
    const status = error instanceof GenerationError ? (['JOB_BUSY', 'SESSION_BUSY', 'IDEMPOTENCY_CONFLICT'].includes(error.code) ? 409 : 503) : (error.type === 'entity.parse.failed' ? 400 : error.type === 'entity.too.large' ? 413 : error.status === 404 ? 404 : 500);
    res.status(status).json({ error: status === 400 ? { code: 'INVALID_JSON', message: 'Request body must be valid JSON.' } : parsed });
  }) as express.ErrorRequestHandler);
  return app;
}