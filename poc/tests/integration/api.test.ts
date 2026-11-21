import { beforeEach, afterEach, describe, it, expect } from 'vitest';
import request from 'supertest';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import sharp from 'sharp';
import { config } from '../../backend/src/config';
import { createApp } from '../../backend/src/app';
import { GenerationJobManager, type Generator } from '../../backend/src/jobs/generationJobManager';
import { LocalImageStorage } from '../../backend/src/services/imageStorage.service';
import { GenerationError } from '../../backend/src/errors';

let directory: string;
let manager: GenerationJobManager;
beforeEach(async () => { directory = await mkdtemp(path.join(os.tmpdir(), 'prism-api-')); });
afterEach(async () => { await manager?.close(); await rm(directory, { recursive: true, force: true }); });
async function setup(generator?: Generator) {
  const settings = { ...config, jobDir: path.join(directory, 'jobs'), imageDir: path.join(directory, 'images'), rateLimit: 100 };
  const storage = new LocalImageStorage(settings.imageDir, settings.maxImageBytes);
  manager = new GenerationJobManager(generator || { generate: async (_prompt, _id, progress) => {
    await progress('OPENING_GEMINI'); await progress('GENERATING'); await progress('DOWNLOADING'); await progress('SAVING');
    return storage.save(await sharp({ create: { width: 512, height: 512, channels: 3, background: '#abc' } }).png().toBuffer(), 'image/png');
  } }, settings.jobDir);
  await manager.initialize(); return { app: createApp(manager, settings), settings };
}
describe('API and persistent jobs (fixture generator)', () => {
  it('completes only after saving, serves the same validated bytes, restores metadata', async () => {
    const { app, settings } = await setup();
    const response = await request(app).post('/api/image-generation').send({ prompt: 'An imaginative city' }).expect(202);
    expect(response.body.status).toBe('SUBMITTING');
    await expect.poll(() => manager.get(response.body.jobId)?.status).toBe('COMPLETED');
    const job = (await request(app).get(`/api/image-generation/${response.body.jobId}`).expect(200)).body;
    const image = await request(app).get(job.imageUrl).expect(200).expect('Content-Type', /image\/png/);
    expect(image.body.length).toBe(job.metadata.bytes);
    await request(app).get(`${job.imageUrl}?download=1`).expect('Content-Disposition', /attachment/);
    const restored = new GenerationJobManager({ generate: async () => { throw new Error('not called'); } }, settings.jobDir);
    await restored.initialize(); expect(restored.get(job.jobId)).toEqual(job);
  });
  it.each(['', ' ', 'x'.repeat(4001)])('rejects invalid prompts', async prompt => {
    const { app } = await setup(); await request(app).post('/api/image-generation').send({ prompt }).expect(400);
  });
  it('rejects malformed JSON, extra fields, traversal, unknown jobs, and foreign origins', async () => {
    const { app } = await setup();
    await request(app).post('/api/image-generation').send({ prompt: 'ok', url: 'https://evil.example' }).expect(400);
    await request(app).post('/api/image-generation').set('Content-Type', 'application/json').send('{').expect(400);
    await request(app).get('/api/generated-images/%2e%2e%2f.env').expect(404);
    await request(app).get('/api/image-generation/missing').expect(404);
    await request(app).post('/api/image-generation').set('Origin', 'https://evil.example').send({ prompt: 'ok' }).expect(403);
    await request(app).get('/api/health').expect(200);
  });
  it('deduplicates request keys and serializes distinct requests', async () => {
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    let calls = 0;
    const { app } = await setup({ generate: async () => { calls++; await gate; throw new GenerationError('AUTH_REQUIRED', 'Manual sign-in required.'); } });
    const key = randomUUID();
    const first = await request(app).post('/api/image-generation').set('Idempotency-Key', key).send({ prompt: 'one' }).expect(202);
    const same = await request(app).post('/api/image-generation').set('Idempotency-Key', key).send({ prompt: 'one' }).expect(202);
    expect(same.body.jobId).toBe(first.body.jobId);
    await request(app).post('/api/image-generation').set('Idempotency-Key', key).send({ prompt: 'different' }).expect(409);
    await request(app).post('/api/image-generation').send({ prompt: 'two' }).expect(409);
    expect(calls).toBe(1); release();
    await expect.poll(() => manager.get(first.body.jobId)?.status).toBe('FAILED');
  });
  it.each(['AUTH_REQUIRED', 'GENERATION_TIMEOUT', 'DOWNLOAD_FAILED', 'INVALID_IMAGE', 'STORAGE_FAILED', 'BROWSER_CRASHED'])('never reports success for %s', async code => {
    const { app } = await setup({ generate: async () => { throw new GenerationError(code, 'Understandable failure'); } });
    const result = await request(app).post('/api/image-generation').send({ prompt: 'test' }).expect(202);
    await expect.poll(() => manager.get(result.body.jobId)?.status).toBe('FAILED');
    const job = manager.get(result.body.jobId)!;
    expect(job.error?.code).toBe(code); expect(job.imageUrl).toBeNull(); expect(job.metadata).toBeNull();
  });
});