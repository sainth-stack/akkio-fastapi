import { describe, it, expect, vi } from 'vitest';
import { EventEmitter } from 'node:events';
import type { BrowserContext, Page } from 'playwright';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import request from 'supertest';
import { config } from '../../backend/src/config';
import { GeminiSessionService } from '../../backend/src/services/geminiSession.service';
import { GenerationJobManager } from '../../backend/src/jobs/generationJobManager';
import { createApp } from '../../backend/src/app';
import { GenerationError } from '../../backend/src/errors';

function setup(verify = vi.fn(async (_page: Page) => {})) {
  const events = new EventEmitter();
  const page = { goto: vi.fn(async () => {}), url: () => 'https://gemini.google.com/app' } as unknown as Page;
  const context = Object.assign(events, { pages: () => [page] }) as unknown as BrowserContext;
  const browser = { launch: vi.fn(async (_headless: boolean) => context), close: vi.fn(async () => { events.emit('close'); }) };
  return { session: new GeminiSessionService(config, browser, verify), browser, verify, events };
}
describe('manual Google sign-in lifecycle', () => {
  it('keeps Chromium open, deduplicates launch, closes only after verification', async () => {
    const { session, browser, verify } = setup();
    expect(session.start().status).toBe('OPENING');
    session.start();
    await expect.poll(() => session.get().status).toBe('WAITING_FOR_LOGIN');
    expect(browser.launch).toHaveBeenCalledExactlyOnceWith(false);
    expect(browser.close).not.toHaveBeenCalled();
    expect(session.busy).toBe(true);
    expect((await session.confirm()).status).toBe('AUTHENTICATED');
    expect(verify).toHaveBeenCalledOnce();
    expect(browser.close).toHaveBeenCalledOnce();
    expect(session.busy).toBe(false);
  });
  it('failed authentication leaves the browser open for login and retry', async () => {
    const verify = vi.fn(async (_page: Page) => { throw new GenerationError('AUTH_REQUIRED', 'Sign in on Google’s page.'); });
    const { session, browser } = setup(verify);
    session.start(); await expect.poll(() => session.get().status).toBe('WAITING_FOR_LOGIN');
    expect((await session.confirm()).error?.code).toBe('AUTH_REQUIRED');
    expect(session.get().status).toBe('WAITING_FOR_LOGIN');
    expect(browser.close).not.toHaveBeenCalled();
    await session.cancel(); expect(session.get().status).toBe('IDLE'); expect(browser.close).toHaveBeenCalledOnce();
  });
  it('reports manual window closure and allows a fresh sign-in', async () => {
    const { session, events } = setup();
    session.start(); await expect.poll(() => session.get().status).toBe('WAITING_FOR_LOGIN');
    events.emit('close');
    expect(session.get().error?.code).toBe('LOGIN_WINDOW_CLOSED');
    await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
    expect(session.start().status).toBe('OPENING');
    await expect.poll(() => session.get().status).toBe('WAITING_FOR_LOGIN');
    await session.cancel();
  });
  it('API rejects credentials, blocks generation during sign-in, and supports cancel', async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'prism-session-'));
    const { session } = setup();
    const manager = new GenerationJobManager({ generate: async () => { throw new Error('must not run'); } }, directory);
    await manager.initialize();
    const app = createApp(manager, config, session);
    try {
      await request(app).post('/api/gemini-session/start').send({ email: 'fake@example.test', password: 'not-a-real-secret' }).expect(400);
      expect(session.get().status).toBe('IDLE');
      await request(app).post('/api/gemini-session/start').send({}).expect(202);
      await expect.poll(() => session.get().status).toBe('WAITING_FOR_LOGIN');
      await request(app).post('/api/image-generation').send({ prompt: 'A city' }).expect(409);
      expect((await request(app).get('/api/gemini-session').expect(200)).body.status).toBe('WAITING_FOR_LOGIN');
      await request(app).post('/api/gemini-session/cancel').send({}).expect(200);
      expect(session.get().status).toBe('IDLE');
    } finally { await session.close(); await manager.close(); await rm(directory, { recursive: true, force: true }); }
  });
});