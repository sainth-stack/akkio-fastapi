import { randomUUID } from 'node:crypto';
import { mkdir, readdir, readFile, writeFile, rename } from 'node:fs/promises';
import path from 'node:path';
import type { GenerationJob, JobStatus } from '../../../shared/types';
import type { StoredImage } from '../services/imageStorage.service';
import { GenerationError, publicError, logStep } from '../errors';

export interface Generator {
  generate(prompt: string, jobId: string, progress: (status: JobStatus) => Promise<void>): Promise<StoredImage>;
  close?(): Promise<void>;
}
export class GenerationJobManager {
  private jobs = new Map<string, GenerationJob>();
  private keys = new Map<string, string>();
  private active: string | null = null;
  private starting = false;
  private shuttingDown = false;
  private running: Promise<void> | null = null;
  constructor(private generator: Generator, private directory: string) {}
  async initialize() {
    await mkdir(this.directory, { recursive: true, mode: 0o700 });
    for (const file of await readdir(this.directory)) {
      if (!/^[0-9a-f-]{36}\.json$/.test(file)) continue;
      const job = JSON.parse(await readFile(path.join(this.directory, file), 'utf8')) as GenerationJob & { requestKey?: string };
      if (!job.jobId || !job.prompt || !job.status) continue;
      if (!['COMPLETED', 'FAILED'].includes(job.status)) {
        job.status = 'FAILED';
        job.error = { code: 'SERVER_RESTARTED', message: 'The server restarted during generation. Submit a new request to retry.' };
        job.updatedAt = new Date().toISOString();
        await this.persist(job, job.requestKey);
      }
      if (job.requestKey) this.keys.set(job.requestKey, job.jobId);
      const { requestKey: _, ...publicJob } = job;
      this.jobs.set(job.jobId, publicJob);
    }
  }
  get busy() { return this.active !== null || this.starting; }
  hasImage(filename: string) { return [...this.jobs.values()].some(job => job.status === 'COMPLETED' && job.imageUrl === `/api/generated-images/${filename}`); }
  get(id: string) { const job = this.jobs.get(id); return job ? structuredClone(job) : undefined; }
  list() { return [...this.jobs.values()].map(job => structuredClone(job)).sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, 100); }
  async submit(prompt: string, key: string) {
    const prior = this.keys.get(key);
    if (prior) {
      const job = this.jobs.get(prior)!;
      if (job.prompt !== prompt) throw new GenerationError('IDEMPOTENCY_CONFLICT', 'This request key was already used for a different prompt.');
      return this.get(prior)!;
    }
    if (this.shuttingDown) throw new GenerationError('SHUTTING_DOWN', 'The service is shutting down. Retry shortly.');
    if (this.busy) throw new GenerationError('JOB_BUSY', 'Another generation is running. Wait for it to finish before retrying.');
    this.starting = true;
    const now = new Date().toISOString();
    const job: GenerationJob = { jobId: randomUUID(), status: 'SUBMITTING', prompt, createdAt: now, updatedAt: now, imageUrl: null, error: null, metadata: null };
    try { await this.persist(job, key); }
    catch { throw new GenerationError('STORAGE_FAILED', 'The generation job could not be saved.'); }
    finally { this.starting = false; }
    this.jobs.set(job.jobId, job);
    this.keys.set(key, job.jobId);
    this.active = job.jobId;
    this.running = new Promise<void>(resolve => setImmediate(() => { void this.run(job, key).finally(resolve); }));
    return this.get(job.jobId)!;
  }
  private async persist(job: GenerationJob, key?: string) {
    const target = path.join(this.directory, `${job.jobId}.json`);
    await writeFile(`${target}.tmp`, JSON.stringify({ ...job, requestKey: key }), { mode: 0o600 });
    await rename(`${target}.tmp`, target);
  }
  private async run(job: GenerationJob, key: string) {
    const update = async (status: JobStatus) => {
      if (status === 'COMPLETED' || status === 'FAILED') throw new Error('Terminal status is owned by the manager');
      job.status = status; job.updatedAt = new Date().toISOString();
      await this.persist(job, key); logStep(job.jobId, status);
    };
    try {
      const stored = await this.generator.generate(job.prompt, job.jobId, update);
      if (!stored.imageUrl || !stored.metadata.bytes) throw new GenerationError('INVALID_IMAGE', 'No saved image was returned.');
      const completed = { ...job, ...stored, status: 'COMPLETED' as const, updatedAt: new Date().toISOString() };
      await this.persist(completed, key);
      this.jobs.set(job.jobId, completed);
      logStep(job.jobId, 'COMPLETED', { bytes: stored.metadata.bytes });
    } catch (error) {
      job.status = 'FAILED'; job.error = publicError(error); job.updatedAt = new Date().toISOString();
      job.imageUrl = null; job.metadata = null;
      try { await this.persist(job, key); } catch { logStep(job.jobId, 'JOB_PERSIST_FAILED'); }
      logStep(job.jobId, 'FAILED', { code: job.error.code });
    } finally { this.active = null; }
  }
  async close() { this.shuttingDown = true; await this.generator.close?.(); await this.running; }
}