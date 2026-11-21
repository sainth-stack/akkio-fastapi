import 'dotenv/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';

export const projectRoot = fileURLToPath(new URL('../../', import.meta.url));
const env = z.object({
  PORT: z.coerce.number().int().min(1).max(65535).default(5181),
  APP_ORIGIN: z.string().url().default('http://127.0.0.1:5180'),
  HEADLESS: z.enum(['true', 'false']).default('false'),
  GENERATION_TIMEOUT_MS: z.coerce.number().int().positive().default(180000),
  NAVIGATION_TIMEOUT_MS: z.coerce.number().int().positive().default(45000),
  DOWNLOAD_TIMEOUT_MS: z.coerce.number().int().positive().default(45000),
  MAX_IMAGE_BYTES: z.coerce.number().int().positive().default(26214400),
  MAX_PROMPT_LENGTH: z.coerce.number().int().min(1).max(10000).default(4000),
  RATE_LIMIT_PER_MINUTE: z.coerce.number().int().positive().default(10),
}).parse(process.env);
export const config = {
  port: env.PORT, origin: new URL(env.APP_ORIGIN).origin, headless: env.HEADLESS === 'true',
  generationTimeout: env.GENERATION_TIMEOUT_MS, navigationTimeout: env.NAVIGATION_TIMEOUT_MS,
  downloadTimeout: env.DOWNLOAD_TIMEOUT_MS, maxImageBytes: env.MAX_IMAGE_BYTES,
  maxPromptLength: env.MAX_PROMPT_LENGTH, rateLimit: env.RATE_LIMIT_PER_MINUTE,
  profileDir: path.resolve(projectRoot, process.env.GEMINI_PROFILE_DIR || '.local/gemini-profile'),
  imageDir: path.resolve(projectRoot, process.env.IMAGE_STORAGE_DIR || '.local/images'),
  jobDir: path.resolve(projectRoot, process.env.JOB_STORAGE_DIR || '.local/jobs'),
  diagnosticsDir: path.resolve(projectRoot, process.env.DIAGNOSTICS_DIR || '.local/diagnostics'),
};
export type Config = typeof config;