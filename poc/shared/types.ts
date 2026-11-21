export const statuses = ['SUBMITTING', 'OPENING_GEMINI', 'GENERATING', 'DOWNLOADING', 'SAVING', 'COMPLETED', 'FAILED'] as const;
export type JobStatus = typeof statuses[number];
export type GenerationStatus = 'IDLE' | JobStatus;
export interface GeminiSession {
  status: 'IDLE' | 'OPENING' | 'WAITING_FOR_LOGIN' | 'VERIFYING' | 'AUTHENTICATED' | 'FAILED';
  error: { code: string; message: string } | null;
}
export interface GenerationJob {
  jobId: string;
  status: JobStatus;
  prompt: string;
  imageUrl: string | null;
  createdAt: string;
  updatedAt: string;
  error: { code: string; message: string } | null;
  metadata: { mime: string; bytes: number; width: number; height: number; sha256: string; source: 'gemini' } | null;
}