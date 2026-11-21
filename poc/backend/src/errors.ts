export class GenerationError extends Error {
  constructor(public code: string, message: string) { super(message); }
}
export function publicError(error: unknown): { code: string; message: string } {
  return error instanceof GenerationError
    ? { code: error.code, message: error.message }
    : { code: 'AUTOMATION_FAILED', message: 'Generation failed unexpectedly. Check the private server diagnostics and retry.' };
}
export function logStep(jobId: string, step: string, details: Record<string, string | number> = {}) {
  console.info(JSON.stringify({ time: new Date().toISOString(), jobId, step, ...details }));
}