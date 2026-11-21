import type { GenerationJob, GeminiSession } from '../../../shared/types';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...init, signal: init?.signal || AbortSignal.timeout(15000) });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error?.message || 'The service is unavailable. Please retry.');
  return body;
}
export const imageApi = {
  submit: (prompt: string, key: string) => request<{ jobId: string; status: string }>('/api/image-generation', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': key }, body: JSON.stringify({ prompt }) }),
  get: (id: string) => request<GenerationJob>(`/api/image-generation/${encodeURIComponent(id)}`),
  history: () => request<{ jobs: GenerationJob[] }>('/api/image-generation'),
  session: () => request<GeminiSession>('/api/gemini-session'),
  sessionAction: (action: 'start' | 'confirm' | 'cancel') => request<GeminiSession>(`/api/gemini-session/${action}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(90000) }),
};