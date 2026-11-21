import { useCallback, useEffect, useRef, useState } from 'react';
import { Aperture, ArrowUpRight, ShieldCheck, RotateCcw } from 'lucide-react';
import type { GenerationJob, GenerationStatus as Status, GeminiSession } from '../../../shared/types';
import { imageApi } from '../services/imageGeneration.api';
import { PromptInput } from '../components/PromptInput';
import { GenerationStatus } from '../components/GenerationStatus';
import { GeneratedImage } from '../components/GeneratedImage';
import { GenerationHistory } from '../components/GenerationHistory';
import { GeminiSignIn } from '../components/GeminiSignIn';

const active = (status: Status) => !['IDLE', 'COMPLETED', 'FAILED'].includes(status);
const readPending = (): { prompt: string; key: string } | null => {
  try { return JSON.parse(localStorage.getItem('prism.pending') || 'null'); } catch { return null; }
};
export function ImageGenerator() {
  const [prompt, setPrompt] = useState(() => readPending()?.prompt || '');
  const [job, setJob] = useState<GenerationJob | null>(null);
  const [jobId, setJobId] = useState<string | null>(() => localStorage.getItem('prism.jobId'));
  const [status, setStatus] = useState<Status>(jobId || readPending() ? 'SUBMITTING' : 'IDLE');
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [session, setSession] = useState<GeminiSession>({ status: 'IDLE', error: null });
  const [sessionPending, setSessionPending] = useState(false);
  const sessionLock = useRef(false);
  const signingIn = ['OPENING', 'WAITING_FOR_LOGIN', 'VERIFYING'].includes(session.status);
  const submitting = useRef(false);
  const refreshHistory = useCallback(() => { void imageApi.history().then(data => setJobs(data.jobs), () => {}); }, []);
  useEffect(refreshHistory, [refreshHistory]);
  useEffect(() => {
    let disposed = false; let timer: ReturnType<typeof setTimeout>;
    const pollSession = async () => {
      try { const result = await imageApi.session(); if (!disposed) setSession(result); } catch {}
      if (!disposed) timer = setTimeout(() => { void pollSession(); }, 1500);
    };
    void pollSession(); return () => { disposed = true; clearTimeout(timer); };
  }, []);
  const sessionAction = async (action: 'start' | 'confirm' | 'cancel') => {
    if (sessionLock.current) return;
    sessionLock.current = true; setSessionPending(true);
    try { setSession(await imageApi.sessionAction(action)); }
    catch (error) { setSession(previous => ({ ...previous, error: { code: 'SIGN_IN_FAILED', message: error instanceof Error ? error.message : 'Could not connect to the sign-in browser. Try again.' } })); }
    finally { sessionLock.current = false; setSessionPending(false); }
  };
  const generate = useCallback(async () => {
    if (submitting.current) return;
    const pending = readPending();
    const value = pending?.prompt || prompt.trim();
    if (!value) { setError('Enter a prompt before generating an image.'); return; }
    submitting.current = true;
    const key = pending?.key || crypto.randomUUID();
    localStorage.setItem('prism.pending', JSON.stringify({ prompt: value, key }));
    setStatus('SUBMITTING'); setError(null); setJob(null);
    try {
      const result = await imageApi.submit(value, key);
      localStorage.setItem('prism.jobId', result.jobId); localStorage.removeItem('prism.pending');
      setJobId(result.jobId);
    } catch (error) { setError(error instanceof Error ? error.message : 'Submission failed. Retry to reconnect to this request.'); setStatus('FAILED'); }
    finally { submitting.current = false; }
  }, [prompt]);
  useEffect(() => { if (!jobId && readPending()) void generate(); }, []);
  useEffect(() => {
    if (!jobId) return;
    let disposed = false; let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await imageApi.get(jobId);
        if (disposed) return;
        setJob(result); setStatus(result.status); setPrompt(result.prompt); setError(result.error?.message || null);
        if (!active(result.status)) { refreshHistory(); return; }
      } catch (error) { if (disposed) return; setError(error instanceof Error ? error.message : 'Connection interrupted. Reconnecting to your job…'); }
      timer = setTimeout(() => { void poll(); }, 1500);
    };
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [jobId, refreshHistory]);
  const retry = () => {
    if (job?.status === 'FAILED') { localStorage.removeItem('prism.pending'); localStorage.removeItem('prism.jobId'); setJobId(null); }
    void generate();
  };
  return <div className="app-shell"><header className="topbar"><a className="brand" href="/"><Aperture size={26}/><span>prism<span className="brand-period">.</span></span></a><span className="topbar-label">THE IMAGE STUDIO</span><a className="gemini-link" href="https://gemini.google.com/app" target="_blank" rel="noreferrer">Open Gemini<ArrowUpRight size={14}/></a></header>
    <main><section className="hero"><div className="hero-tag"><span className="dot"/> A SPACE FOR YOUR IMAGINATION</div><h1>Big ideas.<br/><span>Beautifully made.</span></h1><p>Turn a few words into something worth seeing.<br/>Real image generation. Powered by your Gemini session.</p><span className="hero-number">∞<small>POSSIBILITIES</small></span></section>
      <div className="workspace"><div className="input-column"><GeminiSignIn session={session} pending={sessionPending} disabled={active(status)} onAction={action => { void sessionAction(action); }}/><PromptInput prompt={prompt} setPrompt={setPrompt} busy={active(status) || signingIn || sessionPending} onGenerate={() => { void generate(); }}/><GenerationStatus status={status}/>{error && <div className="error-box" role="alert"><p>{error}</p>{!active(status) && !signingIn && !sessionPending && <button className="text-button" onClick={retry}><RotateCcw size={14}/>Retry generation</button>}</div>}<div className="trust-note"><ShieldCheck size={17}/><p>Your session stays on this machine.<br/><span>No passwords stored. No security checks bypassed.</span></p></div></div><GeneratedImage key={job?.jobId || 'empty'} job={job} status={status}/></div>
      <GenerationHistory jobs={jobs} disabled={active(status)} onSelect={selected => { localStorage.setItem('prism.jobId', selected.jobId); setJobId(selected.jobId); setJob(selected); setStatus(selected.status); setPrompt(selected.prompt); setError(selected.error?.message || null); }}/>
    </main><footer><span>PRISM / GEMINI IMAGE GENERATION POC</span><span>A little text. A whole new perspective.</span></footer></div>;
}