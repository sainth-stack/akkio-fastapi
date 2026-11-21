import { LogIn, Check, LoaderCircle } from 'lucide-react';
import type { GeminiSession } from '../../../shared/types';

export function GeminiSignIn({ session, pending, disabled, onAction }: { session: GeminiSession; pending: boolean; disabled: boolean; onAction: (action: 'start' | 'confirm' | 'cancel') => void }) {
  const waiting = session.status === 'WAITING_FOR_LOGIN';
  const running = ['OPENING', 'WAITING_FOR_LOGIN', 'VERIFYING'].includes(session.status);
  return <section className="panel sign-in-panel" style={{ marginBottom: 16 }} aria-label="Gemini sign-in"><div className="section-heading"><span className="eyebrow">GOOGLE SESSION</span>{session.status === 'AUTHENTICATED' && <span className="pill">VERIFIED</span>}</div>
    <p className="description">{running ? 'In the opened Chromium window, click Google’s Sign in and enter your email and password directly on Google’s page. Complete any security checks, then return here to verify.' : session.status === 'AUTHENTICATED' ? 'Your sign-in was verified and saved on this machine. You can now generate images.' : 'Connect your Gemini account before generating. Your email and password are entered only on Google’s page, never in this app.'}</p>
    <div className="image-actions">{running ? <><button className="text-button" disabled={pending || session.status === 'VERIFYING'} onClick={() => onAction('cancel')}>Cancel sign-in</button><button className="secondary-button" disabled={!waiting || pending} onClick={() => onAction('confirm')}>{pending || !waiting ? <LoaderCircle size={15} className="spin"/> : <Check size={15}/>} {session.status === 'OPENING' ? 'Opening Google…' : session.status === 'VERIFYING' ? 'Verifying sign-in…' : 'I have signed in'}</button></> : <button className="secondary-button" disabled={disabled || pending} onClick={() => onAction('start')}><LogIn size={15}/>{session.status === 'AUTHENTICATED' ? 'Open Gemini sign-in' : 'Sign in to Gemini'}</button>}</div>
    {running && <p className="muted small" role="status">The browser stays open until sign-in is verified or you cancel.</p>}{session.error && <p className="failed description" role="alert">{session.error.message}</p>}
  </section>;
}