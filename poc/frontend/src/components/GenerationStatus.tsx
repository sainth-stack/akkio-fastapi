import { Check, LoaderCircle, AlertCircle } from 'lucide-react';
import type { GenerationStatus as Status } from '../../../shared/types';

const labels: Record<Status, string> = { IDLE: 'Ready when you are', SUBMITTING: 'Submitting your request', OPENING_GEMINI: 'Opening Gemini & checking your session', GENERATING: 'Gemini is creating your image', DOWNLOADING: 'Downloading the original image', SAVING: 'Validating & saving your image', COMPLETED: 'Your image is ready', FAILED: 'Generation needs your attention' };
const steps: Status[] = ['OPENING_GEMINI', 'GENERATING', 'DOWNLOADING', 'SAVING', 'COMPLETED'];
export function GenerationStatus({ status }: { status: Status }) {
  const busy = status !== 'IDLE' && status !== 'COMPLETED' && status !== 'FAILED';
  return <div className={`generation-status ${status === 'FAILED' ? 'failed' : ''}`} role="status" aria-live="polite"><div className="status-title">{busy ? <LoaderCircle className="spin" size={16}/> : status === 'FAILED' ? <AlertCircle size={16}/> : <Check size={16}/>}<span>{labels[status]}</span></div><div className="steps" aria-hidden="true">{steps.map((step, index) => <span key={step} className={steps.indexOf(status) >= index ? 'active' : ''}/>)}</div></div>;
}