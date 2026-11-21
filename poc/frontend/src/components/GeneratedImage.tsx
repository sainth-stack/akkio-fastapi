import { Download, Copy, ImagePlus, Aperture } from 'lucide-react';
import { useState } from 'react';
import type { GenerationJob, GenerationStatus } from '../../../shared/types';

export function GeneratedImage({ job, status }: { job: GenerationJob | null; status: GenerationStatus }) {
  const [copied, setCopied] = useState(false);
  const [imageError, setImageError] = useState(false);
  const completed = job?.status === 'COMPLETED' && job.imageUrl;
  return <section className="panel result-panel"><div className="section-heading"><span className="eyebrow">02 / THE CANVAS</span><span className="pill">{completed ? 'ORIGINAL IMAGE' : 'LIVE OUTPUT'}</span></div>
    {completed ? <><div className="image-wrap">{imageError ? <p role="alert">The saved image could not be loaded. Refresh to retry.</p> : <img key={job.imageUrl!} src={job.imageUrl!} alt={`Generated image: ${job.prompt}`} onError={() => setImageError(true)} onLoad={() => setImageError(false)}/>}</div><div className="image-meta"><span>{job.metadata?.width} × {job.metadata?.height}</span><span>{job.metadata?.mime.split('/')[1].toUpperCase()} · {((job.metadata?.bytes || 0) / 1024 / 1024).toFixed(2)} MB</span></div><div className="image-actions"><button className="secondary-button" onClick={() => { void navigator.clipboard.writeText(job.prompt).then(() => { setCopied(true); }, () => setCopied(false)); }}><Copy size={15}/>{copied ? 'Copied' : 'Copy prompt'}</button><a className="secondary-button" href={`${job.imageUrl}?download=1`} download><Download size={15}/>Download image</a></div></> : <div className="empty-canvas"><div className="canvas-grid"/><div className="empty-icon">{status === 'GENERATING' ? <Aperture size={33} className="spin"/> : <ImagePlus size={33}/>}</div><h3>{['SUBMITTING', 'OPENING_GEMINI', 'GENERATING', 'DOWNLOADING', 'SAVING'].includes(status) ? 'A little imagination is in progress' : 'Your next idea starts here'}</h3><p>No stock images. No stand-ins.<br/>Your actual Gemini creation will appear here.</p><span className="canvas-caption">IMAGINE SOMETHING THAT DOESN’T EXIST. YET.</span></div>}
  </section>;
}