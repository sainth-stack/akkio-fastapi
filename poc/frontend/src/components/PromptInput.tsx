import { ArrowUpRight, Eraser, Sparkles } from 'lucide-react';

export function PromptInput({ prompt, setPrompt, busy, onGenerate }: { prompt: string; setPrompt: (value: string) => void; busy: boolean; onGenerate: () => void }) {
  return <section className="panel prompt-panel">
    <div className="section-heading"><span className="eyebrow">01 / YOUR IDEA</span><span className="small muted">Words into worlds</span></div>
    <label htmlFor="prompt">What do you want to create?</label>
    <p className="muted description">Describe your subject, atmosphere, lighting, and style.</p>
    <div className="textarea-wrap"><textarea id="prompt" placeholder="A glass greenhouse on a distant planet, soft lavender skies, cinematic light, architectural photography…" value={prompt} disabled={busy} onChange={event => setPrompt(event.target.value)} maxLength={4000} rows={7}/><div className="textarea-footer"><span><Sparkles size={13}/> Be as imaginative as you like</span><span>{prompt.length} / 4000</span></div></div>
    <div className="prompt-actions"><button className="text-button" disabled={busy || !prompt} onClick={() => setPrompt('')}><Eraser size={15}/> Clear prompt</button><button className="generate-button" disabled={busy || !prompt.trim()} onClick={onGenerate}>{busy ? 'Creating your image…' : 'Generate image'}<ArrowUpRight size={18}/></button></div>
    <div className="provider-note"><span className="dot"/> Powered by Gemini · Original browser download</div>
  </section>;
}