import React, { useCallback, useEffect, useState } from 'react';
import { apiFetch } from '../utils/api';
import './SampleTemplateView.css';

function formatCell(cell) {
    if (cell == null) return '—';
    return String(cell);
}

function SectionBlock({ section }) {
    if (!section) return null;
    if (section.type === 'callout') {
        return (
            <section className="sample-template__section">
                <h4>{section.title}</h4>
                <p className="sample-template__callout">{section.body}</p>
            </section>
        );
    }
    if (section.type === 'table') {
        return (
            <section className="sample-template__section">
                <h4>{section.title}</h4>
                <table>
                    <thead>
                        <tr>
                            {(section.columns || []).map((c) => (
                                <th key={c}>{c}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {(section.rows || []).map((row, ri) => (
                            <tr key={ri}>
                                {row.map((cell, ci) => (
                                    <td key={ci}>{formatCell(cell)}</td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </section>
        );
    }
    return null;
}

export default function SampleTemplateView({ projectName, onDownloadSampleData }) {
    const [activeId, setActiveId] = useState('');
    const [payload, setPayload] = useState(null);
    const [loading, setLoading] = useState(false);

    const load = useCallback(
        async (templateId) => {
            if (!projectName) {
                setPayload(null);
                return;
            }
            setLoading(true);
            try {
                const q = templateId ? `?template_id=${encodeURIComponent(templateId)}` : '';
                const res = await apiFetch(`/app-builder/projects/${projectName}/sample-report-template${q}`);
                if (res.ok) {
                    const data = await res.json();
                    setPayload(data);
                    if (!templateId && data.template_id) {
                        setActiveId(data.template_id);
                    }
                }
            } catch {
                setPayload(null);
            } finally {
                setLoading(false);
            }
        },
        [projectName],
    );

    useEffect(() => {
        load(activeId || undefined);
    }, [load, activeId]);

    const templates = payload?.templates || [];
    const resolvedId = activeId || payload?.template_id || templates[0]?.id || '';
    const activeMeta = templates.find((t) => t.id === resolvedId) || templates[0];
    const kpis = payload?.kpis || [];
    const sections = payload?.sections || [];
    const generatedAt = payload?.report_generated_at;

    return (
        <div className="sample-template">
            <div className="sample-template__toolbar">
                <div>
                    <h2 className="sample-template__heading">Sample report templates</h2>
                    <p className="sample-template__hint">
                        {payload?.hint ||
                            'Preview the report layouts that ship with your generated app. All data is demonstration-only.'}
                    </p>
                </div>
                {onDownloadSampleData && (
                    <button type="button" className="sample-template__dl" onClick={onDownloadSampleData}>
                        Download full data (Excel)
                    </button>
                )}
            </div>

            <div className="sample-template__picker">
                {templates.map((t) => (
                    <button
                        key={t.id}
                        type="button"
                        className={`sample-template__card${resolvedId === t.id ? ' sample-template__card--active' : ''}`}
                        onClick={() => setActiveId(t.id)}
                    >
                        <span className="sample-template__card-title">{t.title}</span>
                        <span className="sample-template__card-sub">{t.subtitle}</span>
                        <span className="sample-template__card-id">{t.id}</span>
                    </button>
                ))}
            </div>

            <div className="sample-template__paper" aria-busy={loading}>
                <div className="sample-template__banner">Sample template — demonstration data only</div>
                <header className="sample-template__doc-head">
                    <div>
                        <div className="sample-template__org">{payload?.organization || 'Your app'}</div>
                        <h3 className="sample-template__doc-title">{activeMeta?.title || payload?.title || 'Report'}</h3>
                        <div className="sample-template__meta">
                            Template <code>{resolvedId}</code>
                            {generatedAt ? ` · Generated ${new Date(generatedAt).toLocaleString()}` : ''}
                        </div>
                    </div>
                    <div className="sample-template__badge">SAMPLE</div>
                </header>

                <section className="sample-template__kpis">
                    {kpis.map((k) => (
                        <div key={k.label}>
                            <span>{k.label}</span>
                            <strong>{k.value}</strong>
                        </div>
                    ))}
                </section>

                {sections.map((section, idx) => (
                    <SectionBlock key={`${section.title}-${idx}`} section={section} />
                ))}

                <footer className="sample-template__footer">
                    Open the generated app → <strong>Reports</strong> for the same templates with live demo API data and CSV export.
                </footer>
            </div>
        </div>
    );
}
