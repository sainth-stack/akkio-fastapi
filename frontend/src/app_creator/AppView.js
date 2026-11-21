import React, { useEffect, useRef, useState } from 'react';

import Spinner from 'react-bootstrap/Spinner';
import { getFriendlyError } from './utils/errorMessages';

const AppView = ({
    projectName,
    runState,
    isRunLoading,
    logs,
    appRefreshKey = 0,
    buildStatus,
    buildError,
    buildLog,
    onRetryRun,
}) => {
    const previewUrlWithAuth = () => {
        const base = runState?.frontend_url;
        if (!base) return null;
        try {
            const token = localStorage.getItem('access_token') || '';
            if (!token) return base;
            const sep = base.includes('?') ? '&' : '?';
            return `${base}${sep}access_token=${encodeURIComponent(token)}`;
        } catch {
            return base;
        }
    };

    const [showBuildLog, setShowBuildLog] = useState(false);
    const [previewBlocked, setPreviewBlocked] = useState(false);
    const previewLoadedRef = useRef(false);
    const iframeSrc = previewUrlWithAuth();

    useEffect(() => {
        if (!iframeSrc) return undefined;
        previewLoadedRef.current = false;
        setPreviewBlocked(false);
        const t = window.setTimeout(() => {
            if (!previewLoadedRef.current) setPreviewBlocked(true);
        }, 12000);
        return () => window.clearTimeout(t);
    }, [iframeSrc, appRefreshKey]);

    const onIframeLoad = () => {
        previewLoadedRef.current = true;
        setPreviewBlocked(false);
    };
    if (!projectName) {
        return (
            <div style={{ padding: 20, color: '#666' }}>
                Generate an app first.
            </div>
        );
    }

    // Prefer showing the live preview as soon as we have a URL (don't wait for stream to close)
    if (iframeSrc && buildStatus !== 'BUILD_FAILED' && (buildStatus === 'BUILD_SUCCESS' || !isRunLoading)) {
        return (
            <div className="app-view">
                <div className="app-view-header">
                    <div className="app-view-title">
                        Running: <span style={{ fontFamily: 'monospace' }}>{runState.frontend_url}</span>
                    </div>
                    <div className="app-view-actions">
                        <a href={iframeSrc} target="_blank" rel="noreferrer">
                            Open in new tab
                        </a>
                    </div>
                </div>
                {previewBlocked ? (
                    <div
                        className="px-3 py-2 small"
                        style={{ background: '#fef3c7', color: '#92400e', borderBottom: '1px solid #fcd34d' }}
                    >
                        Preview may not have loaded (port 8000 unreachable or asset still building). Use{' '}
                        <strong>Open in new tab</strong> or confirm the API on port 8000 is running and firewall allows it.
                    </div>
                ) : null}
                <iframe
                    key={appRefreshKey}
                    title="Generated App"
                    src={iframeSrc}
                    className="app-iframe"
                    onLoad={onIframeLoad}
                />
            </div>
        );
    }

    if (buildStatus === 'BUILD_FAILED') {
        const failureLog = buildLog || (logs && logs.length ? logs.join('\n') : '');
        return (
            <div style={{ padding: 20, color: '#334155' }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 12,
                    padding: '16px',
                    backgroundColor: '#fef2f2',
                    border: '1px solid #fecaca',
                    borderRadius: 8,
                    marginBottom: 16,
                }}>
                    <span style={{ fontSize: 20 }}>⚠️</span>
                    <div style={{ flex: 1 }}>
                        <h5 style={{ color: '#b91c1c', margin: '0 0 6px 0', fontSize: 15 }}>Build failed</h5>
                        <p style={{ margin: 0, fontSize: 14, color: '#7f1d1d' }}>
                            {getFriendlyError(buildError)}
                        </p>
                    </div>
                </div>

                {failureLog && (
                    <div style={{ marginBottom: 16 }}>
                        <button
                            type="button"
                            onClick={() => setShowBuildLog(prev => !prev)}
                            style={{
                                background: 'none',
                                border: '1px solid #cbd5e1',
                                borderRadius: 4,
                                padding: '4px 12px',
                                fontSize: 12,
                                cursor: 'pointer',
                                color: '#475569',
                                marginBottom: 8,
                            }}
                        >
                            {showBuildLog ? 'Hide technical details' : 'Show technical details'}
                        </button>
                        {showBuildLog && (
                            <pre style={{
                                backgroundColor: '#1e1e1e',
                                color: '#d4d4d4',
                                padding: 16,
                                borderRadius: 8,
                                fontSize: 12,
                                maxHeight: 320,
                                overflow: 'auto',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                            }}>
                                {failureLog}
                            </pre>
                        )}
                    </div>
                )}

                <p style={{ marginTop: 8, fontSize: 13, color: '#64748b' }}>
                    Click <strong>Run App</strong> in the Build tab header to try again.
                </p>
                {onRetryRun && (
                    <button
                        type="button"
                        onClick={onRetryRun}
                        style={{
                            marginTop: 12,
                            padding: '8px 16px',
                            backgroundColor: '#007bff',
                            color: '#fff',
                            border: 'none',
                            borderRadius: 4,
                            cursor: 'pointer',
                            fontWeight: 600,
                        }}
                    >
                        Retry Run
                    </button>
                )}
            </div>
        );
    }

    if (isRunLoading) {
        const recentLogs = logs && logs.length > 0 ? logs.slice(-5) : ["Starting environment..."];
        return (
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%',
                backgroundColor: '#f8f9fa',
                padding: '20px'
            }}>
                <div style={{
                    marginBottom: '20px',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '12px'
                }}>
                    <Spinner animation="border" role="status" variant="primary" style={{ width: '3rem', height: '3rem' }}>
                        <span className="visually-hidden">Loading...</span>
                    </Spinner>
                    <h5 style={{ color: '#495057', fontSize: '18px', fontWeight: '500' }}>Starting Application...</h5>
                </div>

                <div style={{
                    maxWidth: '600px',
                    width: '100%',
                    backgroundColor: '#1e1e1e',
                    borderRadius: '8px',
                    padding: '16px',
                    color: '#d4d4d4',
                    fontFamily: 'Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace',
                    fontSize: '13px',
                    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                    maxHeight: '200px',
                    overflowY: 'hidden'
                }}>
                    <div style={{
                        borderBottom: '1px solid #333',
                        paddingBottom: '8px',
                        marginBottom: '8px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                    }}>
                        <span style={{ color: '#9cdcfe', fontSize: '11px', fontWeight: 'bold', textTransform: 'uppercase' }}>System Logs</span>
                        <div style={{ display: 'flex', gap: '4px' }}>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#ff5f56' }}></div>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#ffbd2e' }}></div>
                            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#27c93f' }}></div>
                        </div>
                    </div>
                    {recentLogs.map((log, index) => (
                        <div key={index} style={{
                            marginBottom: '4px',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            opacity: index === recentLogs.length - 1 ? 1 : 0.7
                        }}>
                            <span style={{ color: '#569cd6', marginRight: '8px' }}>&gt;</span>
                            {log.replace(/\[system\]/i, '').trim()}
                        </div>
                    ))}
                    <div style={{ marginTop: '8px', animation: 'blink 1s step-end infinite' }}>
                        <span style={{ color: '#27c93f' }}>_</span>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div style={{ padding: 20, color: '#666' }}>
            Click <strong>Run App</strong> in the Build tab header to build and preview your generated app.
        </div>
    );
};

export default AppView;
