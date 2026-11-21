import React, { useEffect, useRef } from 'react';

const LogsView = ({ logs, runState }) => {
    const bottomRef = useRef(null);

    useEffect(() => {
        if (!bottomRef.current) return;
        bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    return (
        <div className="logs-view">
            <div className="logs-header">
                <div>
                    <strong>Logs</strong>
                    {runState?.frontend_url ? (
                        <span style={{ marginLeft: 10, color: '#666' }}>
                            App: {runState.frontend_url}
                        </span>
                    ) : null}
                </div>
            </div>
            <div className="logs-console">
                {(logs || []).map((line, idx) => (
                    <div key={idx} className="log-line">{line}</div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default LogsView;

