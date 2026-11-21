import React, { useState, useEffect } from 'react';
import Spinner from 'react-bootstrap/Spinner';
import { FaPlay, FaCheckCircle, FaExclamationCircle, FaTerminal, FaList } from 'react-icons/fa';
import api from '../utils/api';

const TestView = ({ projectName }) => {
    const [tests, setTests] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [output, setOutput] = useState('');
    const [lastRunStatus, setLastRunStatus] = useState(null);

    useEffect(() => {
        fetchTests();
    }, [projectName]);

    const fetchTests = async () => {
        if (!projectName) return;
        try {
            const res = await api.get(`/test-suite/list/${projectName}`);
            setTests(res.data.tests || []);
        } catch (err) {
            console.error("Failed to fetch tests", err);
        }
    };

    const handleGenerateTests = async () => {
        setIsLoading(true);
        try {
            const res = await api.post('/test-suite/generate', { project_name: projectName });
            if (res.status === 200) {
                await fetchTests();
            } else {
                alert(`Error: ${res.data?.detail}`);
            }
        } catch (err) {
            alert(err.response?.data?.detail || "Failed to generate tests");
        } finally {
            setIsLoading(false);
        }
    };

    const handleRunTests = async (testFile = null) => {
        setIsRunning(true);
        setOutput('');
        setLastRunStatus(null);
        try {
            const body = { project_name: projectName };
            if (testFile) {
                body.test_files = [testFile];
            }
            try {
                const token = localStorage.getItem('access_token') || '';
                if (token) body.access_token = token;
            } catch {
                /* ignore */
            }

            const res = await api.post('/test-suite/run', body);
            const data = res.data;
            setOutput(data.output);
            setLastRunStatus(data.success ? 'success' : 'fail');
        } catch (err) {
            setOutput(`Error running tests: ${err.message}`);
            setLastRunStatus('fail');
        } finally {
            setIsRunning(false);
        }
    };

    if (!projectName) {
        return <div style={{ padding: 20, color: '#666' }}>Generate an app first.</div>;
    }

    return (
        <div className="test-view" style={{ padding: '20px', height: '100%', display: 'flex', flexDirection: 'column', gap: '20px' }}>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <FaList /> Test Suite
                    </h3>
                    <p style={{ margin: '5px 0 0', color: '#666', fontSize: '13px' }}>
                        {tests.length > 0 ? `${tests.length} UI e2e scripts available` : 'No UI tests generated yet'}
                    </p>
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                    {tests.length === 0 ? (
                        <button
                            onClick={handleGenerateTests}
                            disabled={isLoading}
                            className="btn btn-primary"
                            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                        >
                            {isLoading ? <Spinner size="sm" animation="border" /> : null}
                            Generate UI Tests
                        </button>
                    ) : (
                        <>
                            <button
                                onClick={handleGenerateTests}
                                disabled={isLoading}
                                className="btn btn-outline-secondary"
                                style={{ fontSize: '13px' }}
                            >
                                {isLoading ? 'Regenerating...' : 'Regenerate UI Tests'}
                            </button>
                            <button
                                onClick={() => handleRunTests()}
                                disabled={isRunning}
                                className="btn btn-success"
                                style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                            >
                                {isRunning ? <Spinner size="sm" animation="border" /> : <FaPlay size={12} />}
                                Run All Tests
                            </button>
                        </>
                    )}
                </div>
            </div>

            <div style={{ display: 'flex', gap: '20px', flex: 1, minHeight: 0 }}>
                <div style={{ flex: '0 0 300px', borderRight: '1px solid #eee', overflowY: 'auto' }}>
                    {tests.length > 0 ? (
                        <div className="list-group">
                            {tests.map(test => (
                                <div key={test} className="list-group-item" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px' }}>
                                    <span style={{ fontSize: '14px', fontFamily: 'monospace' }}>{test}</span>
                                    <button
                                        onClick={() => handleRunTests(test)}
                                        disabled={isRunning}
                                        className="btn btn-sm btn-light"
                                        title="Run this test file"
                                    >
                                        <FaPlay size={10} color="#28a745" />
                                    </button>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div style={{ padding: '20px', textAlign: 'center', color: '#999', fontSize: '13px' }}>
                            No UI tests found. Build the app in App View, then click &quot;Generate UI Tests&quot;.
                        </div>
                    )}
                </div>

                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', backgroundColor: '#1e1e1e', borderRadius: '8px', overflow: 'hidden' }}>
                    <div style={{
                        padding: '10px 15px',
                        backgroundColor: '#2d2d2d',
                        borderBottom: '1px solid #333',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                    }}>
                        <span style={{ color: '#fff', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <FaTerminal /> Test Output
                        </span>
                        {lastRunStatus && (
                            <span style={{
                                fontSize: '12px',
                                fontWeight: 'bold',
                                color: lastRunStatus === 'success' ? '#4caf50' : '#f44336',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px'
                            }}>
                                {lastRunStatus === 'success' ? <FaCheckCircle /> : <FaExclamationCircle />}
                                {lastRunStatus === 'success' ? 'PASSED' : 'FAILED'}
                            </span>
                        )}
                    </div>
                    <pre style={{
                        flex: 1,
                        margin: 0,
                        padding: '15px',
                        color: '#d4d4d4',
                        fontFamily: 'Consolas, Monaco, monospace',
                        fontSize: '13px',
                        overflow: 'auto',
                        whiteSpace: 'pre-wrap'
                    }}>
                        {isRunning ? 'Running tests...\n' : ''}
                        {output || (tests.length > 0 ? 'Ready to run Playwright UI tests (app must be built).' : 'Waiting for UI test generation...')}
                    </pre>
                </div>
            </div>
        </div>
    );
};

export default TestView;
