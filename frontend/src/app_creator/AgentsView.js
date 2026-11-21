import React, { useState } from 'react';
import { getFriendlyError } from './utils/errorMessages';
import { formatPipelineAgentName, sortAgentEntries } from './agentPipeline';

const AgentsView = ({
    agents,
    onRegenerate,
    isLoading,
    selectedDomain,
}) => {
    const sapDomain = selectedDomain === 'sap';
    const entries = sortAgentEntries(Object.entries(agents || {}), { sapDomain });
    const [showDetails, setShowDetails] = useState({});

    if (!entries.length) {
        return (
            <div style={{ padding: 40, color: '#666', textAlign: 'center' }}>
                <h3>Production agent pipeline</h3>
                <p>
                    Complete Plan (PRD → UI/UX → Architecture), then click <strong>Generate Code</strong>.
                    You will see Planner, Styling, App Definition, Coding, Integration, and Validator agents
                    stream live progress like a production release.
                </p>
            </div>
        );
    }

    const getStatusColor = (status) => {
        switch (status) {
            case 'running':
                return '#007bff';
            case 'complete':
                return '#28a745';
            case 'error':
                return '#dc3545';
            case 'stopped':
                return '#6b7280';
            case 'pending':
                return '#94a3b8';
            default:
                return '#6c757d';
        }
    };

    const formatDuration = (started, completed) => {
        if (!started) return '';
        if (!completed) return 'In progress...';
        const duration = (completed - started) / 1000;
        return `Completed in ${duration.toFixed(1)}s`;
    };

    const formatAgentName = (name) => {
        const mapping = {
            'planning_agent_step1_req': 'Planning Agent (Requirements)',
            'planning_agent_step2_plan': 'Planning Agent (Plan)',
            'planning_agent_step3_arch': 'Planning Agent (Architecture)',
            'requirement_agent': 'Requirement Agent',
            'structuring_agent': 'Structuring Agent',
            'architecture_agent': 'Architecture Agent',
            'contract_agent': 'Contract Agent',
            'schema_agent': 'Schema Agent',
            'code_generator_agent': 'Code Generation Agent',
            'coding_agent': 'Coding Agent',
            'validation_agent': 'Validation Agent',
            'backend_verify_agent': 'Backend Verify Agent',
            'build_verify_agent': 'Build Verify Agent',
            'functionality_validator_agent': 'Functionality Validator',
            'frontend_agent': 'Frontend Agent',
            'backend_agent': 'Backend Agent',
            'integration_agent': 'Integration Agent',
            'screen_qa_agent': 'Screen QA Agent',
            'mock_fallback_agent': 'Mock Fallback Agent',
            'runner_agent': 'Execution Agent',
            'update_code_agent': 'Code Update Agent',
            // Frontend-only pipeline stages
            'fo_stage_s1': '① Types & Mock Data',
            'fo_stage_s2': '② Theme & Layout',
            'fo_stage_s3': '③ Shared Components',
            'fo_stage_s4': '④ Page Generation',
            'fo_stage_s5': '⑤ Router & Nav',
            'fo_stage_s6': '⑥ Final Verification',
            'fo_pipeline': 'Frontend Pipeline',
            'planner_agent': 'Planner Agent',
            'styling_agent': 'Styling Agent',
            'coding_agent': 'Coding Agent',
            'integration_agent': 'Integration Agent',
            'validator_agent': 'Validator Agent',
            'app_definition_agent': 'App Definition Agent',
            'sap_dashboard_agent': 'Dashboard Agent',
            'sap_data_definition_agent': 'Data Definition Agent',
            'sap_rag_agent': 'RAG Agent',
            'sap_integration_agent': 'Integration Agent',
            'app_definition': 'App Definition Agent',
            'template_agent': 'Template Binding Agent',
            'catalog_agent': 'Catalog & Domain Agent',
            'planning_agent': 'Planning Contract Agent',
            'theme_agent': 'Theme Agent',
            'mock_data_agent': 'Mock Data Agent',
            'code_generator_agent': 'Code Generator Agent',
        };
        if (mapping[name]) return mapping[name];
        const piped = formatPipelineAgentName(name);
        if (piped) return piped;

        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    };

    return (
        <div className="agents-view" style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: '#fafafa'
        }}>
            <div style={{
                backgroundColor: 'white',
                borderBottom: '1px solid #e0e0e0',
                padding: '20px 24px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <div>
                    <h2 style={{ margin: 0, color: '#333', fontSize: '20px' }}>Production build pipeline</h2>
                    <p style={{ margin: '8px 0 0 0', color: '#666', fontSize: '14px' }}>
                        Planner → Styling → App Definition → Coding → Integration → Validator
                    </p>
                </div>
                {onRegenerate && (
                    <button
                        onClick={onRegenerate}
                        disabled={isLoading}
                        style={{
                            padding: '8px 16px',
                            fontSize: '13px',
                            fontWeight: '600',
                            color: '#dc3545',
                            backgroundColor: 'white',
                            border: '1px solid #dc3545',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            opacity: isLoading ? 0.7 : 1
                        }}
                    >
                        {isLoading ? 'Restarting...' : 'Restart Agents'}
                    </button>
                )}
            </div>

            <div style={{
                flex: 1,
                overflowY: 'auto',
                padding: '24px'
            }}>
                <div className="agents-timeline" style={{ position: 'relative', maxWidth: '900px', margin: '0 auto' }}>
                    {entries.map(([name, info], index) => {
                        const status = info?.status || 'pending';
                        const displayStatus = status === 'pending' ? 'pending' : status;
                        const isLast = index === entries.length - 1;
                        const lastProgress = info?.progress?.length
                            ? info.progress[info.progress.length - 1]?.text
                            : '';
                        const runningLabel = lastProgress || info?.message || 'Processing...';

                        return (
                            <div key={name} style={{ position: 'relative', paddingLeft: '48px', paddingBottom: '28px' }}>
                                {/* Timeline line */}
                                {!isLast && (
                                    <div style={{
                                        position: 'absolute',
                                        left: '19px',
                                        top: '40px',
                                        bottom: '0',
                                        width: '2px',
                                        backgroundColor: '#d0d0d0'
                                    }} />
                                )}

                                {/* Timeline dot */}
                                <div style={{
                                    position: 'absolute',
                                    left: '0',
                                    top: '12px',
                                    width: '40px',
                                    height: '40px',
                                    borderRadius: '50%',
                                    backgroundColor: 'white',
                                    border: `3px solid ${getStatusColor(displayStatus === 'pending' ? 'pending' : status)}`,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    color: getStatusColor(status),
                                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                                }}>
                                    <span style={{
                                        width: '10px',
                                        height: '10px',
                                        borderRadius: '50%',
                                        backgroundColor: getStatusColor(status),
                                        display: 'inline-block'
                                    }} />
                                </div>

                                {/* Agent card */}
                                <div style={{
                                    backgroundColor: 'white',
                                    borderRadius: '8px',
                                    padding: '20px',
                                    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                                    borderLeft: `4px solid ${getStatusColor(status)}`
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                        <h4 style={{ margin: 0, color: '#333', fontSize: '16px', fontWeight: '600' }}>
                                            {formatAgentName(name)}
                                        </h4>
                                        <span style={{
                                            fontSize: '11px',
                                            padding: '5px 14px',
                                            borderRadius: '16px',
                                            backgroundColor: getStatusColor(status),
                                            color: 'white',
                                            fontWeight: '600',
                                            textTransform: 'uppercase',
                                            letterSpacing: '0.5px'
                                        }}>
                                            {displayStatus}
                                        </span>
                                    </div>

                                    {/* Streaming progress log */}
                                    {info?.progress && info.progress.length > 0 && (
                                        <div style={{
                                            marginBottom: '12px',
                                            backgroundColor: '#f8f9fa',
                                            borderRadius: '6px',
                                            padding: '12px',
                                            maxHeight: '300px',
                                            overflowY: 'auto',
                                            fontSize: '13px',
                                            fontFamily: 'Monaco, Consolas, monospace',
                                            border: '1px solid #e0e0e0'
                                        }}>
                                            {(info?.progress || []).map((item, idx) => (
                                                <div
                                                    key={idx}
                                                    style={{
                                                        padding: '6px 0',
                                                        color: item.type === 'complete' ? '#28a745' :
                                                            item.type === 'start' ? '#007bff' : '#555',
                                                        display: 'flex',
                                                        alignItems: 'flex-start',
                                                        borderBottom: idx < info.progress.length - 1 ? '1px solid #eee' : 'none',
                                                        animation: idx === info.progress.length - 1 ? 'slideInProgress 0.3s ease-out' : 'none'
                                                    }}
                                                >
                                                    <span style={{
                                                        marginRight: '8px',
                                                        fontSize: '10px',
                                                        color: '#999',
                                                        minWidth: '45px',
                                                        flexShrink: 0
                                                    }}>
                                                        {new Date(item.timestamp).toLocaleTimeString('en-US', {
                                                            hour12: false,
                                                            hour: '2-digit',
                                                            minute: '2-digit',
                                                            second: '2-digit'
                                                        })}
                                                    </span>
                                                    <span style={{
                                                        marginRight: '8px',
                                                        fontSize: '10px',
                                                        fontWeight: 700,
                                                        textTransform: 'uppercase',
                                                        color: '#64748b',
                                                        minWidth: '48px'
                                                    }}>
                                                        {item.type === 'start' ? 'Start' :
                                                            item.type === 'complete' ? 'Done' :
                                                                item.type === 'stopped' ? 'Stopped' :
                                                                item.type === 'stage_file' ? 'Gen' :
                                                                item.type === 'stage_verify' ? 'Verify' :
                                                                item.type === 'stage_fix' ? 'Fix' :
                                                                'Update'}
                                                    </span>
                                                    <span style={{ flex: 1, lineHeight: '1.5' }}>{item.text}</span>
                                                </div>
                                            ))}

                                            {/* Show loading indicator if running */}
                                            {status === 'running' && (
                                                <div style={{
                                                    padding: '8px 0',
                                                    color: '#007bff',
                                                    display: 'flex',
                                                    alignItems: 'flex-start',
                                                    fontSize: '12px',
                                                    lineHeight: '1.5'
                                                }}>
                                                    <span style={{
                                                        display: 'inline-block',
                                                        width: '6px',
                                                        height: '6px',
                                                        borderRadius: '50%',
                                                        backgroundColor: '#007bff',
                                                        marginRight: '8px',
                                                        marginTop: '5px',
                                                        flexShrink: 0,
                                                        animation: 'pulse 1.5s ease-in-out infinite'
                                                    }} />
                                                    <span>{runningLabel}</span>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {info?.error && (
                                        <div style={{
                                            backgroundColor: '#fff3cd',
                                            border: '1px solid #ffc107',
                                            color: '#856404',
                                            padding: '12px',
                                            borderRadius: '6px',
                                            marginTop: '12px',
                                            fontSize: '13px'
                                        }}>
                                            <strong>Error:</strong> {getFriendlyError(info.error)}
                                        </div>
                                    )}

                                    {info?.data && (
                                        <div style={{
                                            marginTop: '12px',
                                            padding: '14px',
                                            backgroundColor: '#f8f9fa',
                                            borderRadius: '8px',
                                            fontSize: '13px',
                                            border: '1px solid #e9ecef',
                                            boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.05)'
                                        }}>
                                            {/* Summary row — always visible */}
                                            <div style={{
                                                display: 'flex',
                                                justifyContent: 'space-between',
                                                alignItems: 'center',
                                            }}>
                                                <span style={{ color: '#495057', fontWeight: '600' }}>
                                                    {info.data.generated_files
                                                        ? `✓ ${Object.keys(info.data.generated_files).length} files generated`
                                                        : info.data.db_schema?.schema
                                                            ? '✓ Schema ready'
                                                            : info.message || 'Completed'}
                                                </span>
                                                <button
                                                    type="button"
                                                    onClick={() => setShowDetails(prev => ({ ...prev, [index]: !prev[index] }))}
                                                    style={{
                                                        background: 'none',
                                                        border: '1px solid #ced4da',
                                                        borderRadius: '4px',
                                                        padding: '2px 10px',
                                                        fontSize: '11px',
                                                        cursor: 'pointer',
                                                        color: '#6c757d'
                                                    }}
                                                >
                                                    {showDetails[index] ? 'Hide Details' : 'View Details'}
                                                </button>
                                            </div>

                                            {/* Collapsible detail panel */}
                                            {showDetails[index] && (
                                                <div style={{
                                                    marginTop: '10px',
                                                    maxHeight: '200px',
                                                    overflowY: 'auto',
                                                    backgroundColor: '#fff',
                                                    padding: '10px',
                                                    borderRadius: '4px',
                                                    border: '1px solid #dee2e6'
                                                }}>
                                                    {(() => {
                                                        const data = info.data;

                                                        // Special handling for coding agent files
                                                        if (data.generated_files) {
                                                            const fileNames = Object.keys(data.generated_files);
                                                            return (
                                                                <div>
                                                                    <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '12px', color: '#666' }}>
                                                                        {fileNames.map(f => <li key={f}>{f}</li>)}
                                                                    </ul>
                                                                </div>
                                                            );
                                                        }

                                                        // Special handling for db_schema
                                                        if (data.db_schema?.schema) {
                                                            return (
                                                                <div style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '12px' }}>
                                                                    {data.db_schema.schema}
                                                                </div>
                                                            );
                                                        }

                                                        // Fallback to JSON or string
                                                        return (
                                                            <pre style={{
                                                                margin: 0,
                                                                whiteSpace: 'pre-wrap',
                                                                wordBreak: 'break-word',
                                                                fontSize: '12px',
                                                                color: '#333',
                                                                fontFamily: 'Monaco, Consolas, monospace'
                                                            }}>
                                                                {typeof data === 'string'
                                                                    ? data
                                                                    : JSON.stringify(data, null, 2)
                                                                }
                                                            </pre>
                                                        );
                                                    })()}
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {(info?.started_at || info?.completed_at) && (
                                        <div style={{
                                            marginTop: '12px',
                                            fontSize: '12px',
                                            color: '#888',
                                            fontStyle: 'italic'
                                        }}>
                                            {formatDuration(info.started_at, info.completed_at)}
                                        </div>
                                    )}

                                </div>
                            </div>
                        );
                    })}
                </div>

                {entries.every(([_, info]) => info?.status === 'complete') && (
                    <div style={{
                        maxWidth: '900px',
                        margin: '24px auto 0 auto',
                        padding: '18px',
                        backgroundColor: '#d4edda',
                        color: '#155724',
                        borderRadius: '8px',
                        textAlign: 'center',
                        fontWeight: '600',
                        fontSize: '15px',
                        border: '1px solid #c3e6cb'
                    }}>
                        All agents completed successfully! Click "View Generated Code" to continue.
                    </div>
                )}
            </div>
        </div>
    );
};

export default AgentsView;

