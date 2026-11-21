/**
 * Canonical Agentic Builder pipeline — IDs match backend agent_pipeline_events.py
 */
export const SAP_AGENT_PIPELINE_ORDER = [
    'sap_dashboard_agent',
    'sap_data_definition_agent',
    'sap_rag_agent',
    'sap_integration_agent',
];

export const AGENT_PIPELINE_ORDER = [
    'planner_agent',
    'styling_agent',
    'app_definition_agent',
    'coding_agent',
    'integration_agent',
    'validator_agent',
    'catalog_agent',
    'frontend_agent',
    'backend_agent',
    'code_generator_agent',
    'template_agent',
    'theme_agent',
    'mock_data_agent',
    'validation_agent',
    'build_verify_agent',
    'backend_verify_agent',
];

export const AGENT_DISPLAY_NAMES = {
    sap_dashboard_agent: 'Dashboard Agent',
    sap_data_definition_agent: 'Data Definition Agent',
    sap_rag_agent: 'RAG Agent',
    sap_integration_agent: 'Integration Agent',
    planner_agent: 'Planner Agent',
    styling_agent: 'Styling Agent',
    app_definition_agent: 'App Definition Agent',
    coding_agent: 'Coding Agent',
    integration_agent: 'Integration Agent',
    validator_agent: 'Validator Agent',
    catalog_agent: 'Catalog Agent',
    frontend_agent: 'Frontend Agent',
    backend_agent: 'Backend Agent',
    code_generator_agent: 'Code Generator Agent',
    app_definition_agent: 'App Definition Agent',
    template_agent: 'Template Agent',
    theme_agent: 'Theme Agent',
    mock_data_agent: 'Mock Data Agent',
    validation_agent: 'Validator Agent',
    build_verify_agent: 'Validator Agent',
    backend_verify_agent: 'Integration Agent',
};

export function formatPipelineAgentName(agentId) {
    if (!agentId) return 'Agent';
    if (AGENT_DISPLAY_NAMES[agentId]) return AGENT_DISPLAY_NAMES[agentId];
    return agentId
        .split('_')
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ');
}

export function sortAgentEntries(entries, { sapDomain = false } = {}) {
    const order = sapDomain ? SAP_AGENT_PIPELINE_ORDER : AGENT_PIPELINE_ORDER;
    return [...entries].sort(([a], [b]) => {
        const ia = order.indexOf(a);
        const ib = order.indexOf(b);
        const ra = ia === -1 ? 999 : ia;
        const rb = ib === -1 ? 999 : ib;
        if (ra !== rb) return ra - rb;
        return a.localeCompare(b);
    });
}

/** Merge WebSocket / planning stream events into agents state map. */
export function mergeAgentEvent(accumulator, data) {
    if (!data?.agent) return accumulator;
    const agent = data.agent;
    const prev = accumulator[agent] || { progress: [] };
    const ts = Date.now();

    if (data.event === 'agent_start') {
        return {
            ...accumulator,
            [agent]: {
                ...prev,
                status: 'running',
                message: data.message,
                progress: [
                    ...(prev.progress || []),
                    { type: 'start', text: data.message, timestamp: ts },
                ],
                started_at: prev.started_at || ts,
            },
        };
    }

    if (data.event === 'agent_progress') {
        return {
            ...accumulator,
            [agent]: {
                ...prev,
                status: prev.status || 'running',
                message: data.message,
                progress: [
                    ...(prev.progress || []),
                    { type: 'progress', text: data.message, timestamp: ts },
                ],
                started_at: prev.started_at || ts,
            },
        };
    }

    if (data.event === 'agent_complete') {
        return {
            ...accumulator,
            [agent]: {
                ...prev,
                status: 'complete',
                message: data.message,
                data: data.data,
                progress: [
                    ...(prev.progress || []),
                    { type: 'complete', text: data.message, timestamp: ts },
                ],
                completed_at: ts,
            },
        };
    }

    if (data.event === 'agent_error') {
        return {
            ...accumulator,
            [agent]: {
                ...prev,
                status: 'error',
                message: data.message || data.error,
                progress: [
                    ...(prev.progress || []),
                    { type: 'error', text: data.message || data.error, timestamp: ts },
                ],
                completed_at: ts,
            },
        };
    }

    return accumulator;
}

const PRODUCTION_AGENT_SEED = {
    planner_agent: {
        status: 'pending',
        message: 'Queued — will ingest PRD, UI/UX, and architecture contracts.',
        progress: [],
    },
    styling_agent: {
        status: 'pending',
        message: 'Queued — will apply brand tokens and chart palette to the shell.',
        progress: [],
    },
    app_definition_agent: {
        status: 'pending',
        message: 'Queued — will synthesize navigation, pages, blocks, and MVP datasets.',
        progress: [],
    },
    coding_agent: {
        status: 'pending',
        message: 'Queued — will materialize Vite workspace and mock/data.json.',
        progress: [],
    },
    integration_agent: {
        status: 'pending',
        message: 'Queued — will wire React Router, AppShell, and JSON renderer.',
        progress: [],
    },
    validator_agent: {
        status: 'pending',
        message: 'Queued — will run npm install + production build verification.',
        progress: [],
    },
};

export function seedCodegenAgentPipeline() {
    return { ...PRODUCTION_AGENT_SEED };
}

/** Full production pipeline (Agentic / json_app track). */
export function seedProductionCodegenPipeline() {
    return { ...PRODUCTION_AGENT_SEED };
}

const SAP_AGENT_SEED = {
    sap_dashboard_agent: {
        status: 'pending',
        message: 'Step 1 — KPIs + Community ticket feed.',
        progress: [],
    },
    sap_data_definition_agent: {
        status: 'pending',
        message: 'Step 1 — ticket schema and seed rows.',
        progress: [],
    },
    sap_rag_agent: {
        status: 'pending',
        message: 'Step 2 — SAP Community search per ticket.',
        progress: [],
    },
    sap_integration_agent: {
        status: 'pending',
        message: 'Wire data.json, chat, and routes.',
        progress: [],
    },
};

export function seedSapCodegenPipeline() {
    return { ...SAP_AGENT_SEED };
}

