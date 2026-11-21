/**
 * SAP Similarity Tickets — two short prompts (step 1 dashboard, step 2 RAG).
 */
import { AGENTIC_DOMAIN_SAP } from '../agenticDomains';

export const SAP_PROMPT_STEP1 = `Build SAP Similarity Tickets app with Dashboard only.

On the dashboard show KPIs and a simple table of 10 SAP tickets (no extra charts or community panels).

Seed 10 tickets. Use the app chat assistant.

We will load SAP Community RAG in step 2.`;

export const SAP_PROMPT_STEP2 = `Step 2 — load SAP ticket data and context.

For each ticket, search SAP Community and build per-ticket RAG (first hit + suggested answer).

Refresh the dashboard and assistant so answers use that live context.`;

export const SAP_PROMPT_TEMPLATES = [
    {
        id: 'sap_step1_dashboard',
        title: 'Step 1 — Dashboard',
        subtitle: 'Send first, then Generate Code',
        step: 1,
        prompt: SAP_PROMPT_STEP1,
    },
    {
        id: 'sap_step2_rag',
        title: 'Step 2 — SAP data & RAG',
        subtitle: 'Send after app exists (triggers RAG agent)',
        step: 2,
        prompt: SAP_PROMPT_STEP2,
    },
];

export function sapTemplatesForDomain(domainId) {
    return domainId === AGENTIC_DOMAIN_SAP ? SAP_PROMPT_TEMPLATES : [];
}