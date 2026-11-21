import React from 'react';
import ChatSidebarSection from './ChatSidebarSection';
import { AGENTIC_DOMAIN_SAP } from './agenticDomains';
import { SAMPLE_PROMPT_TEMPLATES } from './utils/samplePromptTemplates';
import { sapTemplatesForDomain } from './utils/sapPromptTemplates';
import './SamplePromptTemplates.css';

export default function SamplePromptTemplates({ onSelect, disabled, selectedDomain }) {
    const templates =
        selectedDomain === AGENTIC_DOMAIN_SAP
            ? sapTemplatesForDomain(AGENTIC_DOMAIN_SAP)
            : SAMPLE_PROMPT_TEMPLATES;
    const count = templates.length;

    return (
        <ChatSidebarSection
            title="Prompt templates"
            summary={`${count} samples`}
            defaultOpen={false}
        >
            <p className="sample-prompt-templates__hint">
                {selectedDomain === AGENTIC_DOMAIN_SAP
                    ? 'SAP: use Step 1 before Generate Code, then Step 2 in chat after the app is built.'
                    : 'Read the sample below, then use it in the chat input.'}
            </p>
            <div className="sample-prompt-templates__list">
                {templates.map((t) => (
                    <article key={t.id} className="sample-prompt-templates__card">
                        <header className="sample-prompt-templates__card-head">
                            <div>
                                <h3 className="sample-prompt-templates__item-title">{t.title}</h3>
                                <p className="sample-prompt-templates__item-sub">{t.subtitle}</p>
                            </div>
                            <button
                                type="button"
                                className="sample-prompt-templates__use"
                                disabled={disabled}
                                onClick={() => onSelect && onSelect(t.prompt)}
                            >
                                Use
                            </button>
                        </header>
                        <pre className="sample-prompt-templates__preview">{t.prompt}</pre>
                    </article>
                ))}
            </div>
        </ChatSidebarSection>
    );
}
