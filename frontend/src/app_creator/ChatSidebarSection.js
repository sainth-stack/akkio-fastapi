import React, { useState } from 'react';
import { IoChevronDown } from './AppBuilderIcons';

/**
 * Compact collapsible block for the chat sidebar (SaaS-style).
 */
export default function ChatSidebarSection({
    title,
    summary,
    defaultOpen = false,
    children,
    className = '',
}) {
    const [open, setOpen] = useState(defaultOpen);

    return (
        <div className={`chat-sidebar-section ${open ? 'chat-sidebar-section--open' : ''} ${className}`.trim()}>
            <button
                type="button"
                className="chat-sidebar-section__toggle"
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
            >
                <span className="chat-sidebar-section__title">{title}</span>
                {!open && summary && (
                    <span className="chat-sidebar-section__summary">{summary}</span>
                )}
                <IoChevronDown
                    className={`chat-sidebar-section__chevron${open ? ' chat-sidebar-section__chevron--up' : ''}`}
                    size={14}
                    aria-hidden
                />
            </button>
            {open && <div className="chat-sidebar-section__body">{children}</div>}
        </div>
    );
}
