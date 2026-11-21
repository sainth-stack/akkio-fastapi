import React, { useState, useRef, useEffect } from 'react';
import ModelSelector from './ModelSelector';
import DomainSelector from './DomainSelector';
import SamplePromptTemplates from './SamplePromptTemplates';
import { IconBadge, IoSend, IoSparkles, FaPenToSquare } from './AppBuilderIcons';

const ChatInterface = ({
    onSendMessage,
    messages,
    chatState,
    isInputDisabled,
    isUpdateMode,
    isUpdateCodeInProgress,
    selectedModel,
    onModelChange,
    showDomainSelector,
    selectedDomain,
    promptTemplatesDomain,
    onDomainChange,
    title,
    updateTitle,
    welcomeMessage,
}) => {
    const [input, setInput] = useState("");
    const textareaRef = useRef(null);
    const messagesEndRef = useRef(null);

    const showThinking = chatState === 'waiting';
    const isStreaming = chatState === 'streaming';
    // In update mode, only block input during an active update — not stale chatState after codegen
    const inputDisabled = isInputDisabled
        || (isUpdateMode
            ? isUpdateCodeInProgress
            : (chatState === 'waiting' || chatState === 'streaming'));

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, chatState]);

    useEffect(() => {
        const ta = textareaRef.current;
        if (!ta) return;
        ta.style.height = 'auto';
        const newHeight = Math.min(ta.scrollHeight, 120);
        ta.style.height = newHeight + 'px';
        ta.style.overflowY = ta.scrollHeight > 120 ? 'auto' : 'hidden';
    }, [input]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!input.trim() || inputDisabled) return;
        onSendMessage(input.trim());
        setInput("");
        if (textareaRef.current) textareaRef.current.style.height = 'auto';
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <div className="chat-section">
            <div className="chat-header">
                <div className="chat-header-title">
                    <IconBadge
                        icon={isUpdateMode ? FaPenToSquare : IoSparkles}
                        variant={isUpdateMode ? 'amber' : 'indigo'}
                        size={15}
                    />
                    <h2>{isUpdateMode ? (updateTitle || 'Update App') : (title || 'App Builder')}</h2>
                </div>
            </div>

            <div className="chat-sidebar-tools">
                {showDomainSelector && (
                    <DomainSelector
                        value={selectedDomain}
                        onChange={onDomainChange}
                        disabled={inputDisabled}
                        required
                    />
                )}
                <ModelSelector
                    value={selectedModel}
                    onChange={onModelChange}
                    disabled={inputDisabled}
                />
                <SamplePromptTemplates
                    selectedDomain={promptTemplatesDomain || selectedDomain}
                    disabled={inputDisabled}
                    onSelect={(text) => {
                        setInput(text);
                        requestAnimationFrame(() => textareaRef.current?.focus());
                    }}
                />
            </div>

            <div className="chat-messages">
                {messages.length === 0 && (
                    <div className="message ai">
                        {isUpdateMode
                            ? "Your app is ready! Describe any changes or new features you'd like to add."
                            : (welcomeMessage || "Hi! I'm your App Architect. Describe the app you want to build, and I'll generate the plan and code for you.")
                        }
                    </div>
                )}
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        {msg.content}
                        {isStreaming && msg.role === 'ai' && idx === messages.map(m => m.role).lastIndexOf('ai') && (
                            <span style={{ display: 'inline-block', width: 6, height: 14, marginLeft: 2, background: '#3b82f6', animation: 'pulse 1s infinite' }} />
                        )}
                    </div>
                ))}
                {showThinking && (
                    <div className="message ai" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{
                            display: 'inline-block',
                            width: 14,
                            height: 14,
                            border: '2px solid #cbd5e1',
                            borderTopColor: '#3b82f6',
                            borderRadius: '50%',
                            animation: 'spin 0.7s linear infinite',
                            flexShrink: 0
                        }} />
                        {(isUpdateMode && isUpdateCodeInProgress) ? 'Updating code...' : 'Thinking...'}
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-area">
                <form onSubmit={handleSubmit} className="chat-input-wrapper">
                    <textarea
                        ref={textareaRef}
                        className="chat-input"
                        placeholder={isUpdateMode ? "Describe what you'd like to change or add..." : "Describe your app..."}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        rows={1}
                        disabled={inputDisabled}
                    />
                    <button type="submit" className="send-button" disabled={inputDisabled || !input.trim()} title="Send (Enter)">
                        <IoSend size={16} />
                    </button>
                </form>
                {!isUpdateMode && (
                    <p style={{ margin: '6px 0 0', fontSize: '11px', color: '#94a3b8', textAlign: 'center' }}>
                        Press Enter to send · Shift+Enter for new line
                    </p>
                )}
            </div>
        </div>
    );
};

export default ChatInterface;
