import React, { useEffect, useMemo, useRef, useState } from 'react';
import api from '../utils/api';
import ChatSidebarSection from './ChatSidebarSection';
import {
    IoChevronDown,
    IoCheckmark,
    FaStar,
    TIER_ICONS,
} from './AppBuilderIcons';

const TIER_STYLES = {
    flagship: { bg: '#eef2ff', color: '#4338ca', border: '#c7d2fe' },
    balanced: { bg: '#ecfdf5', color: '#047857', border: '#bbf7d0' },
    fast: { bg: '#fff7ed', color: '#c2410c', border: '#fed7aa' },
    reasoning: { bg: '#f5f3ff', color: '#6d28d9', border: '#ddd6fe' },
    legacy: { bg: '#f1f5f9', color: '#475569', border: '#e2e8f0' },
};

const TierBadge = ({ tier, tiers }) => {
    const style = TIER_STYLES[tier] || TIER_STYLES.balanced;
    const label = tiers?.[tier] || tier;
    return (
        <span
            className="model-tier-badge"
            style={{ background: style.bg, color: style.color, borderColor: style.border }}
        >
            {label}
        </span>
    );
};

const ModelSelector = ({ value, onChange, disabled }) => {
    const [models, setModels] = useState([]);
    const [tiers, setTiers] = useState({});
    const [loading, setLoading] = useState(true);
    const [open, setOpen] = useState(false);
    const rootRef = useRef(null);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const res = await api.get('/app-builder/models');
                const data = res.data;
                if (!cancelled && data?.models) {
                    setModels(data.models);
                    setTiers(data.tiers || {});
                    if (!value && data.default_model) {
                        onChange?.(data.default_model);
                    }
                }
            } catch (e) {
                console.error('Failed to load models:', e);
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        return () => { cancelled = true; };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
        const onDocClick = (e) => {
            if (rootRef.current && !rootRef.current.contains(e.target)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', onDocClick);
        return () => document.removeEventListener('mousedown', onDocClick);
    }, []);

    const selected = useMemo(
        () => models.find((m) => m.id === value) || models.find((m) => m.recommended) || models[0],
        [models, value]
    );

    const grouped = useMemo(() => {
        const order = ['flagship', 'balanced', 'reasoning', 'fast', 'legacy'];
        const map = {};
        models.forEach((m) => {
            const key = m.tier || 'balanced';
            if (!map[key]) map[key] = [];
            map[key].push(m);
        });
        return order.filter((k) => map[k]?.length).map((k) => ({ tier: k, items: map[k] }));
    }, [models]);

    const handleSelect = (id) => {
        onChange?.(id);
        setOpen(false);
    };

    const summaryLabel = loading
        ? 'Loading…'
        : (selected?.label || 'Select model');

    return (
        <ChatSidebarSection title="Model" summary={summaryLabel} defaultOpen={false}>
            <div className="model-selector model-selector--compact" ref={rootRef}>
                <button
                    type="button"
                    className={`model-selector-trigger model-selector-trigger--compact ${open ? 'open' : ''}`}
                    onClick={() => !disabled && !loading && setOpen((v) => !v)}
                    disabled={disabled || loading}
                    aria-haspopup="listbox"
                    aria-expanded={open}
                >
                    <span className="model-selector-current-name">
                        {loading ? 'Loading models…' : (selected?.label || 'Select model')}
                    </span>
                    <div className="model-selector-trigger-meta model-selector-trigger-meta--inline">
                        {selected?.tier && !loading && (
                            <TierBadge tier={selected.tier} tiers={tiers} />
                        )}
                        <IoChevronDown className={`model-selector-chevron ${open ? 'up' : ''}`} size={14} aria-hidden />
                    </div>
                </button>

                {open && !loading && (
                    <div className="model-selector-menu" role="listbox">
                    {grouped.map(({ tier, items }) => (
                        <div key={tier} className="model-selector-group">
                            <div className="model-selector-group-label">
                                {tiers[tier] || tier}
                            </div>
                            {items.map((m) => {
                                const active = m.id === (value || selected?.id);
                                const TierIcon = TIER_ICONS[m.tier] || TIER_ICONS.balanced;
                                return (
                                    <button
                                        key={m.id}
                                        type="button"
                                        role="option"
                                        aria-selected={active}
                                        className={`model-selector-option ${active ? 'active' : ''}`}
                                        onClick={() => handleSelect(m.id)}
                                    >
                                        <span className="model-selector-option-icon" aria-hidden>
                                            <TierIcon size={15} />
                                        </span>
                                        <div className="model-selector-option-text">
                                            <span className="model-selector-option-name">
                                                {m.label}
                                                {m.recommended && (
                                                    <FaStar className="model-option-star" size={11} aria-label="Recommended" />
                                                )}
                                            </span>
                                            <span className="model-selector-option-desc">
                                                {m.description}
                                            </span>
                                        </div>
                                        <div className="model-selector-option-meta">
                                            {active && <IoCheckmark className="model-selector-check" size={16} aria-hidden />}
                                            <TierBadge tier={m.tier} tiers={tiers} />
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    ))}
                    </div>
                )}
            </div>
        </ChatSidebarSection>
    );
};

export default ModelSelector;
