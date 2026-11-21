/** Detect SAP Similarity Tickets apps even when App domain dropdown is not SAP. */

export function definitionLooksLikeSap(defn) {
    if (!defn || typeof defn !== 'object') return false;
    const meta = defn.meta || {};
    if (String(meta.domain || '').toLowerCase() === 'sap') return true;
    if ((meta.capabilities || []).includes('sap_similar_tickets')) return true;
    const name = String(meta.appName || '').toLowerCase();
    if (name.includes('sap') && name.includes('similar')) return true;
    const rows = defn.data?.source_sap_tickets;
    if (Array.isArray(rows) && rows.length) return true;
    for (const p of defn.pages || []) {
        for (const b of p.blocks || []) {
            if (b?.type === 'sapCommunityFeed') return true;
        }
    }
    return false;
}

export function definitionLooksLikeSapFromFiles(files) {
    const raw = files?.['frontend/src/mock/data.json'];
    if (!raw || typeof raw !== 'string') return false;
    try {
        return definitionLooksLikeSap(JSON.parse(raw));
    } catch {
        return false;
    }
}

export function textLooksLikeSapStep2(text) {
    const low = String(text || '').toLowerCase();
    if (/\b(step\s*2|prompt\s*2|second prompt)\b/.test(low)) return true;
    if (low.includes('load sap ticket')) return true;
    if (/\bper-ticket\b/.test(low) && low.includes('rag')) return true;
    return false;
}

export function textLooksLikeSapApp(text) {
    const low = String(text || '').toLowerCase();
    return (
        /\bsap\b.*\b(similar|similarity)\b/.test(low)
        || /\bsimilarity tickets\b/.test(low)
        || (low.includes('sap community') && low.includes('ticket'))
    );
}