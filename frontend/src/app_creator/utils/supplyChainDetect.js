/** Detect inventory / supply-chain apps from plan text or PRD for sample template UI. */
export function isSupplyChainContext(plan, prd, requirement = '') {
    const parts = [requirement, prd];
    if (plan && typeof plan === 'object') {
        try {
            parts.push(JSON.stringify(plan));
        } catch {
            /* ignore */
        }
    } else if (plan) {
        parts.push(String(plan));
    }
    const text = parts.join(' ').toLowerCase();
    const keywords = [
        'supply chain',
        'inventory management',
        'inventory level',
        'stock-out',
        'stock out',
        'reorder',
        'purchase order',
        'safety stock',
        'supplier lead',
        'replenishment',
        'inventory aging',
        'turnover report',
        'excess stock',
        'warehouse',
        'material receipt',
    ];
    return keywords.some((k) => text.includes(k));
}
